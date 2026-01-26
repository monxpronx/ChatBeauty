"""
Unified LLM client supporting Ollama and vLLM (direct GPU) backends.

Usage:
    from ml.utils.llm_client import LLMClient

    # Ollama (default)
    client = LLMClient(backend="ollama")

    # vLLM Direct (auto GPU detection)
    client = LLMClient(backend="vllm")

    # vLLM with specific GPUs
    client = LLMClient(backend="vllm", gpu_ids=[0, 1])

    # Generate
    response = client.generate(prompt="Hello, world!")
"""

import json
import requests
from typing import Optional, Literal, List, Union
from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Optional[str]]:
        """Generate text from multiple prompts"""
        pass

    @abstractmethod
    def verify_connection(self) -> bool:
        """Verify the backend is available"""
        pass


class OllamaBackend(BaseLLMBackend):
    """Ollama API backend"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.3,
        top_p: float = 0.9,
        timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.api_endpoint = f"{self.base_url}/api/generate"

    def verify_connection(self) -> bool:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if any(self.model in name for name in model_names):
                    print(f"[Ollama] Connected, model '{self.model}' available")
                    return True
                print(f"[Ollama] Model '{self.model}' not found. Available: {model_names}")
                return False
            return False
        except Exception as e:
            print(f"[Ollama] Connection failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Ollama API"""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return None

        except Exception as e:
            print(f"[Ollama] Generation error: {e}")
            return None

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Optional[str]]:
        """Generate text for multiple prompts (sequential for Ollama)"""
        return [self.generate(p, **kwargs) for p in prompts]


class VLLMDirectBackend(BaseLLMBackend):
    """vLLM Direct backend - loads model directly on GPU"""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        gpu_ids: Optional[List[int]] = None,
        gpu_memory_utilization: float = 0.8,
    ):
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.gpu_ids = gpu_ids
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._sampling_params = None
        self._init_failed = False  # Track if initialization already failed

        # IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch/vllm
        if self.gpu_ids:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            print(f"[vLLM] Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    def _init_model(self):
        """Initialize vLLM model (lazy loading)"""
        if self._llm is not None:
            return

        # Don't retry if initialization already failed
        if self._init_failed:
            raise RuntimeError("[vLLM] Model initialization previously failed. Restart the process to retry.")

        try:
            import torch
            from vllm import LLM, SamplingParams
        except ImportError:
            self._init_failed = True
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        # Detect GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"[vLLM] Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self._init_failed = True
            raise RuntimeError("[vLLM] No GPU available. vLLM requires CUDA.")

        # Determine tensor parallel size
        if self.gpu_ids:
            tensor_parallel_size = len(self.gpu_ids)
            print(f"[vLLM] Using GPUs: {self.gpu_ids}")
        else:
            tensor_parallel_size = gpu_count

        # Initialize vLLM with version-safe parameters
        print(f"[vLLM] Loading model: {self.model_name}")
        try:
            # Check vLLM version for API compatibility
            import vllm
            vllm_version = getattr(vllm, "__version__", "0.0.0")
            print(f"[vLLM] vLLM version: {vllm_version}")

            # Base kwargs that work across versions
            llm_kwargs = {
                "model": self.model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
            }

            self._llm = LLM(**llm_kwargs)
        except TypeError as e:
            # Handle API changes between vLLM versions
            if "unexpected keyword argument" in str(e):
                print(f"[vLLM] Warning: API compatibility issue, trying minimal config: {e}")
                try:
                    self._llm = LLM(
                        model=self.model_name,
                        trust_remote_code=True,
                    )
                except Exception as e2:
                    self._init_failed = True
                    raise RuntimeError(f"[vLLM] Failed with minimal config: {e2}")
            else:
                self._init_failed = True
                raise RuntimeError(f"[vLLM] Failed to initialize model: {e}")
        except Exception as e:
            self._init_failed = True
            raise RuntimeError(f"[vLLM] Failed to initialize model: {e}")

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        print(f"[vLLM] Model loaded successfully")

    def verify_connection(self) -> bool:
        """Verify GPU is available and model can be loaded"""
        try:
            import torch
            if not torch.cuda.is_available():
                print("[vLLM] No GPU available")
                return False

            gpu_count = torch.cuda.device_count()
            print(f"[vLLM] {gpu_count} GPU(s) available")
            return True

        except Exception as e:
            print(f"[vLLM] Verification failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using vLLM"""
        results = self.generate_batch([prompt], **kwargs)
        return results[0] if results else None

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Optional[str]]:
        """Generate text for multiple prompts (batched for efficiency)"""
        # Initialize model (will raise if init failed before)
        self._init_model()

        try:
            from vllm import SamplingParams

            # Override sampling params if provided
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            outputs = self._llm.generate(prompts, sampling_params)

            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append(None)

            return results

        except Exception as e:
            print(f"[vLLM] Generation error: {e}")
            return [None] * len(prompts)


class LLMClient:
    """
    Unified LLM client that supports multiple backends.

    Args:
        backend: "ollama" or "vllm"
        base_url: Server URL for Ollama (ignored for vllm)
        model: Model name
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens (vllm only)
        gpu_ids: Specific GPU IDs to use (vllm only)
        gpu_memory_utilization: GPU memory fraction (vllm only)
    """

    def __init__(
        self,
        backend: Literal["ollama", "vllm"] = "ollama",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512,
        gpu_ids: Optional[List[int]] = None,
        gpu_memory_utilization: float = 0.8,
        timeout: int = 60
    ):
        self.backend_name = backend

        if backend == "ollama":
            self._backend = OllamaBackend(
                base_url=base_url or "http://localhost:11434",
                model=model or "llama3.1:8b",
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            )
        elif backend == "vllm":
            self._backend = VLLMDirectBackend(
                model=model or "meta-llama/Llama-3.1-8B-Instruct",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                gpu_ids=gpu_ids,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'vllm'")

    def verify_connection(self) -> bool:
        """Verify backend connection"""
        return self._backend.verify_connection()

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text from prompt"""
        return self._backend.generate(prompt, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Optional[str]]:
        """Generate text from multiple prompts (batched for vLLM)"""
        return self._backend.generate_batch(prompts, **kwargs)

    def generate_json(self, prompt: str, **kwargs) -> Optional[dict]:
        """Generate and parse JSON response"""
        response = self.generate(prompt, **kwargs)
        if not response:
            return None

        try:
            text = response
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
