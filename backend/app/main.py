from fastapi import FastAPI
from app.api.routes import recommend

app = FastAPI(debug=True)

app.include_router(recommend.router)

@app.get("/")
def root():
    return {"message" : "hi"}