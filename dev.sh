#!/usr/bin/env bash
set -e
source ~/.bashrc
source /data/ephemeral/home/py310/bin/activate
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Project root: $PROJECT_ROOT"

echo "Starting frontend..."
cd "$PROJECT_ROOT/frontend"
npm run dev &

FRONTEND_PID=$!

echo "Starting backend..."
cd "$PROJECT_ROOT/backend"
uvicorn app.main:app --reload &


echo "Frontend: http://localhost:5173"
echo "Backend : http://localhost:8000"



BACKEND_PID=$!

echo ""
echo "Frontend PID: $FRONTEND_PID"
echo "Backend  PID: $BACKEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "echo 'Stopping...'; kill $FRONTEND_PID $BACKEND_PID" INT

wait
