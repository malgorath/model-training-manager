#!/bin/bash

# Model Training Manager - Development Startup Script
# Starts both backend and frontend development servers

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"

# PID files for cleanup
BACKEND_PID_FILE="${SCRIPT_DIR}/.backend.pid"
FRONTEND_PID_FILE="${SCRIPT_DIR}/.frontend.pid"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Kill backend if running
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p "$BACKEND_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping backend (PID: $BACKEND_PID)...${NC}"
            kill "$BACKEND_PID" 2>/dev/null || true
            wait "$BACKEND_PID" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Kill frontend if running
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping frontend (PID: $FRONTEND_PID)...${NC}"
            kill "$FRONTEND_PID" 2>/dev/null || true
            wait "$FRONTEND_PID" 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # Kill any remaining uvicorn or vite processes
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    
    echo -e "${GREEN}Cleanup complete.${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Check if backend venv exists
if [ ! -d "${BACKEND_DIR}/venv" ]; then
    echo -e "${YELLOW}Backend virtual environment not found. Creating...${NC}"
    cd "$BACKEND_DIR"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${BLUE}Installing backend dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Backend virtual environment created.${NC}"
else
    echo -e "${GREEN}Backend virtual environment found.${NC}"
fi

# Check if frontend node_modules exists
if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
    echo -e "${YELLOW}Frontend dependencies not found. Installing...${NC}"
    cd "$FRONTEND_DIR"
    npm install
    echo -e "${GREEN}Frontend dependencies installed.${NC}"
else
    echo -e "${GREEN}Frontend dependencies found.${NC}"
fi

# Start backend
echo -e "\n${BLUE}Starting backend server...${NC}"
cd "$BACKEND_DIR"
source venv/bin/activate

# Start uvicorn in background
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > "${SCRIPT_DIR}/.backend.log" 2>&1 &

BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
echo -e "${BLUE}Backend logs: ${SCRIPT_DIR}/.backend.log${NC}"

# Wait a moment for backend to start
sleep 2

# Check if backend is actually running
if ! ps -p "$BACKEND_PID" > /dev/null 2>&1; then
    echo -e "${RED}Backend failed to start. Check ${SCRIPT_DIR}/.backend.log for errors.${NC}"
    cat "${SCRIPT_DIR}/.backend.log"
    exit 1
fi

# Start frontend
echo -e "\n${BLUE}Starting frontend server...${NC}"
cd "$FRONTEND_DIR"

# Start npm dev server in background
npm run dev \
    > "${SCRIPT_DIR}/.frontend.log" 2>&1 &

FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
echo -e "${BLUE}Frontend logs: ${SCRIPT_DIR}/.frontend.log${NC}"

# Wait a moment for frontend to start
sleep 2

# Check if frontend is actually running
if ! ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
    echo -e "${RED}Frontend failed to start. Check ${SCRIPT_DIR}/.frontend.log for errors.${NC}"
    cat "${SCRIPT_DIR}/.frontend.log"
    exit 1
fi

# Display status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Development servers started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "${BLUE}API Docs:${NC} http://localhost:8000/api/docs"
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Tail logs in real-time
tail -f "${SCRIPT_DIR}/.backend.log" "${SCRIPT_DIR}/.frontend.log" 2>/dev/null &
TAIL_PID=$!

# Wait for processes
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
