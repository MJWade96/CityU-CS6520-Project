#!/bin/bash
# Startup script for Medical RAG System

echo "=========================================="
echo "  Medical RAG System - Startup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project Root: ${PROJECT_ROOT}${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo -e "\n${YELLOW}Checking Python...${NC}"
if command_exists python3; then
    PYTHON_CMD=python3
elif command_exists python; then
    PYTHON_CMD=python
else
    echo -e "${RED}Error: Python not found. Please install Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}Found: $($PYTHON_CMD --version)${NC}"

# Check if virtual environment exists
VENV_PATH="$PROJECT_ROOT/python-rag/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}Virtual environment activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q fastapi uvicorn langchain langchain-community langchain-openai faiss-cpu sentence-transformers pydantic
echo -e "${GREEN}Dependencies installed${NC}"

# Start Python backend
echo -e "\n${YELLOW}Starting Python backend on port 8000...${NC}"
cd "$PROJECT_ROOT/python-rag"
$PYTHON_CMD -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to start
echo -e "\n${YELLOW}Waiting for backend to initialize...${NC}"
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}Backend is running at http://localhost:8000${NC}"
else
    echo -e "${RED}Warning: Backend may not be running properly${NC}"
fi

# Start Next.js frontend (optional)
echo -e "\n${YELLOW}To start the Next.js frontend, run:${NC}"
echo -e "  cd $PROJECT_ROOT && bun run dev"

echo -e "\n${GREEN}=========================================="
echo "  System is ready!"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "==========================================${NC}"

# Keep the script running
wait $BACKEND_PID
