# Setup Guide

This guide provides detailed instructions for setting up the Model Training Manager.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk Space**: 20GB+ for models and data
- **CPU**: Modern multi-core processor
- **GPU**: Optional, but recommended for faster training (NVIDIA CUDA compatible)

### Software Requirements

- Python 3.11+
- Node.js 20+
- npm 9+
- Git
- **Ollama installed locally**

## Installation

### Step 1: Install Ollama (Required)

Ollama must be installed and running locally on your machine.

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

Start Ollama and pull the model:

```bash
# Start Ollama server
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:3b

# Verify it's running
curl http://localhost:11434/api/tags
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd trainers
```

### Step 3: Backend Setup

1. **Create Virtual Environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Environment** (Optional)
   
   Create a `.env` file in the `backend` directory:
   ```bash
   DATABASE_URL=sqlite:///./trainers.db
   UPLOAD_DIR=./uploads
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   MAX_WORKERS=8
   DEBUG=false
   ```

   Or export environment variables:
   ```bash
   export DATABASE_URL="sqlite:///./trainers.db"
   export UPLOAD_DIR="./uploads"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export DEBUG=true
   ```

4. **Run the Backend**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The backend will be available at http://localhost:8000

### Step 4: Frontend Setup

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Run the Frontend**
   ```bash
   npm run dev
   ```

   The frontend will be available at http://localhost:5173

3. **Build for Production** (Optional)
   ```bash
   npm run build
   npm run preview
   ```

### Step 5: Verify Installation

1. **Check Backend Health**:
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status":"healthy","version":"1.0.0","environment":"development"}
   ```

2. **Check API Docs**:
   Open http://localhost:8000/api/docs in your browser

3. **Check Frontend**:
   Open http://localhost:5173 in your browser

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./trainers.db` | Database connection URL |
| `UPLOAD_DIR` | `./uploads` | Dataset upload directory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model for training |
| `MAX_WORKERS` | `8` | Maximum concurrent workers |
| `DEBUG` | `false` | Enable debug mode |

### Frontend Configuration

The frontend API base URL can be configured in `frontend/src/services/api.ts`:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
```

Set `VITE_API_BASE_URL` in a `.env` file in the `frontend` directory if needed.

## GPU Support (Ollama)

If you have an NVIDIA GPU, Ollama will automatically use it when installed locally. Verify with:

```bash
# Check GPU is being used
ollama run llama3.2:3b "Hello, how are you?"
# Watch GPU usage with nvidia-smi in another terminal
```

## Running Tests

### Backend Tests

```bash
cd backend
source venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run in watch mode
npm test -- --watch

# Run with coverage
npm test -- --coverage
```

## Troubleshooting

### Ollama Connection Issues

**Model not found**
```bash
# List available models
ollama list

# Pull model if missing
ollama pull llama3.2:3b
```

**Connection refused**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

**Ollama not using GPU**
```bash
# Verify CUDA is available
nvidia-smi

# Check Ollama is using GPU
ollama ps

# Reinstall Ollama with GPU support if needed
```

### Backend Issues

**Database errors**
```bash
# Reset database
cd backend
rm trainers.db
# Restart backend - tables will be recreated automatically
```

**Import errors**
```bash
# Ensure virtual environment is activated
cd backend
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Port already in use**
```bash
# Find process using port
lsof -i :8000

# Kill the process if needed
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --port 8001
```

### Frontend Issues

**Build errors**
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules
rm package-lock.json
npm install
```

**Port already in use**
```bash
# Use a different port
npm run dev -- --port 5174
```

**API connection errors**
- Ensure backend is running on http://localhost:8000
- Check `VITE_API_BASE_URL` in `.env` if using custom URL
- Verify CORS is enabled in backend (default: allows all origins)

## Updating

### Update Application

```bash
# Pull latest changes
git pull origin main

# Update backend dependencies
cd backend
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Update frontend dependencies
cd ../frontend
npm install
```

### Update Ollama Model

```bash
# Pull latest model version
ollama pull llama3.2:3b
```

## Backup and Restore

### Backup Data

```bash
# Create backup directory
mkdir -p backups

# Backup database
cp backend/trainers.db backups/

# Backup uploads
cp -r backend/uploads backups/
```

### Restore Data

```bash
# Restore database
cp backups/trainers.db backend/

# Restore uploads
cp -r backups/uploads backend/

# Restart backend
```

## Development Workflow

### Running in Development Mode

**Backend** (with hot-reload):
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

**Frontend** (with hot-reload):
```bash
cd frontend
npm run dev
```

### Code Formatting

**Backend**:
```bash
cd backend
source venv/bin/activate
black app tests
isort app tests
```

**Frontend**:
```bash
cd frontend
npm run format  # if configured
```

### Linting

**Backend**:
```bash
cd backend
source venv/bin/activate
flake8 app tests
mypy app
```

**Frontend**:
```bash
cd frontend
npm run lint
```
