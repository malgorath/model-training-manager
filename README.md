# Model Training Manager

A professional-grade application for managing model training with worker orchestration, dataset management, and seamless Ollama integration.

## Features

- **Dataset Management**: Upload and manage CSV/JSON training datasets
- **Training Job Orchestration**: Create, monitor, and manage training jobs
- **Worker Pool Management**: Scale training with concurrent worker threads
- **Multiple Training Types**: Support for QLoRA, Unsloth, RAG, and standard fine-tuning
- **Real-time Monitoring**: Track training progress, epochs, and loss metrics
- **Modern UI**: Clean, responsive interface built with React and TypeScript

## Technology Stack

### Backend
- Python 3.11+
- FastAPI
- SQLAlchemy (SQLite)
- Pydantic

### Frontend
- React 18+
- TypeScript
- Vite
- TailwindCSS
- React Query

### Infrastructure
- Ollama (local installation)
- Python venv for backend
- npm for frontend

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ and npm
- **Ollama installed locally** with llama3.2:3b model
- At least 8GB RAM recommended

### Install Ollama (Required)

Install and run Ollama on your local machine:

```bash
# Install Ollama (see https://ollama.ai for your OS)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the model (in another terminal)
ollama pull llama3.2:3b
```

Ensure Ollama is running at `http://localhost:11434` before starting the application.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd trainers
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**:
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Ensure Ollama is running**:
   ```bash
   # Verify Ollama is accessible
   curl http://localhost:11434/api/tags
   ```

2. **Start the Backend**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start the Frontend** (in a new terminal):
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000/api/docs

## Running Tests

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest --cov=app --cov-report=term-missing
```

### Frontend Tests
```bash
cd frontend
npm test
```

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/datasets/` | Upload dataset |
| GET | `/api/v1/datasets/` | List datasets |
| POST | `/api/v1/jobs/` | Create training job |
| GET | `/api/v1/jobs/` | List training jobs |
| GET | `/api/v1/jobs/{id}/status` | Get job status |
| POST | `/api/v1/jobs/{id}/start` | Start a job manually |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel job |
| GET | `/api/v1/config/` | Get configuration |
| PATCH | `/api/v1/config/` | Update configuration |
| GET | `/api/v1/workers/` | Get worker status |
| POST | `/api/v1/workers/` | Control workers |

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

Create a `.env` file in the `backend` directory to override defaults:

```bash
DATABASE_URL=sqlite:///./trainers.db
UPLOAD_DIR=./uploads
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
MAX_WORKERS=8
DEBUG=false
```

### Training Parameters

Default training parameters can be configured via the Settings page:

- **Batch Size**: 1-64 (default: 4)
- **Learning Rate**: 1e-6 to 1.0 (default: 2e-4)
- **Epochs**: 1-100 (default: 3)
- **LoRA Rank**: 1-256 (default: 16)
- **LoRA Alpha**: 1-512 (default: 32)
- **LoRA Dropout**: 0.0-1.0 (default: 0.05)

## Project Structure

```
trainers/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Configuration, database
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── workers/       # Training workers
│   ├── tests/             # Backend tests
│   ├── venv/              # Python virtual environment
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API client
│   │   └── types/         # TypeScript types
│   ├── node_modules/      # npm dependencies
│   └── package.json
└── docs/                   # Documentation
```

## Training Types

### QLoRA (Quantized LoRA)
- 4-bit quantization for memory efficiency
- Low-rank adaptation for fast fine-tuning
- Best for limited GPU memory scenarios

### Unsloth (Optimized LoRA)
- Optimized LoRA training with Unsloth library
- Faster training with reduced memory usage
- Best for efficient fine-tuning

### RAG (Retrieval-Augmented Generation)
- Builds vector index from dataset
- Trains retrieval and generation components
- Best for knowledge-intensive tasks

### Standard Fine-tuning
- Traditional supervised fine-tuning
- Rule-based training configuration
- Best for straightforward adaptation tasks

## Troubleshooting

### Common Issues

1. **Ollama connection refused**:
   ```bash
   # Ensure Ollama is running locally
   ollama serve
   
   # Verify it's accessible
   curl http://localhost:11434/api/tags
   
   # Pull the model if needed
   ollama pull llama3.2:3b
   ```

2. **Database errors**:
   ```bash
   # Reset database
   rm backend/trainers.db
   # Restart backend - tables will be recreated
   ```

3. **Port conflicts**:
   ```bash
   # Check for processes using ports
   lsof -i :5173  # Frontend
   lsof -i :8000  # Backend
   lsof -i :11434 # Ollama
   ```

4. **Out of memory**:
   - Reduce batch size
   - Reduce max workers
   - Use QLoRA training type

5. **Python import errors**:
   ```bash
   # Ensure virtual environment is activated
   cd backend
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Frontend build errors**:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM serving
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://react.dev/) for the frontend framework
- [PEFT](https://huggingface.co/docs/peft) for QLoRA implementation
- [Unsloth](https://github.com/unslothai/unsloth) for optimized LoRA training