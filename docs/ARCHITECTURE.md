# Architecture Documentation

This document describes the architecture of the Model Training Manager application.

## System Overview

The Model Training Manager is a full-stack application designed to manage LLM fine-tuning workflows. It provides a clean interface for uploading datasets, configuring training parameters, and orchestrating concurrent training workers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  React Frontend                          │    │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │    │
│  │  │Dashboard│  │ Datasets │  │ Jobs    │  │ Settings │  │    │
│  │  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  │    │
│  │       │            │             │            │         │    │
│  │       └────────────┴─────────────┴────────────┘         │    │
│  │                        │                                 │    │
│  │              ┌─────────▼─────────┐                      │    │
│  │              │    API Client     │                      │    │
│  │              │  (React Query)    │                      │    │
│  │              └─────────┬─────────┘                      │    │
│  └────────────────────────┼────────────────────────────────┘    │
│                           │ HTTP/REST                            │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  FastAPI Backend                         │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │                  API Layer                        │   │    │
│  │  │  /datasets  /jobs  /config  /workers  /health    │   │    │
│  │  └─────────────────────┬────────────────────────────┘   │    │
│  │                        │                                 │    │
│  │  ┌─────────────────────▼────────────────────────────┐   │    │
│  │  │               Service Layer                       │   │    │
│  │  │  ┌──────────┐ ┌─────────────┐ ┌──────────────┐  │   │    │
│  │  │  │ Dataset  │ │  Training   │ │   Ollama     │  │   │    │
│  │  │  │ Service  │ │  Service    │ │   Service    │  │   │    │
│  │  │  └────┬─────┘ └──────┬──────┘ └──────┬───────┘  │   │    │
│  │  └───────┼──────────────┼───────────────┼──────────┘   │    │
│  │          │              │               │               │    │
│  │  ┌───────▼──────────────▼───────────────┼──────────┐   │    │
│  │  │            Worker Pool               │          │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐   │          │   │    │
│  │  │  │Worker 1│ │Worker 2│ │Worker N│   │          │   │    │
│  │  │  └───┬────┘ └───┬────┘ └───┬────┘   │          │   │    │
│  │  └──────┼──────────┼──────────┼────────┼──────────┘   │    │
│  │         │          │          │        │               │    │
│  │         └──────────┼──────────┘        │               │    │
│  │                    │                   │               │    │
│  └────────────────────┼───────────────────┼───────────────┘    │
│                       │                   │                     │
│  ┌────────────────────▼───────┐  ┌───────▼──────────────────┐  │
│  │         SQLite             │  │        Ollama            │  │
│  │  ┌─────────┐ ┌──────────┐  │  │    localhost:11434       │  │
│  │  │Datasets │ │   Jobs   │  │  │  ┌─────────────────┐     │  │
│  │  └─────────┘ └──────────┘  │  │  │   llama3.2:3b   │     │  │
│  │  ┌─────────┐               │  │  └─────────────────┘     │  │
│  │  │ Config  │               │  │                          │  │
│  │  └─────────┘               │  │                          │  │
│  └────────────────────────────┘  └──────────────────────────┘  │
│                        Infrastructure Layer                      │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Frontend (React + TypeScript)

The frontend is a single-page application built with React and TypeScript.

#### Key Components
- **Layout**: Main application shell with navigation
- **DashboardPage**: Overview with stats and recent activity
- **DatasetsPage**: Dataset management with upload
- **TrainingJobsPage**: Job creation and monitoring
- **SettingsPage**: Configuration management

#### State Management
- **React Query**: Server state management with caching
- **Local State**: Component-level state with React hooks

#### Styling
- **TailwindCSS**: Utility-first CSS framework
- **Custom Design System**: Consistent components and theming

### Backend (FastAPI + Python)

The backend is a RESTful API built with FastAPI.

#### Layer Structure

1. **API Layer** (`app/api/`)
   - Route handlers and request/response handling
   - Input validation with Pydantic
   - Error handling and HTTP responses

2. **Service Layer** (`app/services/`)
   - Business logic implementation
   - Orchestration between components
   - Transaction management

3. **Data Layer** (`app/models/`)
   - SQLAlchemy ORM models
   - Database relationships
   - Data persistence

4. **Worker Layer** (`app/workers/`)
   - Training worker implementation
   - Worker pool management
   - Job processing logic

### Database (SQLite)

SQLite is used for data persistence with three main tables:

#### Tables

1. **datasets**
   - Stores dataset metadata
   - Links to file system for actual data
   - Tracks row/column counts

2. **training_jobs**
   - Training job configurations
   - Progress tracking
   - Status management

3. **training_config**
   - Global configuration settings
   - Default parameters
   - Worker settings

### Ollama Integration

Ollama provides local LLM serving capabilities.

#### Integration Points
- Health checks
- Model availability
- Training inference
- Model information retrieval

## Data Flow

### Dataset Upload Flow

```
1. User selects file → Frontend validates type
2. Frontend sends multipart request → Backend receives
3. Dataset Service parses file → Extracts metadata
4. File saved to disk → Metadata saved to DB
5. Response returned → Frontend updates UI
```

### Training Job Flow

```
1. User creates job → Frontend sends request
2. Training Service validates → Creates job record
3. Job queued → Worker pool notified
4. Idle worker picks up → Status updated to running
5. Training executes → Progress updates sent
6. Training completes → Status updated to completed
7. Frontend polls → UI shows completion
```

### Worker Lifecycle

```
1. Start command received → Workers spawned
2. Worker registers → Added to pool
3. Worker checks queue → Picks up job if available
4. Job executed → Updates progress in DB
5. Job completes → Worker returns to idle
6. Stop command → Workers gracefully shutdown
```

## Security Considerations

### Input Validation
- Pydantic schemas validate all inputs
- File type restrictions (CSV, JSON only)
- File size limits enforced
- SQL injection prevention via ORM

### Runtime Security
- Proper file permissions
- Input sanitization
- Secure file handling

### API Security
- CORS configuration
- Request rate limiting (recommended for production)
- Authentication (implement for production)

## Scalability

### Current Design
- Single-instance deployment
- SQLite database (single-writer)
- Thread-based worker pool

### Scaling Recommendations

1. **Horizontal Scaling**
   - Replace SQLite with PostgreSQL
   - Use Redis for job queue
   - Kubernetes deployment

2. **Worker Scaling**
   - Separate worker service
   - GPU scheduling
   - Auto-scaling based on queue depth

3. **Frontend Scaling**
   - CDN for static assets
   - Multiple Nginx instances
   - Load balancing

## Deployment Options

### Local Development
- Run backend and frontend directly
- Easy setup with Python venv and npm
- Fast iteration and debugging

### Production Deployment
- Use process manager (systemd, supervisor)
- Reverse proxy (nginx, Apache) for frontend
- WSGI/ASGI server (gunicorn, uvicorn) for backend
- Multiple instances for load balancing

## Performance Considerations

### Database
- Indexes on frequently queried fields
- Connection pooling
- Query optimization

### API
- Async handlers for I/O operations
- Response caching where appropriate
- Pagination for list endpoints

### Training
- Batch processing
- Memory-efficient data loading
- GPU utilization optimization

## Monitoring and Observability

### Recommended Tools
- Prometheus for metrics
- Grafana for dashboards
- Structured logging (JSON)
- Distributed tracing (optional)

### Key Metrics
- Request latency
- Error rates
- Worker utilization
- Training throughput
- Queue depth

