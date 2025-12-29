# Model Training Manager - Makefile
# Provides common commands for development and deployment

.PHONY: help install dev test lint format build up down logs clean

# Default target
help:
	@echo "Model Training Manager - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install all dependencies"
	@echo "  make dev         - Start development environment"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View service logs"
	@echo "  make clean       - Remove containers and volumes"
	@echo ""
	@echo "Utilities:"
	@echo "  make pull-model  - Pull the Ollama model"
	@echo "  make shell-backend - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"

# Development setup
install:
	cd backend && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

# Run development servers
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo ""
	@echo "Services started:"
	@echo "  Backend: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/api/docs"
	@echo "  Ollama: http://localhost:11434"
	@echo ""
	@echo "To start frontend development server:"
	@echo "  cd frontend && npm run dev"

# Run tests
test: test-backend test-frontend

test-backend:
	cd backend && python -m pytest -v --cov=app --cov-report=term-missing

test-frontend:
	cd frontend && npm run test

# Linting
lint: lint-backend lint-frontend

lint-backend:
	cd backend && python -m flake8 app tests
	cd backend && python -m mypy app

lint-frontend:
	cd frontend && npm run lint

# Formatting
format: format-backend format-frontend

format-backend:
	cd backend && python -m black app tests
	cd backend && python -m isort app tests

format-frontend:
	cd frontend && npm run lint -- --fix

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/api/docs"
	@echo "  Ollama: http://localhost:11434"

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v --rmi local
	rm -rf backend/__pycache__ backend/.pytest_cache backend/.coverage backend/htmlcov
	rm -rf frontend/node_modules frontend/dist

# Utilities
pull-model:
	docker exec trainers-ollama ollama pull llama3.2:3b

shell-backend:
	docker exec -it trainers-backend bash

shell-frontend:
	docker exec -it trainers-frontend sh

# Database management
db-reset:
	docker-compose exec backend rm -f /app/data/trainers.db
	docker-compose restart backend

# Production deployment
prod-build:
	docker-compose -f docker-compose.yml build --no-cache

prod-up:
	docker-compose -f docker-compose.yml up -d

prod-down:
	docker-compose -f docker-compose.yml down

