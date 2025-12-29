# API Documentation

This document provides detailed documentation for the Model Training Manager API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement appropriate authentication mechanisms.

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production"
}
```

---

### Datasets

#### POST /datasets/

Upload a new dataset.

**Request**
- Content-Type: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | CSV or JSON file |
| name | string | Yes | Dataset name |
| description | string | No | Dataset description |

**Response** (201 Created)
```json
{
  "id": 1,
  "name": "Training Dataset",
  "description": "Sample training data",
  "filename": "data.csv",
  "file_path": "/uploads/abc123.csv",
  "file_type": "csv",
  "file_size": 1024,
  "row_count": 100,
  "column_count": 2,
  "columns": "[\"input\", \"output\"]",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

**Errors**
- `400 Bad Request`: Invalid file type or content

---

#### GET /datasets/

List all datasets with pagination.

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| page_size | integer | 10 | Items per page (max 100) |

**Response** (200 OK)
```json
{
  "items": [...],
  "total": 50,
  "page": 1,
  "page_size": 10,
  "pages": 5
}
```

---

#### GET /datasets/{id}

Get a specific dataset by ID.

**Response** (200 OK)
```json
{
  "id": 1,
  "name": "Training Dataset",
  ...
}
```

**Errors**
- `404 Not Found`: Dataset not found

---

#### PATCH /datasets/{id}

Update a dataset's metadata.

**Request Body**
```json
{
  "name": "Updated Name",
  "description": "Updated description"
}
```

**Response** (200 OK)
```json
{
  "id": 1,
  "name": "Updated Name",
  ...
}
```

---

#### DELETE /datasets/{id}

Delete a dataset.

**Response** (204 No Content)

**Errors**
- `404 Not Found`: Dataset not found

---

### Training Jobs

#### POST /jobs/

Create a new training job.

**Request Body**
```json
{
  "name": "My Training Job",
  "description": "Fine-tuning with custom data",
  "dataset_id": 1,
  "training_type": "qlora",
  "model_name": "llama3.2:3b",
  "batch_size": 4,
  "learning_rate": 0.0002,
  "epochs": 3,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

**Response** (201 Created)
```json
{
  "id": 1,
  "name": "My Training Job",
  "status": "pending",
  "training_type": "qlora",
  "progress": 0.0,
  "current_epoch": 0,
  ...
}
```

**Errors**
- `400 Bad Request`: Invalid dataset or parameters

---

#### GET /jobs/

List all training jobs with pagination.

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| page_size | integer | 10 | Items per page |
| status | string | null | Filter by status |

**Status Values**: `pending`, `queued`, `running`, `completed`, `failed`, `cancelled`

---

#### GET /jobs/{id}

Get a specific training job.

---

#### GET /jobs/{id}/status

Get the current status of a training job.

**Response** (200 OK)
```json
{
  "id": 1,
  "status": "running",
  "progress": 45.5,
  "current_epoch": 2,
  "current_loss": 0.234,
  "error_message": null
}
```

---

#### POST /jobs/{id}/cancel

Cancel a training job.

**Response** (200 OK)
```json
{
  "id": 1,
  "status": "cancelled",
  ...
}
```

**Errors**
- `400 Bad Request`: Job cannot be cancelled
- `404 Not Found`: Job not found

---

### Configuration

#### GET /config/

Get the current training configuration.

**Response** (200 OK)
```json
{
  "id": 1,
  "max_concurrent_workers": 4,
  "active_workers": 2,
  "default_model": "llama3.2:3b",
  "default_training_type": "qlora",
  "default_batch_size": 4,
  "default_learning_rate": 0.0002,
  "default_epochs": 3,
  "default_lora_r": 16,
  "default_lora_alpha": 32,
  "default_lora_dropout": 0.05,
  "auto_start_workers": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

---

#### PATCH /config/

Update the training configuration.

**Request Body**
```json
{
  "max_concurrent_workers": 8,
  "default_batch_size": 8,
  "auto_start_workers": true
}
```

---

### Workers

#### GET /workers/

Get the current worker pool status.

**Response** (200 OK)
```json
{
  "total_workers": 4,
  "active_workers": 4,
  "idle_workers": 2,
  "busy_workers": 2,
  "max_workers": 8,
  "workers": [
    {
      "id": "worker-abc123",
      "status": "busy",
      "current_job_id": 5,
      "jobs_completed": 10,
      "started_at": "2024-01-01T00:00:00Z",
      "last_activity": "2024-01-01T01:00:00Z"
    }
  ],
  "jobs_in_queue": 3
}
```

---

#### POST /workers/

Control the worker pool.

**Request Body**
```json
{
  "action": "start",
  "worker_count": 2
}
```

**Actions**:
- `start`: Start new workers (requires `worker_count`)
- `stop`: Stop all workers
- `restart`: Restart all workers

**Response** (200 OK)
Returns the updated worker pool status.

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

## Data Types

### TrainingStatus
- `pending`: Job created, waiting for worker
- `queued`: Job queued for processing
- `running`: Job currently being processed
- `completed`: Job finished successfully
- `failed`: Job failed with error
- `cancelled`: Job was cancelled

---

### Projects

#### POST /projects/

Create a new training project with traits and dataset allocations.

**Request Body**
```json
{
  "name": "My Reasoning Project",
  "description": "Fine-tuning for reasoning tasks",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "training_type": "qlora",
  "max_rows": 50000,
  "output_directory": "/output/my-project",
  "traits": [
    {
      "trait_type": "reasoning",
      "datasets": [
        {"dataset_id": 1, "percentage": 100.0}
      ]
    },
    {
      "trait_type": "coding",
      "datasets": [
        {"dataset_id": 2, "percentage": 100.0}
      ]
    },
    {
      "trait_type": "general_tools",
      "datasets": [
        {"dataset_id": 3, "percentage": 50.0},
        {"dataset_id": 4, "percentage": 30.0},
        {"dataset_id": 5, "percentage": 20.0}
      ]
    }
  ]
}
```

**Validation Rules**:
- Reasoning trait: exactly 1 dataset, 100% allocation
- Coding trait: exactly 1 dataset, 100% allocation
- General/Tools trait: 1+ datasets, percentages must sum to 100%
- No dataset can be used twice in the same project
- Output directory must be writable

**Response** (201 Created)
```json
{
  "id": 1,
  "name": "My Reasoning Project",
  "status": "pending",
  "progress": 0.0,
  "traits": [...],
  ...
}
```

**Errors**
- `400 Bad Request`: Validation failed (invalid trait configuration, percentages don't sum to 100%, etc.)

---

#### GET /projects/

List all projects with pagination.

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| page_size | integer | 50 | Items per page (max 100) |

**Response** (200 OK)
```json
{
  "items": [...],
  "total": 10,
  "page": 1,
  "page_size": 50
}
```

---

#### GET /projects/{id}

Get a specific project by ID with full trait and dataset allocation details.

**Response** (200 OK)
```json
{
  "id": 1,
  "name": "My Reasoning Project",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "status": "running",
  "progress": 45.5,
  "traits": [
    {
      "id": 1,
      "trait_type": "reasoning",
      "datasets": [
        {"dataset_id": 1, "percentage": 100.0}
      ]
    }
  ],
  ...
}
```

---

#### PATCH /projects/{id}

Update project metadata (name, description). Cannot update traits or datasets after creation.

**Request Body**
```json
{
  "name": "Updated Project Name",
  "description": "Updated description"
}
```

---

#### POST /projects/{id}/start

Start training for a project. Validates model availability before queuing.

**Response** (200 OK)
Returns updated project with status set to "pending".

**Errors**
- `400 Bad Request`: Model not available
- `404 Not Found`: Project not found

---

#### POST /projects/{id}/validate

Validate the trained model after training completes. Performs file checks and loading tests.

**Response** (200 OK)
```json
{
  "valid": true,
  "files": {
    "valid": true,
    "files_checked": ["config.json", "tokenizer.json"],
    "errors": []
  },
  "loading": {
    "valid": true,
    "errors": []
  },
  "errors": []
}
```

---

#### POST /projects/validate-output-dir

Validate that an output directory is writable.

**Request Body**
```json
{
  "output_directory": "/output/my-project"
}
```

**Response** (200 OK)
```json
{
  "valid": true,
  "writable": true,
  "path": "/output/my-project",
  "errors": []
}
```

**Errors**
- `400 Bad Request`: Directory doesn't exist or isn't writable

---

#### POST /projects/validate-model

Check if a model is available locally.

**Request Body**
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct"
}
```

**Response** (200 OK)
```json
{
  "available": true,
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "path": "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/abc123",
  "errors": []
}
```

---

#### GET /projects/models/available

List all models available in local HuggingFace cache.

**Response** (200 OK)
```json
[
  "meta-llama/Llama-3.2-3B-Instruct",
  "microsoft/Phi-3-mini-4k-instruct",
  "mistralai/Mistral-7B-Instruct-v0.2"
]
```

---

### TrainingType
- `qlora`: Quantized LoRA training
- `rag`: Retrieval-Augmented Generation
- `standard`: Standard fine-tuning

### WorkerStatus
- `idle`: Worker waiting for jobs
- `busy`: Worker processing a job
- `stopped`: Worker is stopped
- `error`: Worker encountered an error

