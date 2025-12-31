# Model Type Implementation Documentation

**Date**: 2024-12-30  
**Author**: AI Assistant  
**Status**: Completed

## Overview

This document describes the implementation of the `model_type` field for projects, which allows users to specify the exact model architecture type (e.g., 'llama', 'bert') when creating training projects. This ensures proper model loading and prevents configuration errors during training.

## Problem Statement

Previously, when loading models for training, the system would sometimes encounter errors like:
```
'dict' object has no attribute 'model_type'
```

This occurred because the model configuration was being loaded as a dictionary instead of a proper `Config` object from the transformers library. The solution was to:

1. Add a `model_type` field to projects
2. Auto-detect model types from model config.json files
3. Filter available model types to only show compatible ones
4. Use the model_type during training to ensure proper config loading

## Implementation Details

### Backend Changes

#### 1. Database Schema (`backend/app/models/project.py`)

Added `model_type` column to the `Project` model:

```python
model_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
```

#### 2. Database Migration (`backend/app/core/database.py`)

Added migration to add `model_type` column to existing projects table:

```python
# Add model_type column if it doesn't exist
if "model_type" not in columns:
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE projects ADD COLUMN model_type VARCHAR(50)"))
    except Exception:
        pass  # Column might already exist
```

#### 3. API Endpoint (`backend/app/api/endpoints/projects.py`)

**New Endpoint**: `GET /api/v1/projects/models/{model_name:path}/types`

This endpoint:
- Reads the model's `config.json` to detect the `model_type`
- Loads the actual config using `AutoConfig.from_pretrained()` to get the Config class
- Filters `CONFIG_MAPPING` to find all model types that map to the same Config class
- Returns only compatible model types (e.g., for Llama models: `code_llama`, `llama`)

**Response Format**:
```json
{
  "model_type": "llama",
  "available_types": ["code_llama", "llama"],
  "recommended": "llama",
  "model_name": "meta-llama/Llama-3.2-3B"
}
```

**Key Implementation**:
```python
# Load the actual config to get the Config class
config = AutoConfig.from_pretrained(
    str(model_path),
    trust_remote_code=True,
)
config_class = type(config)

# Find all model types in CONFIG_MAPPING that map to the same Config class
compatible_types = [
    model_type 
    for model_type, mapped_class in CONFIG_MAPPING.items()
    if mapped_class == config_class
]
```

#### 4. Project Service (`backend/app/services/project_service.py`)

Updated `create_project()` to save `model_type`:

```python
project = Project(
    # ... other fields ...
    model_type=project_data.get("model_type"),
    # ...
)
```

#### 5. Training Worker (`backend/app/workers/training_worker.py`)

Enhanced model loading to use `model_type` and ensure proper Config object loading:

- Reads `model_type` from `config.json` first
- Uses `CONFIG_MAPPING` to load the specific Config class
- Ensures config is never a dict before passing to `AutoModelForCausalLM`
- Adds defensive checks and error handling

### Frontend Changes

#### 1. Project Form (`frontend/src/components/ProjectForm.tsx`)

**Added Model Type Field**:
- Appears after base model is selected and validated
- Auto-fetches available model types when base model changes
- Auto-selects recommended model type
- Shows loading state while fetching
- Displays warnings if selected type doesn't match detected type

**Implementation**:
```typescript
// Fetch model types when base_model changes
useEffect(() => {
  if (formData.base_model && availableModels && availableModels.includes(formData.base_model)) {
    setLoadingModelTypes(true);
    projectApi.getModelTypes(formData.base_model)
      .then((data) => {
        setModelTypes(data);
        // Auto-set recommended model_type if available
        if (data.recommended && !formData.model_type) {
          setFormData(prev => ({ ...prev, model_type: data.recommended }));
        }
      })
      // ...
  }
}, [formData.base_model, availableModels]);
```

**Form Field**:
```tsx
{formData.base_model && modelValid === true && (
  <div>
    <label htmlFor="model_type" className="label">
      <HelpTooltip content="The specific model architecture type (e.g., 'llama', 'bert'). Auto-detected from model config.json.">
        Model Type *
      </HelpTooltip>
    </label>
    {/* Dropdown with available types */}
  </div>
)}
```

#### 2. API Client (`frontend/src/services/api.ts`)

Added `getModelTypes` method:

```typescript
getModelTypes: async (modelName: string): Promise<ModelTypesResponse> => {
  const response = await api.get(`/projects/models/${encodeURIComponent(modelName)}/types`);
  return response.data;
},
```

#### 3. TypeScript Types (`frontend/src/types/index.ts`)

Added `model_type` to `Project` and `ProjectCreate` interfaces:

```typescript
export interface Project {
  // ... other fields ...
  model_type: string | null;
}

export interface ProjectCreate {
  // ... other fields ...
  model_type: string;
}

export interface ModelTypesResponse {
  model_type: string;
  available_types: string[];
  recommended: string | null;
  model_name: string;
}
```

## Testing

### Backend Tests

Integration test updated to use `model_type`:
- `backend/tests/integration_test_training_methods_ui.py` - Fetches and uses model_type when creating projects

### Frontend E2E Tests

Created Playwright E2E test:
- `frontend/e2e/training-methods.spec.ts` - Tests all 4 training methods via UI
- Interacts with the form to select model_type
- Verifies projects are created and visible in dashboard

## Migration Notes

- Existing projects without `model_type` will have `NULL` values
- New projects require `model_type` to be set
- The system auto-detects and recommends model types, but users can override

## Benefits

1. **Prevents Configuration Errors**: Ensures proper Config object loading during training
2. **Better User Experience**: Auto-detects and recommends correct model types
3. **Filtered Options**: Only shows compatible model types (e.g., 2 options instead of 50+)
4. **Type Safety**: Explicit model_type field prevents runtime errors

## Related Files

- `backend/app/models/project.py` - Project model with model_type field
- `backend/app/api/endpoints/projects.py` - Model types endpoint
- `backend/app/services/project_service.py` - Project creation with model_type
- `backend/app/workers/training_worker.py` - Model loading using model_type
- `frontend/src/components/ProjectForm.tsx` - Model type selection UI
- `frontend/src/services/api.ts` - API client for model types
- `frontend/src/types/index.ts` - TypeScript type definitions

## Future Improvements

- Cache model type detection results
- Support for custom model types
- Validation of model_type against actual model config at runtime
