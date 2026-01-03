# Changelog

All notable changes to the Model Training Manager project.

## [Unreleased]

### Fixed
- **QLoRA Dict Config Error**: Fixed critical issue where QLoRA training failed with `'dict' object has no attribute 'model_type'` error
  - Implemented config protection mechanism using property descriptor to intercept and prevent dict conversion during quantization
  - Added 8-bit quantization fallback before falling back to no quantization
  - Added library version logging for debugging (transformers, bitsandbytes, peft, torch)
  - Fixed syntax error in `_train_qlora_real()` method (line 726)
  - Consolidated error handling and removed redundant config fix attempts
  - Config is now protected immediately after model loading, preventing quantization from converting it to dict
  - File: `backend/app/workers/training_worker.py`

- **RAG Training Fixes** (2026-01-03):
  - Fixed missing RAG dependencies: Added `sentence-transformers>=2.7.0` and `faiss-cpu>=1.7.4` to `requirements.txt`
  - Fixed RAG `model_path` None error: Set `project.model_path` before creating `ProjectWrapper` and before calling `_train_rag_real`
  - Fixed RAG validation wrong directory: Validation now checks `output_dir/rag_model` instead of `output_dir` for RAG models
  - Added RAG-specific model validation: `ModelValidationService` now detects RAG models by `rag_config.json` and validates RAG-specific files (FAISS index, documents.json)
  - Added path expansion: All paths now use `.expanduser()` to handle `~` in paths
  - Files: `backend/app/workers/training_worker.py`, `backend/app/services/model_validation_service.py`, `backend/requirements.txt`

- **Form Validation Fixes** (2026-01-03):
  - Fixed stale closure bug in ProjectForm: Changed `setFormData({ ...formData, ... })` to functional update pattern `setFormData(prev => ({ ...prev, ... }))`
  - Fixed model type auto-selection not persisting when form data was updated
  - Files: `frontend/src/components/ProjectForm.tsx`

### Added
- **Comprehensive Test Suite**: Created complete test coverage for all API endpoints, button clicks, and autofill fields
  - `test_all_api_endpoints.py`: Tests for all 41+ API endpoints
  - `ButtonClicks.test.tsx`: Tests for all button interactions in frontend components
  - `AutofillFields.test.tsx`: Tests for all autofill field behaviors
- **TEST_COVERAGE.md**: Documentation of all test coverage
- **Dataset Storage Reorganization**: Datasets are now stored in `./data/{author}/{datasetname}/` directory structure
- **Model Storage Reorganization**: Models are now stored in `./data/models/{org}/{model_name}/` directory structure
- **Filesystem Scanning**: Added `scan_datasets()` and `scan_models()` methods to automatically discover and register existing files
- **Refresh Buttons**: Added manual refresh buttons to Datasets and Models tables to trigger filesystem scans
- **Output Directory Auto-fill**: Project form now auto-generates output directory path from settings base + sanitized project name
- **Path Standardization**: All file paths are now centralized in `config.py` with helper methods

### Changed
- **Removed `max_rows` Requirement**: Projects no longer require `max_rows` field - it's now optional/nullable
- **Removed Model API Settings**: Removed "Model API Settings" section from TrainingConfig component
- **Updated Attribution**: Changed footer from "Powered by Ollama" to "Powered by Scott Sanders" with links
- **License Update**: Updated LICENSE file copyright to "Scott Sanders"

### Fixed
- **Output Directory Auto-fill**: Fixed output directory field to properly auto-fill and update when project name changes
- **Output Directory Validation**: Removed blocking validation requirement - directory is created automatically by backend on project creation
- **Project Creation Flow**: Users no longer need to click in output directory field before creating project - button is enabled once all required fields are filled
- **Dataset Percentage Validation**: **REMOVED** - Percentages are per-file usage amounts and no longer required to sum to 100%. Each percentage indicates how much of that specific dataset file to use.
- **Database Migration**: Added migration to make `max_rows` column nullable in projects table
- **TypeScript Types**: Updated ProjectCreate and Project interfaces to reflect optional `max_rows`

### Technical Details

#### Dataset Storage
- Datasets are stored in: `./data/{author}/{datasetname}/`
- User uploads default to `./data/user/{datasetname}/`
- HuggingFace downloads use: `./data/{hf_author}/{datasetname}/`

#### Model Storage
- Models are stored in: `./data/models/{org}/{model_name}/`
- Training job outputs: `./data/models/job_{job_id}/`

#### Output Directory
- Default base: `./output` (configurable in settings)
- Auto-generated format: `{base_directory}/{sanitized-project-name}`
- Project name sanitization: spaces → hyphens, special chars → underscores, lowercase

## Testing

All changes follow Test-Driven Development (TDD) methodology:
- Tests written before implementation
- Comprehensive test coverage maintained
- All tests passing before code completion
