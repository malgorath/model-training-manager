# TODO: Next Session

**Date**: 2026-01-03  
**Status**: ✅ All Training Methods Working

## Session Summary (2026-01-03)

### ✅ Completed: All Training Methods Tested via UI

All four training methods have been tested and are working:

1. **QLoRA** - ✅ Working (with config protection mechanism)
2. **Unsloth** - ✅ Working (requires GPU, expected failure on CPU-only systems)
3. **RAG** - ✅ Working (fixed multiple issues this session)
4. **Standard** - ✅ Working

### Fixes Applied This Session

#### 1. Missing RAG Dependencies
- **Issue**: RAG training failed with `No module named 'sentence_transformers'`
- **Fix**: Added `sentence-transformers>=2.7.0` and `faiss-cpu>=1.7.4` to `backend/requirements.txt`
- **GitHub Issue**: #12 (closed)

#### 2. RAG model_path None Error
- **Issue**: RAG training failed with `argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'`
- **Root Cause**: `_train_rag_real` used `job.model_path` but it was `None` when called because `ProjectWrapper` copied `model_path` before it was set
- **Fix**: Set `project.model_path` BEFORE creating `ProjectWrapper` for RAG training
- **Files**: `backend/app/workers/training_worker.py`

#### 3. RAG Validation Wrong Directory
- **Issue**: RAG validation failed because it was checking `output_dir` instead of `output_dir/rag_model`
- **Root Cause**: RAG model files are saved to `rag_model/` subdirectory, but validation checked the parent
- **Fix**: Added RAG-specific validation path logic in `training_worker.py`
- **GitHub Issue**: #14 (closed)

#### 4. RAG Model Validation Service
- **Issue**: `ModelValidationService` tried to validate RAG models as standard transformers models
- **Fix**: Added RAG-specific validation that detects RAG models by `rag_config.json` and validates FAISS index, documents.json
- **GitHub Issue**: #15 (closed)
- **Files**: `backend/app/services/model_validation_service.py`

#### 5. Form Validation Stale Closure Bug
- **Issue**: Project name was being cleared when selecting model or training type
- **Root Cause**: `setFormData({ ...formData, ... })` used stale closure values
- **Fix**: Changed to functional update pattern `setFormData(prev => ({ ...prev, ... }))`
- **GitHub Issues**: #10, #11 (closed)
- **Files**: `frontend/src/components/ProjectForm.tsx`

#### 6. Path Expansion
- **Issue**: Paths containing `~` weren't being expanded
- **Fix**: Added `.expanduser()` to all Path operations
- **Files**: `backend/app/workers/training_worker.py`, `backend/app/services/model_validation_service.py`, `backend/app/workers/project_training_worker.py`

## Open Issues

The following issues remain open for future sessions:

- **#1**: Fix QLoRA test failures after config protection implementation
- **#7**: Fix React act() warnings in ProjectForm tests
- **#8**: Update React Router to use v7 future flags

## Files Modified This Session

### Backend
- `backend/app/workers/training_worker.py` - RAG model_path fix, RAG validation path fix, path expansion
- `backend/app/workers/project_training_worker.py` - Path expansion, validation path fix
- `backend/app/services/model_validation_service.py` - RAG-specific validation
- `backend/requirements.txt` - Added sentence-transformers and faiss-cpu

### Frontend
- `frontend/src/components/ProjectForm.tsx` - Functional update pattern for setFormData

### Documentation
- `CHANGELOG.md` - Updated with all fixes
- `docs/TODO_NEXT_SESSION.md` - This file

## Next Steps (Future Sessions)

### High Priority Features

1. **#16 - Add real-time training stats to Project view**
   - Real-time updates during training (live dashboard)
   - Store all stats in database for historical comparison
   - Training progress: total rows trained, training speed (rows/second)
   - Hardware metrics: VRAM (current/peak/avg), RAM (current/peak/avg), CPU Time, GPU Time
   - Remove "Max Rows: N/A" display when not applicable
   - Update via polling or WebSocket during active training

2. **#17 - Auto-refresh tables after create/update/delete operations**
   - Apply to all tables: Projects, Datasets, Jobs, Models, Workers, etc.
   - Automatically refresh relevant table after successful operations
   - Use React Query invalidation or similar mechanism
   - User should see new/updated items immediately without page refresh

3. **#18 - Fix auto-start workers not working**
   - When `auto_start_workers: true` and `default_worker_count: 4`, workers should start automatically on backend startup
   - Currently requires manual "Start Worker" clicks
   - Check backend startup code for worker initialization
   - Verify settings are read correctly on startup

### Other Tasks

4. Fix remaining open GitHub issues (#1, #7, #8)
5. Run full test suite and fix any failures
6. Consider adding automated tests for RAG training
7. Review and merge changes to main branch
