# Log Monitoring and Error Tracking Documentation

**Date**: 2024-12-30  
**Author**: AI Assistant  
**Status**: Completed

## Overview

This document describes the log monitoring setup and error tracking procedures for the Model Training Manager application.

## Log Files

### Backend Logs

**Location**: `.backend.log` (in project root)

Contains:
- Application startup logs
- API request/response logs
- Database migration logs
- Worker pool logs
- Training job logs
- Error traces and stack traces

### Frontend Logs

**Location**: `.frontend.log` (in project root)

Contains:
- Vite dev server logs
- Build errors
- Runtime errors

## Monitoring Procedures

### Daily Log Checks

According to `.cursorrules`:
> **Mandatory Log Checks**: If an error occurs, you MUST read the logs (`storage/logs/`, `app.log`, etc.) via BASH to find the root cause before fixing.

### Common Commands

```bash
# Check backend logs for errors
tail -200 .backend.log | grep -E 'ERROR|Error|Traceback|Exception|Failed|failed' | head -50

# Check frontend logs for errors
tail -200 .frontend.log | grep -E 'ERROR|Error|Traceback|Exception|Failed|failed' | head -50

# View recent backend activity
tail -100 .backend.log

# View recent frontend activity
tail -100 .frontend.log

# Search for specific errors
grep -i "model_type" .backend.log | tail -20
```

## Recent Fixes

### 1. Migration Errors (2024-12-30)

**Issue**: Duplicate column errors during database migrations
```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) duplicate column name: model_api_url
```

**Fix**: Wrapped all `ALTER TABLE` statements in try-except blocks to handle cases where columns already exist:

```python
# Add model_type column if it doesn't exist
if "model_type" not in columns:
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE projects ADD COLUMN model_type VARCHAR(50)"))
    except Exception:
        pass  # Column might already exist
```

**Files Changed**:
- `backend/app/core/database.py`

### 2. Model Types Endpoint 404 (2024-12-30)

**Issue**: Route parameter not handling slashes in model names
```
GET /api/v1/projects/models/meta-llama/Llama-3.2-3B/types HTTP/1.1" 404 Not Found
```

**Fix**: Changed route parameter from `{model_name}` to `{model_name:path}` to handle slashes:

```python
@router.get("/models/{model_name:path}/types", tags=["Models"])
```

**Files Changed**:
- `backend/app/api/endpoints/projects.py`

### 3. Missing Type Import (2024-12-30)

**Issue**: `NameError: name 'Any' is not defined`

**Fix**: Added missing import:
```python
from typing import Annotated, Optional, Any
```

**Files Changed**:
- `backend/app/api/endpoints/projects.py`

### 4. Model Type Filtering (2024-12-30)

**Issue**: Endpoint returning all 50+ model types instead of only compatible ones

**Fix**: Filtered `CONFIG_MAPPING` to only return model types that map to the same Config class:

```python
# Load the actual config to get the Config class
config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
config_class = type(config)

# Find all model types in CONFIG_MAPPING that map to the same Config class
compatible_types = [
    model_type 
    for model_type, mapped_class in CONFIG_MAPPING.items()
    if mapped_class == config_class
]
```

**Files Changed**:
- `backend/app/api/endpoints/projects.py`

## Error Patterns to Watch For

### Database Errors
- `duplicate column name` - Migration issue
- `NOT NULL constraint failed` - Missing required fields
- `no such table` - Migration not run

### API Errors
- `404 Not Found` - Route not found or parameter issue
- `500 Internal Server Error` - Backend exception
- `400 Bad Request` - Validation error

### Model Loading Errors
- `'dict' object has no attribute 'model_type'` - Config loading issue
- `Model not found` - Model path resolution issue
- `Failed to load model` - Model format or path issue

### Frontend Errors
- `Cannot read properties of null` - Null safety issue
- `TypeError` - Type mismatch
- `404` - API endpoint not found

## Log Analysis Workflow

1. **Identify Error**: Check logs for ERROR, Exception, or Failed messages
2. **Extract Context**: Get full stack trace and surrounding log lines
3. **Root Cause**: Analyze error message and stack trace
4. **Fix Implementation**: Implement fix following TDD
5. **Verify**: Check logs again to confirm fix
6. **Document**: Update this document with the fix

## Automation

### Future Improvements

- Set up log aggregation (e.g., ELK stack)
- Implement log rotation
- Add structured logging (JSON format)
- Set up alerts for critical errors
- Create log analysis dashboard

## Related Files

- `.backend.log` - Backend application logs
- `.frontend.log` - Frontend dev server logs
- `backend/app/core/database.py` - Database migration logs
- `backend/app/workers/training_worker.py` - Training job logs

## References

- `.cursorrules` - Section 4: LOG DIAGNOSTICS & ERROR TRACKING
- Python logging: https://docs.python.org/3/library/logging.html
- FastAPI logging: https://fastapi.tiangolo.com/advanced/logging/
