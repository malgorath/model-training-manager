# Model Loading Dict Config Issue - In Progress

**Date**: 2024-12-30  
**Status**: Blocked - Needs Investigation  
**Priority**: Critical

## Problem Summary

Training projects are failing with the error:
```
Failed to load model from data/models/meta-llama/Llama-3.2-3B: 'dict' object has no attribute 'model_type' (fix failed: 'dict' object has no attribute 'model_type')
```

This error occurs during QLoRA training when loading models with 4-bit quantization.

## Root Cause Analysis

The issue appears to be a compatibility problem between:
- `transformers` library's `AutoModelForCausalLM.from_pretrained()`
- `bitsandbytes` quantization (`BitsAndBytesConfig`)
- Model config loading

**What's happening:**
1. We load a proper `Config` object (e.g., `LlamaConfig`) using `CONFIG_MAPPING[model_type]`
2. We pass this config to `AutoModelForCausalLM.from_pretrained()` with `quantization_config`
3. During quantization loading, the BitsAndBytes library internally accesses `model.config.model_type`
4. At that point, `model.config` has been converted to a `dict` instead of a `Config` object
5. The access fails with `'dict' object has no attribute 'model_type'`

**Why fixes fail:**
- Even when we load the model without the `config` parameter and immediately fix `model.config`, the quantization process still converts it to a dict
- The error occurs DURING model loading, before we can fix it
- Exception handlers that try to reload and fix also encounter the same issue

## What Was Attempted (3 Attempts)

### Attempt 1: Use Project model_type + Fix CONFIG_MAPPING.get()
- **Changes:**
  - Updated `_train_qlora_real()` to use `job._project.model_type` if available
  - Changed `CONFIG_MAPPING.get(model_type)` to `CONFIG_MAPPING[model_type]` (because `CONFIG_MAPPING` is a `_LazyConfigMapping` that doesn't support `.get()`)
  - Updated `_train_standard_real()` similarly
- **Result**: ❌ Still fails - quantization converts config to dict during loading

### Attempt 2: Enhanced Exception Handler
- **Changes:**
  - Improved exception handler to use project `model_type` when reloading
  - Added more robust config reloading logic
- **Result**: ❌ Fix attempt itself fails with same error

### Attempt 3: Load Without Config Parameter
- **Changes:**
  - Load model WITHOUT passing `config` parameter
  - Immediately fix `model.config = config` after loading
  - Added verification checks
- **Result**: ❌ Still fails - quantization process converts config to dict internally before we can fix it

## Current Code State

### Key Files Modified:
- `backend/app/workers/training_worker.py`
  - `_train_qlora_real()` - Lines ~670-1100
  - `_train_standard_real()` - Lines ~1694-1950
  - Both methods now:
    - Use `job._project.model_type` if available
    - Use direct `CONFIG_MAPPING[model_type]` access
    - Have extensive error handling for dict config issues

### Project Model Type Field:
- Projects now have a `model_type` field (e.g., "llama", "code_llama")
- This is set during project creation via the UI
- The training worker should use this instead of reading from config.json

## Next Steps / Options to Try

### Option 1: Load Model Without Quantization First
**Approach:**
1. Load model WITHOUT quantization
2. Fix `model.config` immediately
3. Apply quantization manually (if possible) or skip quantization for now

**Pros:** Model will load and training can proceed (without quantization)
**Cons:** QLoRA won't work as intended (needs quantization)

**Code Location:** Already partially implemented in exception handlers, but needs to be the PRIMARY path

### Option 2: Investigate Transformers/BitsAndBytes Versions
**Approach:**
- Check if this is a known issue with specific versions
- Try upgrading/downgrading transformers or bitsandbytes
- Check GitHub issues for similar problems

**Command to check versions:**
```bash
cd backend && source venv/bin/activate && pip list | grep -E "transformers|bitsandbytes|peft"
```

### Option 3: Use Different Quantization Method
**Approach:**
- Try 8-bit quantization instead of 4-bit
- Use different quantization config
- Check if there's a way to preserve config object during quantization

### Option 4: Patch Transformers Library
**Approach:**
- Monkey-patch the quantization loading to preserve config object
- Override the config conversion behavior
- This is risky and may break with library updates

### Option 5: Load Config After Model Loading
**Approach:**
- Don't pass config at all during model loading
- After model is loaded, immediately replace `model.config` with proper Config object
- This might work if we can do it before any internal access

**Implementation idea:**
```python
# Load model without config
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # NO config parameter
)

# IMMEDIATELY replace config before ANY other operation
# This must happen before tokenizer loading or prepare_model_for_kbit_training
model.config = config  # Our proper Config object

# Verify
if isinstance(model.config, dict):
    raise RuntimeError("Config still dict")
```

**Note:** This was tried in Attempt 3 but still failed. The issue is that quantization accesses `model.config.model_type` DURING loading, before we can fix it.

## Testing

### How to Reproduce:
```bash
# Create a project via API
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-dict-issue",
    "base_model": "meta-llama/Llama-3.2-3B",
    "model_type": "llama",
    "training_type": "qlora",
    "output_directory": "./output/test",
    "traits": [{
      "trait_type": "reasoning",
      "datasets": [{"dataset_id": 1, "percentage": 100}]
    }]
  }'

# Start training
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/start

# Check logs
tail -f .backend.log | grep -E "ERROR|dict|model_type"
```

### Expected Behavior:
- Model should load with proper Config object
- `prepare_model_for_kbit_training()` should succeed
- Training should proceed

### Current Behavior:
- Model loading fails with dict config error
- Error occurs during quantization loading
- Fix attempts also fail

## Related Files

- `backend/app/workers/training_worker.py` - Main training worker (2106 lines)
- `backend/app/models/project.py` - Project model with `model_type` field
- `backend/app/api/endpoints/projects.py` - Model types endpoint
- `docs/MODEL_TYPE_IMPLEMENTATION.md` - Model type implementation docs

## Environment Details

- **OS**: Manjaro Linux
- **Python**: 3.11+
- **Model**: meta-llama/Llama-3.2-3B
- **Model Type**: llama (or code_llama)
- **Training Type**: qlora (4-bit quantization)

### Library Versions (as of 2024-12-30)
- **transformers**: 4.57.2
- **bitsandbytes**: 0.49.0
- **peft**: 0.18.0
- **torch**: 2.9.1

## Notes

- The `model_type` field is now properly stored in projects
- The UI correctly sets `model_type` during project creation
- The backend correctly reads `model_type` from projects
- The issue is specifically with quantization loading converting config to dict

## References

- Transformers library: https://github.com/huggingface/transformers
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- PEFT (LoRA): https://github.com/huggingface/peft

## Status for Tomorrow

**Current State:**
- ✅ Model type field implemented and working
- ✅ Project creation sets model_type correctly
- ✅ Training worker reads model_type from project
- ❌ Model loading fails during quantization due to dict config conversion

**Next Priority:**
1. Try Option 5 (load without config, fix immediately) with more aggressive timing
2. If that fails, investigate Option 2 (version compatibility)
3. If still failing, consider Option 1 (skip quantization for now to unblock)

**Key Insight:**
The quantization library is accessing `model.config.model_type` DURING the `from_pretrained()` call, before we can fix it. We need to either:
- Prevent the dict conversion during loading
- Or find a way to intercept and fix it before the access happens
