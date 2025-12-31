# TODO: Next Session

**Date**: 2024-12-30  
**Status**: Ready for next session

## Critical Issue: Model Loading Dict Config Error

### Problem
QLoRA training fails with: `'dict' object has no attribute 'model_type'`

### Immediate Next Steps (Priority Order)

1. **Try Option 5 with More Aggressive Timing**
   - Load model without config parameter
   - Use monkey-patch or context manager to intercept config access
   - Fix config BEFORE any internal access happens
   - File: `backend/app/workers/training_worker.py` - `_train_qlora_real()`

2. **Check Library Versions**
   - Document current versions (see below)
   - Search for known issues with these versions
   - Try upgrading/downgrading if needed

3. **Test Without Quantization**
   - Temporarily disable quantization to verify model loading works
   - This will confirm if issue is quantization-specific
   - If it works, we know the issue is with BitsAndBytes

4. **Alternative: Use 8-bit Quantization**
   - Try `load_in_8bit=True` instead of 4-bit
   - May have different config handling

## Current Library Versions (2024-12-30)

- **transformers**: 4.57.2
- **bitsandbytes**: 0.49.0
- **peft**: 0.18.0
- **torch**: 2.9.1

Check with: `cd backend && source venv/bin/activate && pip list | grep -E 'transformers|bitsandbytes|peft|torch'`

## Test Commands

```bash
# Create test project
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-dict-fix",
    "base_model": "meta-llama/Llama-3.2-3B",
    "model_type": "llama",
    "training_type": "qlora",
    "output_directory": "./output/test",
    "traits": [{"trait_type": "reasoning", "datasets": [{"dataset_id": 1, "percentage": 100}]}]
  }'

# Start and monitor
curl -X POST http://localhost:8000/api/v1/projects/{id}/start
tail -f .backend.log | grep -E "ERROR|dict|model_type|Config"
```

## Files to Review

- `backend/app/workers/training_worker.py` - Lines 670-1100 (`_train_qlora_real`)
- `docs/MODEL_LOADING_DICT_ISSUE.md` - Full issue documentation
- `docs/MODEL_TYPE_IMPLEMENTATION.md` - Model type implementation

## Success Criteria

- [ ] Model loads without dict config error
- [ ] `prepare_model_for_kbit_training()` succeeds
- [ ] Training proceeds past model loading
- [ ] Test project completes successfully (or at least gets past loading)

## Notes

- The `model_type` field is working correctly
- The issue is specifically with quantization loading
- All 3 attempts failed - need a different approach
