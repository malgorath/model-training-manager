# Model Resolution Guide

This guide explains how the Model Training Manager resolves and validates models for training.

## Overview

The Model Resolution Service is responsible for locating models on your system, validating their format, and ensuring they're available before training starts. This prevents training failures due to missing or corrupted models.

## Model Resolution Process

When you start a training project, the system:

1. **Checks Model Overrides**: First checks if you've configured a custom local path for the model
2. **Searches HuggingFace Cache**: Looks in the default HuggingFace cache directory (`~/.cache/huggingface/hub`)
3. **Validates Format**: Ensures the model has all required files (config.json, tokenizer files, etc.)
4. **Fails Explicitly**: If the model isn't found, training fails immediately with a clear error message (no simulation fallback)

## Model Cache Locations

### Default HuggingFace Cache

The system checks the HuggingFace cache at:
```
~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
```

For example, `meta-llama/Llama-3.2-3B-Instruct` would be found at:
```
~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/{hash}/
```

### Configuring Custom Cache Path

You can configure a custom cache path in Settings:
1. Go to Settings
2. Find "Directory Settings"
3. Set "Model Cache Path" to your custom location
4. Save changes

## Model Path Overrides

For models stored in custom locations, you can configure path overrides:

1. Set the `model_cache_path` in settings
2. Or configure overrides via the API (future feature)

## Model Format Requirements

A valid model must have:

### Required Files
- `config.json`: Model configuration in JSON format
- At least one tokenizer file:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `tokenizer.model`

### Recommended Files
- Model weights: `*.safetensors` or `*.bin` files
- `generation_config.json`: Generation parameters

## Model Validation

After training completes, the system automatically validates:

1. **File Checks**: All required files exist
2. **Format Validation**: JSON files are valid
3. **Loading Test**: Model can be loaded successfully
4. **Inference Test**: Model can perform inference (optional, future feature)

## Troubleshooting

### Model Not Found Error

**Error**: `Model 'xxx' not found. Please ensure the model is downloaded...`

**Solutions**:
1. Download the model using HuggingFace CLI:
   ```bash
   huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
   ```

2. Verify the model is in cache:
   ```bash
   ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/
   ```

3. Configure a custom path if the model is elsewhere

### Model Format Error

**Error**: `Model format invalid: missing config.json...`

**Solutions**:
1. Re-download the model
2. Ensure you have the full model (not just tokenizer)
3. Check file permissions

### Permission Errors

**Error**: `Cannot write to directory...`

**Solutions**:
1. Check directory permissions: `ls -la /output/path`
2. Ensure you have write access
3. Create the directory if it doesn't exist

## Best Practices

1. **Pre-download Models**: Download models before creating projects
2. **Verify Models**: Use the "Validate Model" feature in the project form
3. **Use Absolute Paths**: Specify full paths for output directories
4. **Check Cache Space**: HuggingFace models can be large (several GB each)

## API Usage

### Check Model Availability

```bash
curl -X POST http://localhost:8000/api/v1/projects/validate-model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

### List Available Models

```bash
curl http://localhost:8000/api/v1/projects/models/available
```

### Validate Output Directory

```bash
curl -X POST http://localhost:8000/api/v1/projects/validate-output-dir \
  -H "Content-Type: application/json" \
  -d '{"output_directory": "/output/my-project"}'
```
