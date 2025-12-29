# Project-Based Training Tutorial

This tutorial walks you through creating and managing training projects with traits and dataset allocations.

## Overview

Projects allow you to create models with specific capabilities (traits):
- **Reasoning**: For logical reasoning tasks
- **Coding**: For code generation and understanding
- **General/Tools**: For general-purpose tasks, can combine multiple datasets

Each trait uses datasets with specified percentages of training data.

## Step-by-Step Guide

### Step 1: Prepare Datasets

Before creating a project, ensure you have datasets uploaded:

1. Go to **Datasets** page
2. Upload CSV or JSON files with training data
3. Verify datasets are processed and ready

### Step 2: Configure Output Directory

1. Go to **Settings**
2. Under "Directory Settings", set **Output Directory Base**
3. Test the directory is writable
4. Save settings

### Step 3: Create a Project

1. Go to **Projects** page
2. Click **New Project**

#### Step 3.1: Basic Information

- **Project Name**: Give your project a descriptive name
- **Description**: Optional description
- **Base Model**: Enter the HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`)
  - The system will validate the model is available
  - Use the model selector to browse available local models
- **Training Type**: Choose QLoRA, Unsloth, RAG, or Standard
- **Max Rows**: Select 50K, 100K, 250K, 500K, or 1M

Click **Next**.

#### Step 3.2: Reasoning Trait

- **Select Dataset**: Choose exactly one dataset
- **Percentage**: Automatically set to 100%
- Reasoning traits require exactly one dataset

Click **Next**.

#### Step 3.3: Coding Trait (Optional)

- **Add Coding Trait**: Click to add (optional)
- **Select Dataset**: Choose one dataset (must be different from reasoning)
- **Percentage**: Automatically 100%
- Coding traits require exactly one dataset

Click **Next**.

#### Step 3.4: General/Tools Trait (Optional)

- **Add General/Tools Trait**: Click to add (optional)
- **Add Multiple Datasets**: You can add multiple datasets here
- **Set Percentages**: Each dataset gets a percentage
- **Total Must Equal 100%**: The sum of all percentages must be exactly 100%
- Each dataset can only be used once per project

Click **Next**.

#### Step 3.5: Output Directory

- **Output Directory**: Enter the full path where the trained model will be saved
- The system validates the directory is writable
- Use the base directory from settings as a starting point

Click **Create Project**.

### Step 4: Start Training

1. Find your project in the list
2. Click **Start Training**
3. The system will:
   - Validate the model is available
   - Validate the output directory
   - Queue the project for processing
   - Combine datasets based on percentages
   - Start training

### Step 5: Monitor Progress

1. Click on your project to view details
2. Monitor:
   - Training progress percentage
   - Current epoch
   - Loss values
   - Training logs

### Step 6: Validate Model

After training completes:

1. The system automatically validates the model
2. You can manually trigger validation from the project detail page
3. Validation checks:
   - All files exist
   - Model can be loaded
   - Model format is valid

## Example Project Configurations

### Reasoning-Only Project

```json
{
  "name": "Reasoning Model",
  "traits": [
    {
      "trait_type": "reasoning",
      "datasets": [{"dataset_id": 1, "percentage": 100.0}]
    }
  ]
}
```

### Multi-Trait Project

```json
{
  "name": "Multi-Capability Model",
  "traits": [
    {
      "trait_type": "reasoning",
      "datasets": [{"dataset_id": 1, "percentage": 100.0}]
    },
    {
      "trait_type": "coding",
      "datasets": [{"dataset_id": 2, "percentage": 100.0}]
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

## Dataset Allocation Rules

1. **No Duplicate Datasets**: Each dataset can only be used once per project
2. **Reasoning**: Exactly 1 dataset, 100%
3. **Coding**: Exactly 1 dataset, 100%
4. **General/Tools**: 1+ datasets, percentages must sum to 100%
5. **Row Limits**: The system samples from datasets based on percentages and the max_rows setting

## Troubleshooting

### Percentage Doesn't Sum to 100%

- Ensure all percentages in General/Tools trait sum to exactly 100.0
- Use decimal values if needed (e.g., 33.33, 33.33, 33.34)

### Dataset Already Used

- Each dataset can only be used once across all traits
- Select different datasets for each trait

### Model Not Found

- Ensure the model is downloaded to HuggingFace cache
- Verify the model name is correct
- Use the model selector to see available models

### Output Directory Not Writable

- Check directory permissions
- Ensure the directory exists or can be created
- Use absolute paths

## Tips

- **Start Simple**: Create a reasoning-only project first
- **Validate Early**: Use validation features before starting training
- **Monitor Resources**: Large models and datasets require significant memory
- **Save Configurations**: Keep track of successful project configurations
- **Test Incrementally**: Test with smaller max_rows first
