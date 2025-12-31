# Integration Tests for Training Methods

## Overview

These integration tests simulate a real user using the frontend UI to test all training methods. They use the **actual database** so results are visible in the dashboard for review.

## Running the Tests

```bash
# Make sure the backend server is running
cd backend
source venv/bin/activate
python3 tests/integration_test_training_methods_ui.py
```

Or make it executable and run directly:

```bash
chmod +x backend/tests/integration_test_training_methods_ui.py
./backend/tests/integration_test_training_methods_ui.py
```

## What the Tests Do

1. **Create Test Datasets**: Creates small datasets (20 rows each) for each training method
2. **Load Available Models**: Fetches available models from the dropdown (same as UI)
3. **Create Projects**: Creates projects via API for each training method:
   - QLoRA
   - Unsloth
   - RAG
   - Standard
4. **Start Training**: Starts training for each project (simulates clicking "Start Training")
5. **Wait for Completion**: Monitors training progress until completion
6. **Verify Dashboard Visibility**: Ensures all projects are visible in the dashboard

## Requirements

- Backend server must be running on `http://localhost:8000`
- At least one model must be downloaded and available
- Database must be accessible (uses actual database, not test database)

## Test Results

After running, you can:
- View all test projects in the Projects page
- View all test jobs in the Training Jobs page
- Review training logs, progress, and results
- Verify all training methods work correctly

## Test Output

The script will print:
- Progress for each step
- Status of each training method
- Summary of all tests
- Project IDs for dashboard review

## Notes

- Tests use small datasets (20 rows) for quick execution
- All results persist in the database for dashboard review
- Tests wait up to 10 minutes per training method
- If a test fails, it will continue with the next method
