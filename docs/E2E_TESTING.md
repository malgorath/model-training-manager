# End-to-End (E2E) Testing Documentation

**Date**: 2024-12-30  
**Author**: AI Assistant  
**Status**: In Progress

## Overview

This document describes the E2E testing setup using Playwright for testing the Model Training Manager application through the actual UI, without API mocking.

## Testing Philosophy

Following `.cursorrules` requirements:
- **User-Centric E2E tests only** (e.g., Playwright)
- **NO API MOCKING** - Tests interact with the real application
- Tests simulate real user behavior through the UI
- All tests must pass before considering work complete

## Setup

### Prerequisites

- Node.js 20+
- Playwright installed: `npx playwright install chromium`
- Backend server running on `http://localhost:8000`
- Frontend server running on `http://localhost:3001`

### Installation

```bash
cd frontend
npm install
npx playwright install chromium
```

### Configuration

**File**: `frontend/playwright.config.ts`

```typescript
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false, // Run tests sequentially
  workers: 1, // Run one test at a time
  use: {
    baseURL: 'http://localhost:3001',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: 'echo "Frontend should be running on http://localhost:3001"',
    url: 'http://localhost:3001',
    reuseExistingServer: true, // Don't start server, assume it's running
  },
});
```

## Test Structure

### Test File: `frontend/e2e/training-methods.spec.ts`

Tests all 4 training methods:
- QLoRA
- Unsloth
- RAG
- Standard

### Test Flow

Each test simulates a real user:

1. **Navigate to Projects Page**
   - Opens `/projects`
   - Waits for page to load

2. **Click "New Project" Button**
   - Finds and clicks the "New Project" button
   - Waits for form modal to appear

3. **Fill Step 1: Basic Info**
   - Enter project name
   - Select base model from dropdown
   - Wait for model validation
   - Select model type (auto-detected)
   - Select training type

4. **Click Next to Step 2**
   - Validates all required fields are filled
   - Clicks "Next" button

5. **Fill Step 2: Reasoning Trait**
   - Select dataset from dropdown
   - Set percentage (defaults to 100%)

6. **Submit Form**
   - Clicks "Create Project" button
   - Waits for success

7. **Start Training**
   - Finds project in list
   - Clicks "Start" button

8. **Verify Dashboard Visibility**
   - Navigates to dashboard
   - Verifies project appears

## Running Tests

### Run All Tests

```bash
cd frontend
npx playwright test e2e/training-methods.spec.ts
```

### Run with UI (Headed Mode)

```bash
npx playwright test e2e/training-methods.spec.ts --headed
```

### Run with Debugging

```bash
npx playwright test e2e/training-methods.spec.ts --debug
```

### View Test Report

```bash
npx playwright show-report
```

## Test Helpers

### `fillProjectForm()`

Fills out the multi-step project creation form:
- Handles model selection and validation
- Waits for model type detection
- Selects datasets
- Handles form state transitions

### `startTraining()`

Finds a project in the list and starts training:
- Locates project by name
- Clicks start button
- Handles loading states

### `waitForDatasets()`

Waits for datasets to be available in dropdowns:
- Polls for dataset options
- Handles async loading

## Current Status

### Completed
- ✅ Test structure and configuration
- ✅ Basic form filling logic
- ✅ Model selection and validation
- ✅ Model type selection

### In Progress
- 🔄 Dataset selection (needs refinement)
- 🔄 Form submission verification
- 🔄 Training start verification

### Known Issues

1. **Dataset Selection**: The test sometimes fails to find the dataset dropdown on Step 2. This is being refined.

2. **Form State**: Need to ensure proper waiting for form state transitions between steps.

3. **Error Handling**: Need better error messages when selectors fail.

## Best Practices

1. **Use Explicit Waits**: Always wait for elements to be visible/enabled before interacting
2. **Retry Logic**: Implement retry logic for flaky operations (max 3 attempts per `.cursorrules`)
3. **Screenshots**: Tests automatically capture screenshots on failure
4. **Videos**: Tests record videos for debugging failures
5. **No API Mocking**: All tests use real backend and frontend

## Integration with CI/CD

Tests should be run:
- Before every commit
- In CI/CD pipeline
- After major feature changes

## Related Files

- `frontend/e2e/training-methods.spec.ts` - Main test file
- `frontend/playwright.config.ts` - Playwright configuration
- `.cursorrules` - Testing requirements and rules

## Future Improvements

- Add tests for other pages (Datasets, Settings, etc.)
- Add tests for error scenarios
- Add performance testing
- Add accessibility testing
- Parallel test execution (when safe)
