/**
 * Tests for ProjectForm component - Output Directory Auto-fill functionality.
 * 
 * Following TDD methodology: Tests define expected behavior before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Mock API modules
vi.mock('../services/api', () => ({
  datasetApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Dataset',
          filename: 'test.csv',
          file_type: 'csv',
          file_size: 1024,
          row_count: 100,
          column_count: 2,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    }),
  },
  configApi: {
    get: vi.fn().mockResolvedValue({
      id: 1,
      max_concurrent_workers: 4,
      active_workers: 2,
      default_model: 'llama3.2:3b',
      default_training_type: 'qlora',
      default_batch_size: 4,
      default_learning_rate: 0.0002,
      default_epochs: 3,
      default_lora_r: 16,
      default_lora_alpha: 32,
      default_lora_dropout: 0.05,
      auto_start_workers: false,
      output_directory_base: './output',
    }),
  },
  projectApi: {
    listAvailableModels: vi.fn().mockResolvedValue(['llama3.2:3b', 'meta-llama/Llama-3.2-3B-Instruct']),
    validateOutputDir: vi.fn().mockResolvedValue({ valid: true, writable: true }),
    validateModel: vi.fn().mockResolvedValue({ available: true }),
    create: vi.fn().mockResolvedValue({ id: 1, name: 'Test Project' }),
  },
}));

// Import component after mocks
import ProjectForm from '../components/ProjectForm';

/**
 * Create a wrapper with providers for testing.
 */
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe('ProjectForm - Output Directory Auto-fill', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('allows project creation without clicking in output directory field', async () => {
    const user = userEvent.setup();
    render(<ProjectForm />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Create Project')).toBeInTheDocument();
    });

    // Enter project name
    const nameInput = screen.getByLabelText(/Project Name/i) as HTMLInputElement;
    await user.type(nameInput, 'Test Project');

    // Select base model
    const baseModelSelect = screen.getByLabelText(/Base Model/i) as HTMLSelectElement;
    await user.selectOptions(baseModelSelect, 'llama3.2:3b');

    // Navigate through all steps
    const nextButton = screen.getByText('Next');
    await user.click(nextButton); // Step 2
    
    // Select dataset for reasoning trait
    await waitFor(() => {
      const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
      expect(datasetSelect).toBeInTheDocument();
    });
    const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
    await user.selectOptions(datasetSelect, '1');

    await user.click(screen.getByText('Next')); // Step 3
    await user.click(screen.getByText('Next')); // Step 4
    await user.click(screen.getByText('Next')); // Step 5

    // Verify output directory is auto-filled
    await waitFor(() => {
      const outputDirInput = screen.getByLabelText(/Output Directory/i) as HTMLInputElement;
      expect(outputDirInput.value).toBe('./output/test-project');
    });

    // Verify create button is enabled without clicking in output directory field
    await waitFor(() => {
      const createButton = screen.getByText('Create Project');
      expect(createButton).not.toBeDisabled();
    });
  });

  it('does not require output directory validation to create project', async () => {
    const user = userEvent.setup();
    render(<ProjectForm />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Create Project')).toBeInTheDocument();
    });

    // Fill form and navigate to step 5
    const nameInput = screen.getByLabelText(/Project Name/i) as HTMLInputElement;
    await user.type(nameInput, 'Test Project');

    const baseModelSelect = screen.getByLabelText(/Base Model/i) as HTMLSelectElement;
    await user.selectOptions(baseModelSelect, 'llama3.2:3b');

    const nextButton = screen.getByText('Next');
    await user.click(nextButton);
    
    await waitFor(() => {
      const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
      expect(datasetSelect).toBeInTheDocument();
    });
    const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
    await user.selectOptions(datasetSelect, '1');

    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));

    // Verify create button is enabled even if validation hasn't run
    await waitFor(() => {
      const createButton = screen.getByText('Create Project');
      expect(createButton).not.toBeDisabled();
    });
  });

  it('auto-fills output directory with base + project name when name is entered', async () => {
    const user = userEvent.setup();
    render(<ProjectForm />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Create Project')).toBeInTheDocument();
    });

    // Enter project name
    const nameInput = screen.getByLabelText(/Project Name/i) as HTMLInputElement;
    await user.type(nameInput, 'My Test Project');

    // Select base model
    const baseModelSelect = screen.getByLabelText(/Base Model/i) as HTMLSelectElement;
    await user.selectOptions(baseModelSelect, 'llama3.2:3b');

    // Navigate to step 2 - need to select a dataset for reasoning trait
    const nextButton = screen.getByText('Next');
    await user.click(nextButton); // Step 2
    
    // Select dataset for reasoning trait
    await waitFor(() => {
      const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
      expect(datasetSelect).toBeInTheDocument();
    });
    const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
    await user.selectOptions(datasetSelect, '1');

    // Navigate through remaining steps
    await user.click(screen.getByText('Next')); // Step 3
    await user.click(screen.getByText('Next')); // Step 4
    await user.click(screen.getByText('Next')); // Step 5

    await waitFor(() => {
      const outputDirInput = screen.getByLabelText(/Output Directory/i) as HTMLInputElement;
      expect(outputDirInput.value).toBe('./output/my-test-project');
    }, { timeout: 5000 });
  });

  it('updates output directory when project name changes in step 1', async () => {
    const user = userEvent.setup();
    render(<ProjectForm />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByLabelText(/Project Name/i)).toBeInTheDocument();
    });

    const nameInput = screen.getByLabelText(/Project Name/i) as HTMLInputElement;
    await user.type(nameInput, 'First Project');

    const baseModelSelect = screen.getByLabelText(/Base Model/i) as HTMLSelectElement;
    await user.selectOptions(baseModelSelect, 'llama3.2:3b');

    // Navigate to step 2 and select dataset
    const nextButton = screen.getByText('Next');
    await user.click(nextButton);
    
    await waitFor(() => {
      const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
      expect(datasetSelect).toBeInTheDocument();
    });
    const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
    await user.selectOptions(datasetSelect, '1');

    // Navigate to step 5
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));

    await waitFor(() => {
      const outputDirInput = screen.getByLabelText(/Output Directory/i) as HTMLInputElement;
      expect(outputDirInput.value).toBe('./output/first-project');
    }, { timeout: 5000 });

    // Go back and change project name
    await user.click(screen.getByText('Previous'));
    await user.click(screen.getByText('Previous'));
    await user.click(screen.getByText('Previous'));
    await user.click(screen.getByText('Previous'));

    await user.clear(nameInput);
    await user.type(nameInput, 'Second Project');

    // Navigate back to step 5
    await user.click(screen.getByText('Next'));
    await waitFor(() => {
      const datasetSelectAgain = screen.getByRole('combobox') as HTMLSelectElement;
      if (datasetSelectAgain) {
        user.selectOptions(datasetSelectAgain, '1');
      }
    });
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));

    await waitFor(() => {
      const outputDirInput = screen.getByLabelText(/Output Directory/i) as HTMLInputElement;
      expect(outputDirInput.value).toBe('./output/second-project');
    }, { timeout: 5000 });
  });

  it('sanitizes project name in output directory path', async () => {
    const user = userEvent.setup();
    render(<ProjectForm />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByLabelText(/Project Name/i)).toBeInTheDocument();
    });

    const nameInput = screen.getByLabelText(/Project Name/i) as HTMLInputElement;
    await user.type(nameInput, 'My Project Name!');

    const baseModelSelect = screen.getByLabelText(/Base Model/i) as HTMLSelectElement;
    await user.selectOptions(baseModelSelect, 'llama3.2:3b');

    const nextButton = screen.getByText('Next');
    await user.click(nextButton);
    
    // Select dataset for reasoning trait
    await waitFor(() => {
      const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
      expect(datasetSelect).toBeInTheDocument();
    });
    const datasetSelect = screen.getByRole('combobox') as HTMLSelectElement;
    await user.selectOptions(datasetSelect, '1');

    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));

    await waitFor(() => {
      const outputDirInput = screen.getByLabelText(/Output Directory/i) as HTMLInputElement;
      // Should sanitize: spaces to hyphens, special chars to underscores, lowercase
      expect(outputDirInput.value).toBe('./output/my-project-name_');
      expect(outputDirInput.value).not.toContain(' ');
      expect(outputDirInput.value).not.toContain('!');
    }, { timeout: 5000 });
  });
});
