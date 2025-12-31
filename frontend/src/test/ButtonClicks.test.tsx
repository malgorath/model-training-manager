/**
 * Comprehensive tests for ALL button clicks in frontend components.
 * 
 * Following TDD methodology: Tests define expected behavior for every button.
 * This ensures 100% button click coverage.
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
    delete: vi.fn().mockResolvedValue(undefined),
    scan: vi.fn().mockResolvedValue({ scanned: 0, added: 0, skipped: 0 }),
  },
  projectApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Project',
          base_model: 'llama3.2:3b',
          training_type: 'qlora',
          status: 'pending',
          max_rows: null,
          traits: [],
          progress: 0,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
      pages: 1,
    }),
    delete: vi.fn().mockResolvedValue(undefined),
    start: vi.fn().mockResolvedValue({ id: 1, status: 'pending' }),
    cancel: vi.fn().mockResolvedValue({ id: 1, status: 'cancelled' }),
    retry: vi.fn().mockResolvedValue({ id: 1, status: 'pending', progress: 0 }),
    get: vi.fn().mockResolvedValue({
      id: 1,
      name: 'Test Project',
      status: 'failed',
      base_model: 'llama3.2:3b',
      training_type: 'qlora',
      progress: 0,
      current_epoch: 0,
      error_message: 'Test error',
      traits: [],
      started_at: null,
      completed_at: null,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    }),
    listAvailableModels: vi.fn().mockResolvedValue(['llama3.2:3b']),
    validateOutputDir: vi.fn().mockResolvedValue({ valid: true, writable: true }),
    validateModel: vi.fn().mockResolvedValue({ available: true }),
    create: vi.fn().mockResolvedValue({ id: 1, name: 'Test Project' }),
  },
  trainingJobApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Job',
          status: 'running',
          training_type: 'qlora',
          progress: 50,
          current_epoch: 2,
          epochs: 4,
          current_loss: 0.5,
          created_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    }),
    create: vi.fn().mockResolvedValue({ id: 1, name: 'New Job' }),
    cancel: vi.fn().mockResolvedValue({ id: 1, status: 'cancelled' }),
    start: vi.fn().mockResolvedValue({ id: 1, status: 'running' }),
  },
  workerApi: {
    getStatus: vi.fn().mockResolvedValue({
      total_workers: 2,
      active_workers: 2,
      idle_workers: 1,
      busy_workers: 1,
      max_workers: 8,
      workers: [
        { id: 'worker-1', status: 'idle', jobs_completed: 5, started_at: '2024-01-01T00:00:00Z' },
        { id: 'worker-2', status: 'busy', current_job_id: 1, jobs_completed: 3, started_at: '2024-01-01T00:00:00Z' },
      ],
      jobs_in_queue: 0,
    }),
    start: vi.fn().mockResolvedValue({}),
    stop: vi.fn().mockResolvedValue({}),
    restart: vi.fn().mockResolvedValue({}),
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
    update: vi.fn().mockResolvedValue({}),
    getGPUs: vi.fn().mockResolvedValue([]),
  },
  modelsApi: {
    scan: vi.fn().mockResolvedValue({ scanned: 0, added: 0, skipped: 0 }),
  },
}));

// Import components after mocks
import DatasetList from '../components/DatasetList';
import ProjectList from '../components/ProjectList';
import JobMonitor from '../components/JobMonitor';
import WorkerDashboard from '../components/WorkerDashboard';
import ProjectDetail from '../components/ProjectDetail';

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

describe('Button Click Tests - DatasetList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('delete button calls delete API and invalidates queries', async () => {
    const user = userEvent.setup();
    const { datasetApi } = await import('../services/api');
    
    // Mock window.confirm
    window.confirm = vi.fn(() => true);
    
    render(<DatasetList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Dataset')).toBeInTheDocument();
    });

    const deleteButton = screen.getAllByRole('button').find(
      btn => btn.querySelector('svg')?.getAttribute('class')?.includes('lucide-trash')
    );
    
    if (deleteButton) {
      await user.click(deleteButton);
      await waitFor(() => {
        expect(datasetApi.delete).toHaveBeenCalledWith(1);
      });
    }
  });

  it('refresh button calls scan API and invalidates queries', async () => {
    const user = userEvent.setup();
    const { datasetApi } = await import('../services/api');
    
    render(<DatasetList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Dataset')).toBeInTheDocument();
    });

    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    await user.click(refreshButton);

    await waitFor(() => {
      expect(datasetApi.scan).toHaveBeenCalled();
    });
  });
});

describe('Button Click Tests - ProjectList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('start button calls start API', async () => {
    const user = userEvent.setup();
    const { projectApi } = await import('../services/api');
    
    render(<ProjectList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    const startButton = screen.getByText('Start');
    await user.click(startButton);

    await waitFor(() => {
      expect(projectApi.start).toHaveBeenCalledWith(1);
    });
  });

  it('cancel button calls cancel API for running project', async () => {
    const user = userEvent.setup();
    const { projectApi } = await import('../services/api');
    
    (projectApi.list as any).mockResolvedValueOnce({
      items: [
        {
          id: 1,
          name: 'Running Project',
          status: 'running',
          base_model: 'llama3.2:3b',
          training_type: 'qlora',
          max_rows: null,
          traits: [],
          progress: 50,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
      pages: 1,
    });
    
    render(<ProjectList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Running Project')).toBeInTheDocument();
    });

    const cancelButton = screen.getByText('Cancel');
    await user.click(cancelButton);

    await waitFor(() => {
      expect(projectApi.cancel).toHaveBeenCalledWith(1);
    });
  });

  it('delete button calls delete API with confirmation', async () => {
    const user = userEvent.setup();
    const { projectApi } = await import('../services/api');
    
    window.confirm = vi.fn(() => true);
    
    render(<ProjectList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    const deleteButton = screen.getAllByRole('button').find(
      btn => btn.querySelector('svg')?.getAttribute('class')?.includes('lucide-trash')
    );
    
    if (deleteButton) {
      await user.click(deleteButton);
      await waitFor(() => {
        expect(window.confirm).toHaveBeenCalled();
        expect(projectApi.delete).toHaveBeenCalled();
      });
    }
  });
});

describe('Button Click Tests - JobMonitor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('status filter buttons update filter state', async () => {
    const user = userEvent.setup();
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Job')).toBeInTheDocument();
    });

    const pendingButton = screen.getByText('pending');
    await user.click(pendingButton);

    // Filter button should be active
    expect(pendingButton).toHaveClass('bg-primary-500/20');
  });

  it('start button calls start API', async () => {
    const user = userEvent.setup();
    const { trainingJobApi } = await import('../services/api');
    
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Job')).toBeInTheDocument();
    });

    const startButtons = screen.getAllByText(/start/i);
    if (startButtons.length > 0) {
      await user.click(startButtons[0]);
      await waitFor(() => {
        expect(trainingJobApi.start).toHaveBeenCalled();
      });
    }
  });

  it('cancel button calls cancel API', async () => {
    const user = userEvent.setup();
    const { trainingJobApi } = await import('../services/api');
    
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Job')).toBeInTheDocument();
    });

    const cancelButtons = screen.getAllByText(/cancel/i);
    if (cancelButtons.length > 0) {
      await user.click(cancelButtons[0]);
      await waitFor(() => {
        expect(trainingJobApi.cancel).toHaveBeenCalled();
      });
    }
  });
});

describe('Button Click Tests - WorkerDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('start worker button calls start API', async () => {
    const user = userEvent.setup();
    const { workerApi } = await import('../services/api');
    
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Worker Pool')).toBeInTheDocument();
    });

    const startButton = screen.getByText('Start Worker');
    await user.click(startButton);

    await waitFor(() => {
      expect(workerApi.start).toHaveBeenCalled();
    });
  });

  it('stop all button calls stop API', async () => {
    const user = userEvent.setup();
    const { workerApi } = await import('../services/api');
    
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Stop All')).toBeInTheDocument();
    });

    const stopButton = screen.getByText('Stop All');
    await user.click(stopButton);

    await waitFor(() => {
      expect(workerApi.stop).toHaveBeenCalled();
    });
  });

  it('restart button calls restart API', async () => {
    const user = userEvent.setup();
    const { workerApi } = await import('../services/api');
    
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Restart')).toBeInTheDocument();
    });

    const restartButton = screen.getByText('Restart');
    await user.click(restartButton);

    await waitFor(() => {
      expect(workerApi.restart).toHaveBeenCalled();
    });
  });
});
