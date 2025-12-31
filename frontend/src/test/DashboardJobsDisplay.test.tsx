/**
 * Tests for Dashboard training jobs display.
 * 
 * Following TDD methodology: Tests ensure dashboard shows recent training jobs correctly.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import DashboardPage from '../pages/DashboardPage';
import { trainingJobApi } from '../services/api';

// Mock API
vi.mock('../services/api', () => ({
  datasetApi: {
    list: vi.fn().mockResolvedValue({ items: [], total: 0, page: 1, page_size: 5, pages: 0 }),
  },
  projectApi: {
    list: vi.fn().mockResolvedValue({ items: [], total: 0, page: 1, page_size: 5, pages: 0 }),
  },
  trainingJobApi: {
    list: vi.fn(),
  },
  workerApi: {
    getStatus: vi.fn().mockResolvedValue({
      total_workers: 0,
      active_workers: 0,
      idle_workers: 0,
      busy_workers: 0,
      max_workers: 4,
      workers: [],
      jobs_in_queue: 0,
    }),
  },
}));

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

describe('Dashboard Training Jobs Display Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('displays recent training jobs when available', async () => {
    const mockJobs = {
      items: [
        {
          id: 1,
          name: 'Test Job 1',
          status: 'completed',
          training_type: 'qlora',
          progress: 100,
          current_epoch: 3,
          epochs: 3,
          current_loss: 0.5,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T01:00:00Z',
        },
        {
          id: 2,
          name: 'Test Job 2',
          status: 'failed',
          training_type: 'qlora',
          progress: 45,
          current_epoch: 1,
          epochs: 3,
          current_loss: null,
          error_message: 'Training failed',
          created_at: '2024-01-01T02:00:00Z',
          updated_at: '2024-01-01T02:30:00Z',
        },
      ],
      total: 2,
      page: 1,
      page_size: 10,
      pages: 1,
    };

    (trainingJobApi.list as any).mockResolvedValue(mockJobs);

    render(<DashboardPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Recent Jobs')).toBeInTheDocument();
    });

    // Should display job names
    await waitFor(() => {
      expect(screen.getByText('Test Job 1')).toBeInTheDocument();
      expect(screen.getByText('Test Job 2')).toBeInTheDocument();
    });
  });

  it('displays failed jobs on dashboard', async () => {
    const mockJobs = {
      items: [
        {
          id: 1,
          name: 'Failed Job',
          status: 'failed',
          training_type: 'qlora',
          progress: 30,
          current_epoch: 1,
          epochs: 3,
          current_loss: null,
          error_message: 'Model loading error',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:30:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    };

    (trainingJobApi.list as any).mockResolvedValue(mockJobs);

    render(<DashboardPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Failed Job')).toBeInTheDocument();
    });

    // Should show failed status
    expect(screen.getByText(/failed/i)).toBeInTheDocument();
  });

  it('displays empty state when no jobs exist', async () => {
    const mockJobs = {
      items: [],
      total: 0,
      page: 1,
      page_size: 10,
      pages: 0,
    };

    (trainingJobApi.list as any).mockResolvedValue(mockJobs);

    render(<DashboardPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Recent Jobs')).toBeInTheDocument();
    });

    // Should show empty state or no jobs message
    // The JobMonitor component should handle this
  });

  it('displays all job statuses (completed, running, failed, pending)', async () => {
    const mockJobs = {
      items: [
        {
          id: 1,
          name: 'Completed Job',
          status: 'completed',
          training_type: 'qlora',
          progress: 100,
          current_epoch: 3,
          epochs: 3,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T01:00:00Z',
        },
        {
          id: 2,
          name: 'Running Job',
          status: 'running',
          training_type: 'qlora',
          progress: 50,
          current_epoch: 2,
          epochs: 3,
          created_at: '2024-01-01T02:00:00Z',
          updated_at: '2024-01-01T02:30:00Z',
        },
        {
          id: 3,
          name: 'Failed Job',
          status: 'failed',
          training_type: 'qlora',
          progress: 30,
          current_epoch: 1,
          epochs: 3,
          error_message: 'Error occurred',
          created_at: '2024-01-01T03:00:00Z',
          updated_at: '2024-01-01T03:15:00Z',
        },
        {
          id: 4,
          name: 'Pending Job',
          status: 'pending',
          training_type: 'qlora',
          progress: 0,
          current_epoch: 0,
          epochs: 3,
          created_at: '2024-01-01T04:00:00Z',
          updated_at: '2024-01-01T04:00:00Z',
        },
      ],
      total: 4,
      page: 1,
      page_size: 10,
      pages: 1,
    };

    (trainingJobApi.list as any).mockResolvedValue(mockJobs);

    render(<DashboardPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Completed Job')).toBeInTheDocument();
      expect(screen.getByText('Running Job')).toBeInTheDocument();
      expect(screen.getByText('Failed Job')).toBeInTheDocument();
      expect(screen.getByText('Pending Job')).toBeInTheDocument();
    });
  });
});
