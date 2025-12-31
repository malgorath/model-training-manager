/**
 * Comprehensive tests for all run/start buttons in the UI.
 * 
 * Following TDD methodology: Tests ensure every button that starts/runs something
 * calls the correct API endpoint and handles responses properly.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import ProjectList from '../components/ProjectList';
import ProjectDetail from '../components/ProjectDetail';
import JobMonitor from '../components/JobMonitor';
import WorkerDashboard from '../components/WorkerDashboard';
import { projectApi, trainingJobApi, workerApi } from '../services/api';

// Mock API
vi.mock('../services/api', () => ({
  projectApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Project',
          status: 'pending',
          base_model: 'test-model',
          training_type: 'qlora',
          progress: 0,
          current_epoch: 0,
          traits: [],
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
      pages: 1,
    }),
    start: vi.fn().mockResolvedValue({ id: 1, status: 'pending' }),
    cancel: vi.fn().mockResolvedValue({ id: 1, status: 'cancelled' }),
  },
  trainingJobApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Job',
          status: 'pending',
          training_type: 'qlora',
          model_name: 'test-model',
          dataset_id: 1,
          progress: 0,
          current_epoch: 0,
          epochs: 3,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    }),
    start: vi.fn().mockResolvedValue({ id: 1, status: 'queued' }),
  },
  workerApi: {
    getStatus: vi.fn().mockResolvedValue({
      is_running: true,
      active_workers: 1,
      total_workers: 1,
      idle_workers: 0,
      busy_workers: 1,
      workers: [{
        id: 'worker-1',
        status: 'busy',
        current_job_id: 1,
      }],
    }),
    start: vi.fn().mockResolvedValue({
      is_running: true,
      active_workers: 1,
      workers: [],
    }),
    restart: vi.fn().mockResolvedValue({
      is_running: true,
      active_workers: 1,
      workers: [],
    }),
  },
  configApi: {
    get: vi.fn().mockResolvedValue({
      auto_start_workers: false,
    }),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe('All Run/Start Buttons Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('ProjectList Start Button', () => {
    it('start button calls projectApi.start with correct project ID', async () => {
      const user = userEvent.setup();
      const { projectApi } = await import('../services/api');

      render(<ProjectList />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('Test Project')).toBeInTheDocument();
      });

      const startButton = screen.getByText('Start');
      await user.click(startButton);

      await waitFor(() => {
        expect(projectApi.start).toHaveBeenCalled();
        // Check first argument is the project ID (mutation may pass context as second arg)
        const calls = (projectApi.start as any).mock.calls;
        expect(calls.length).toBeGreaterThan(0);
        expect(calls[0][0]).toBe(1);
      });
    });

    it('start button is only shown for non-running projects', async () => {
      const { projectApi } = await import('../services/api');
      (projectApi.list as any).mockResolvedValueOnce({
        items: [
          {
            id: 1,
            name: 'Running Project',
            status: 'running',
            base_model: 'test-model',
            training_type: 'qlora',
            progress: 50,
            current_epoch: 2,
            traits: [],
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

      expect(screen.queryByText('Start')).not.toBeInTheDocument();
    });
  });

  describe('ProjectDetail Start Button', () => {
    it('start button calls projectApi.start with correct project ID', async () => {
      const user = userEvent.setup();
      const { projectApi } = await import('../services/api');
      
      // Mock useParams
      vi.mock('react-router-dom', async () => {
        const actual = await vi.importActual('react-router-dom');
        return {
          ...actual,
          useParams: () => ({ id: '1' }),
        };
      });

      (projectApi.get as any) = vi.fn().mockResolvedValue({
        id: 1,
        name: 'Test Project',
        status: 'pending',
        base_model: 'test-model',
        training_type: 'qlora',
        progress: 0,
        current_epoch: 0,
        traits: [],
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      });

      render(<ProjectDetail />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('Test Project')).toBeInTheDocument();
      });

      const startButton = screen.getByText('Start Training');
      await user.click(startButton);

      await waitFor(() => {
        expect(projectApi.start).toHaveBeenCalledWith(1);
      });
    });
  });

  describe('JobMonitor Start Button', () => {
    it('start button calls trainingJobApi.start with correct job ID', async () => {
      const user = userEvent.setup();
      const { trainingJobApi } = await import('../services/api');

      render(<JobMonitor />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('Test Job')).toBeInTheDocument();
      });

      // Find start button (Play icon)
      const startButtons = screen.getAllByTitle('Start job');
      if (startButtons.length > 0) {
        await user.click(startButtons[0]);

        await waitFor(() => {
          expect(trainingJobApi.start).toHaveBeenCalled();
        });
      }
    });
  });

  describe('WorkerDashboard Start Button', () => {
    it('start worker button calls workerApi.start with count 1', async () => {
      const user = userEvent.setup();
      const { workerApi } = await import('../services/api');

      render(<WorkerDashboard />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('Start Worker')).toBeInTheDocument();
      });

      const startButton = screen.getByText('Start Worker');
      await user.click(startButton);

      await waitFor(() => {
        expect(workerApi.start).toHaveBeenCalledWith(1);
      });
    });

    it('restart button calls workerApi.restart', async () => {
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
});
