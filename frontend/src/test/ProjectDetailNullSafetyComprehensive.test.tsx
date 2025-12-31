/**
 * Comprehensive tests for ProjectDetail null safety and edge cases.
 * 
 * Following TDD methodology: Tests ensure ProjectDetail handles all edge cases
 * including undefined project, null fields, and training job navigation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, useParams } from 'react-router-dom';
import ProjectDetail from '../components/ProjectDetail';
import { projectApi } from '../services/api';

// Mock useParams
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => ({ id: '1' }),
  };
});

// Mock API
vi.mock('../services/api', () => ({
  projectApi: {
    get: vi.fn(),
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

describe('ProjectDetail Comprehensive Null Safety Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('handles undefined project without crashing', async () => {
    (projectApi.get as any).mockResolvedValue(undefined);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    // Should show loading state, not crash
    await waitFor(() => {
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeTruthy();
    }, { timeout: 2000 });
  });

  it('handles null project.started_at in calculateSpeedStats', async () => {
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'pending',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 0,
      current_epoch: 0,
      started_at: null,
      completed_at: null,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should not crash and speed stats should not be displayed
    expect(screen.queryByText(/Training Speed Stats/i)).not.toBeInTheDocument();
  });

  it('handles undefined project.started_at in calculateSpeedStats', async () => {
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'pending',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 0,
      current_epoch: 0,
      started_at: undefined,
      completed_at: undefined,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should not crash
    expect(screen.queryByText(/Training Speed Stats/i)).not.toBeInTheDocument();
  });

  it('handles project with started_at but zero progress', async () => {
    const startedAt = new Date(Date.now() - 3600000);
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'pending',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 0,
      current_epoch: 0,
      started_at: startedAt.toISOString(),
      completed_at: null,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Speed stats should not be displayed for pending projects with 0 progress
    expect(screen.queryByText(/Training Speed Stats/i)).not.toBeInTheDocument();
  });

  it('handles project with null current_loss', async () => {
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'running',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 50,
      current_epoch: 2,
      current_loss: null,
      started_at: new Date(Date.now() - 3600000).toISOString(),
      completed_at: null,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should not crash when current_loss is null
    expect(screen.queryByText(/Loss:/i)).not.toBeInTheDocument();
  });

  it('handles project with undefined current_loss', async () => {
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'running',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 50,
      current_epoch: 2,
      current_loss: undefined,
      started_at: new Date(Date.now() - 3600000).toISOString(),
      completed_at: null,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should not crash when current_loss is undefined
    expect(screen.queryByText(/Loss:/i)).not.toBeInTheDocument();
  });

  it('calculates speed stats correctly for running project with all fields', async () => {
    const startedAt = new Date(Date.now() - 3600000);
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'running',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 50,
      current_epoch: 2,
      current_loss: 0.5,
      started_at: startedAt.toISOString(),
      completed_at: null,
      traits: [],
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    (projectApi.get as any).mockResolvedValue(mockProject);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Speed stats should be displayed for running projects
    await waitFor(() => {
      expect(screen.getByText(/Training Speed Stats/i)).toBeInTheDocument();
    });
  });

  it('handles project loading state without calling calculateSpeedStats', async () => {
    // Mock a delayed response
    (projectApi.get as any).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve({
        id: 1,
        name: 'Test Project',
        status: 'pending',
        base_model: 'test-model',
        training_type: 'qlora',
        progress: 0,
        current_epoch: 0,
        started_at: null,
        completed_at: null,
        traits: [],
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      }), 100))
    );

    render(<ProjectDetail />, { wrapper: createWrapper() });

    // Should show loading state first
    const spinner = document.querySelector('.animate-spin');
    expect(spinner).toBeTruthy();

    // Should not crash during loading
    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    }, { timeout: 2000 });
  });
});
