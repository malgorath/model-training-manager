/**
 * Tests for ProjectDetail null safety.
 * 
 * Following TDD methodology: Tests ensure ProjectDetail handles null/undefined project gracefully.
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

describe('ProjectDetail Null Safety Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('handles undefined project gracefully without crashing', async () => {
    // Mock API to return undefined initially
    (projectApi.get as any).mockResolvedValue(undefined);

    render(<ProjectDetail />, { wrapper: createWrapper() });

    // Should show loading spinner
    await waitFor(() => {
      const spinner = screen.queryByRole('status') || 
                     document.querySelector('.animate-spin');
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

    // Should render without crashing
    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Speed stats should not be displayed for projects without started_at
    expect(screen.queryByText(/Training Speed Stats/i)).not.toBeInTheDocument();
  });

  it('handles project with started_at but no progress', async () => {
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'pending',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 0,
      current_epoch: 0,
      started_at: '2024-01-01T00:00:00Z',
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

    // Speed stats should not be displayed for pending projects
    expect(screen.queryByText(/Training Speed Stats/i)).not.toBeInTheDocument();
  });

  it('calculates speed stats correctly for running project', async () => {
    const startedAt = new Date(Date.now() - 3600000); // 1 hour ago
    const mockProject = {
      id: 1,
      name: 'Test Project',
      status: 'running',
      base_model: 'test-model',
      training_type: 'qlora',
      progress: 50,
      current_epoch: 2,
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
});
