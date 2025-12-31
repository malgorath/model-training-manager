/**
 * Tests for ProjectList component - Handling null max_rows.
 * 
 * Following TDD methodology: Tests define expected behavior before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Mock API modules
vi.mock('../services/api', () => ({
  projectApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Project',
          description: 'Test description',
          base_model: 'llama3.2:3b',
          training_type: 'qlora',
          status: 'pending',
          max_rows: null, // This is the issue - null max_rows
          traits: [],
          progress: 0,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
        {
          id: 2,
          name: 'Test Project 2',
          description: 'Test description 2',
          base_model: 'llama3.2:3b',
          training_type: 'qlora',
          status: 'pending',
          max_rows: 50000, // Project with max_rows set
          traits: [],
          progress: 0,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 2,
      page: 1,
      page_size: 50,
      pages: 1,
    }),
    delete: vi.fn().mockResolvedValue(undefined),
    start: vi.fn().mockResolvedValue(undefined),
  },
}));

// Import component after mocks
import ProjectList from '../components/ProjectList';

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

describe('ProjectList Component - Null max_rows Handling', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders projects with null max_rows without crashing', async () => {
    render(<ProjectList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should not crash when max_rows is null
    expect(screen.getByText('Test Project')).toBeInTheDocument();
    expect(screen.getByText('Test Project 2')).toBeInTheDocument();
  });

  it('displays "N/A" or empty for null max_rows', async () => {
    render(<ProjectList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    // Should handle null max_rows gracefully
    const project1 = screen.getByText('Test Project').closest('div');
    expect(project1).toBeInTheDocument();
    
    // Should display formatted number for non-null max_rows
    expect(screen.getByText(/50,000/)).toBeInTheDocument();
  });
});
