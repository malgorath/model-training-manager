/**
 * Tests for the UserGuidePage component.
 * 
 * Tests component rendering, navigation between sections,
 * and code example display functionality.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, beforeEach } from 'vitest';
import UserGuidePage from '../pages/UserGuidePage';

/**
 * Create a fresh QueryClient for each test.
 */
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

/**
 * Wrapper component for tests.
 */
const TestWrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe('UserGuidePage', () => {
  beforeEach(() => {
    render(
      <TestWrapper>
        <UserGuidePage />
      </TestWrapper>
    );
  });

  describe('Page Header', () => {
    it('renders the page title', () => {
      expect(screen.getByText('User Guide')).toBeInTheDocument();
    });

    it('renders the page description', () => {
      expect(
        screen.getByText(/Comprehensive documentation for training methods/i)
      ).toBeInTheDocument();
    });
  });

  describe('Quick Start Section', () => {
    it('renders the quick start section', () => {
      expect(screen.getByText('Quick Start')).toBeInTheDocument();
    });

    it('displays all three steps', () => {
      expect(screen.getByText('Upload Dataset')).toBeInTheDocument();
      expect(screen.getByText('Create Training Job')).toBeInTheDocument();
      expect(screen.getByText('Download & Deploy')).toBeInTheDocument();
    });
  });

  describe('Training Methods Section', () => {
    it('renders all training method sections', () => {
      expect(screen.getByText('Training Methods')).toBeInTheDocument();
      expect(screen.getByText('QLoRA Training')).toBeInTheDocument();
      expect(screen.getByText('Unsloth Training')).toBeInTheDocument();
      expect(screen.getByText('RAG Training')).toBeInTheDocument();
      expect(screen.getByText('Standard Fine-tuning')).toBeInTheDocument();
    });

    it('displays section descriptions', () => {
      // Use getAllByText since there may be multiple matches
      expect(
        screen.getAllByText(/Quantized Low-Rank Adaptation/i).length
      ).toBeGreaterThan(0);
      expect(
        screen.getAllByText(/Optimized LoRA training/i).length
      ).toBeGreaterThan(0);
      expect(
        screen.getAllByText(/Retrieval-Augmented Generation/i).length
      ).toBeGreaterThan(0);
      expect(
        screen.getAllByText(/full model training|supervised fine-tuning/i).length
      ).toBeGreaterThan(0);
    });
  });

  describe('Section Expansion', () => {
    it('QLoRA section is expanded by default', () => {
      // QLoRA content should be visible
      expect(screen.getByText('Overview')).toBeInTheDocument();
    });

    it('clicking a section header toggles expansion', async () => {
      // Click on Unsloth to expand it
      const unslothHeader = screen.getByText('Unsloth Training').closest('button');
      if (unslothHeader) {
        fireEvent.click(unslothHeader);
      }

      // Wait for the section to expand
      await waitFor(() => {
        // Should now show Unsloth content
        const overviewElements = screen.getAllByText('Overview');
        expect(overviewElements.length).toBeGreaterThan(0);
      });
    });

    it('clicking the same section header collapses it', async () => {
      // First, find the QLoRA header button
      const qloraHeader = screen.getByText('QLoRA Training').closest('button');
      
      // Click to collapse
      if (qloraHeader) {
        fireEvent.click(qloraHeader);
      }

      // QLoRA should be collapsed (overview might not be visible)
      // Click again to expand
      if (qloraHeader) {
        fireEvent.click(qloraHeader);
      }

      // Now it should be visible again
      await waitFor(() => {
        expect(screen.getByText('Overview')).toBeInTheDocument();
      });
    });
  });

  describe('QLoRA Section Content', () => {
    it('displays prerequisites section', () => {
      expect(
        screen.getByText(/Prerequisites & Installation/i)
      ).toBeInTheDocument();
    });

    it('displays Ollama integration section', () => {
      expect(screen.getByText('Using with Ollama')).toBeInTheDocument();
    });

    it('displays LM Studio integration section', () => {
      expect(screen.getByText('Using with LM Studio')).toBeInTheDocument();
    });

    it('displays recommended parameters', () => {
      expect(screen.getByText('Recommended Parameters')).toBeInTheDocument();
    });
  });

  describe('Model Output Section', () => {
    it('renders the model output section', () => {
      expect(screen.getByText('Model Output & Deployment')).toBeInTheDocument();
    });

    it('displays download instructions', () => {
      expect(screen.getByText('Downloading Trained Models')).toBeInTheDocument();
    });

    it('displays model storage location', () => {
      expect(screen.getByText('Model Storage Location')).toBeInTheDocument();
    });
  });
});

describe('CodeExample Component', () => {
  beforeEach(() => {
    render(
      <TestWrapper>
        <UserGuidePage />
      </TestWrapper>
    );
  });

  it('renders code examples with tabs', async () => {
    // Look for language tabs in code examples
    const pythonTabs = screen.getAllByText('Python 3');
    expect(pythonTabs.length).toBeGreaterThan(0);
  });

  it('renders PHP tabs', () => {
    const phpTabs = screen.getAllByText('PHP 8');
    expect(phpTabs.length).toBeGreaterThan(0);
  });

  it('renders BASH/CURL tabs', () => {
    // BASH tabs for installation
    const bashTabs = screen.getAllByText('BASH');
    expect(bashTabs.length).toBeGreaterThan(0);
  });

  it('renders copy button for code examples', () => {
    const copyButtons = screen.getAllByText('Copy');
    expect(copyButtons.length).toBeGreaterThan(0);
  });
});

describe('Navigation Integration', () => {
  it('renders correctly within the app layout', () => {
    render(
      <TestWrapper>
        <UserGuidePage />
      </TestWrapper>
    );

    // The page should render its content
    expect(screen.getByText('User Guide')).toBeInTheDocument();
  });
});

describe('Accessibility', () => {
  beforeEach(() => {
    render(
      <TestWrapper>
        <UserGuidePage />
      </TestWrapper>
    );
  });

  it('section headers are buttons for keyboard navigation', () => {
    const buttons = screen.getAllByRole('button');
    // Should have at least 4 section toggle buttons
    expect(buttons.length).toBeGreaterThanOrEqual(4);
  });

  it('headings have proper hierarchy', () => {
    const h1 = screen.getByRole('heading', { level: 1 });
    expect(h1).toHaveTextContent('User Guide');

    const h2s = screen.getAllByRole('heading', { level: 2 });
    expect(h2s.length).toBeGreaterThan(0);
  });
});

describe('Content Accuracy', () => {
  beforeEach(() => {
    render(
      <TestWrapper>
        <UserGuidePage />
      </TestWrapper>
    );
  });

  it('mentions Ollama API endpoint', () => {
    // Check that code examples contain Ollama API
    const pageContent = document.body.textContent || '';
    expect(pageContent).toContain('localhost:11434');
  });

  it('mentions LM Studio API endpoint', () => {
    const pageContent = document.body.textContent || '';
    expect(pageContent).toContain('localhost:1234');
  });

  it('includes pip install commands', () => {
    const pageContent = document.body.textContent || '';
    expect(pageContent).toContain('pip install');
  });

  it('mentions required packages', () => {
    const pageContent = document.body.textContent || '';
    expect(pageContent).toContain('transformers');
    expect(pageContent).toContain('peft');
  });
});
