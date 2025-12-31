import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for E2E UI tests.
 * Tests interact with the actual UI, not APIs.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false, // Run tests sequentially to avoid conflicts
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // Run one test at a time
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3001',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    command: 'echo "Frontend should be running on http://localhost:3001"',
    url: 'http://localhost:3001',
    reuseExistingServer: true, // Don't start server, assume it's running
    timeout: 120 * 1000,
  },
});
