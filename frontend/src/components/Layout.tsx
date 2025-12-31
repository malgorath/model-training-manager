import { Outlet, NavLink } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  LayoutDashboard,
  Database,
  Play,
  Settings,
  Cpu,
  BookOpen,
  FolderGit2,
  Package,
} from 'lucide-react';
import { clsx } from 'clsx';
import { datasetApi, projectApi, modelsApi, trainingJobApi } from '../services/api';

/**
 * Navigation item configuration.
 */
const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, countKey: null },
  { path: '/datasets', label: 'Datasets', icon: Database, countKey: 'datasets' },
  { path: '/projects', label: 'Projects', icon: FolderGit2, countKey: 'projects' },
  { path: '/models', label: 'Models', icon: Package, countKey: 'models' },
  { path: '/training', label: 'Training Jobs', icon: Play, countKey: 'trainingJobs' },
  { path: '/guide', label: 'User Guide', icon: BookOpen, countKey: null },
  { path: '/settings', label: 'Settings', icon: Settings, countKey: null },
];

/**
 * Main layout component with navigation sidebar.
 */
export default function Layout() {
  // Fetch counts for menu items
  const { data: datasetsData } = useQuery({
    queryKey: ['datasets', 1],
    queryFn: () => datasetApi.list(1, 1), // Just get total count
    select: (data) => data.total,
  });

  const { data: projectsData } = useQuery({
    queryKey: ['projects', 1],
    queryFn: () => projectApi.list(1, 1), // Just get total count
    select: (data) => data.total,
  });

  const { data: modelsData } = useQuery({
    queryKey: ['models', 'local'],
    queryFn: () => modelsApi.listLocal(),
    select: (data) => Array.isArray(data) ? data.length : 0,
  });

  const { data: trainingJobsData } = useQuery({
    queryKey: ['jobs', 1],
    queryFn: () => trainingJobApi.list(1, 1), // Just get total count
    select: (data) => data.total,
  });

  const counts = {
    datasets: datasetsData ?? 0,
    projects: projectsData ?? 0,
    models: modelsData ?? 0,
    trainingJobs: trainingJobsData ?? 0,
  };

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-surface-800 bg-surface-900/80 backdrop-blur-xl">
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center gap-3 border-b border-surface-800 px-6">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-accent-500">
              <Cpu className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">Trainers</h1>
              <p className="text-xs text-surface-400">Model Training Manager</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 p-4">
            {navItems.map((item) => {
              const count = item.countKey ? counts[item.countKey as keyof typeof counts] : null;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center justify-between gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-all duration-200',
                      isActive
                        ? 'bg-primary-500/10 text-primary-400 shadow-lg shadow-primary-500/5'
                        : 'text-surface-400 hover:bg-surface-800 hover:text-surface-100'
                    )
                  }
                >
                  <div className="flex items-center gap-3">
                    <item.icon className="h-5 w-5" />
                    {item.label}
                  </div>
                  {count !== null && (
                    <span className="rounded-full bg-surface-800 px-2 py-0.5 text-xs font-semibold text-surface-300">
                      {count}
                    </span>
                  )}
                </NavLink>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t border-surface-800 p-4">
            <div className="rounded-lg bg-surface-800/50 p-4">
              <p className="text-xs text-surface-400">
                Powered by{' '}
                <a 
                  href="http://scott-sanders.dev" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary-400 hover:text-primary-300 underline"
                >
                  Scott Sanders
                </a>
              </p>
              <p className="mt-1 text-xs text-surface-500">
                <a 
                  href="/LICENSE" 
                  target="_blank"
                  className="text-surface-400 hover:text-surface-300 underline"
                >
                  MIT License
                </a>
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-64 flex-1 p-8">
        <Outlet />
      </main>
    </div>
  );
}

