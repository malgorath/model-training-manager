import { Outlet, NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  Play,
  Settings,
  Cpu,
  BookOpen,
  FolderGit2,
} from 'lucide-react';
import { clsx } from 'clsx';

/**
 * Navigation item configuration.
 */
const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/datasets', label: 'Datasets', icon: Database },
  { path: '/projects', label: 'Projects', icon: FolderGit2 },
  { path: '/training', label: 'Training Jobs', icon: Play },
  { path: '/guide', label: 'User Guide', icon: BookOpen },
  { path: '/settings', label: 'Settings', icon: Settings },
];

/**
 * Main layout component with navigation sidebar.
 */
export default function Layout() {
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
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-all duration-200',
                    isActive
                      ? 'bg-primary-500/10 text-primary-400 shadow-lg shadow-primary-500/5'
                      : 'text-surface-400 hover:bg-surface-800 hover:text-surface-100'
                  )
                }
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </NavLink>
            ))}
          </nav>

          {/* Footer */}
          <div className="border-t border-surface-800 p-4">
            <div className="rounded-lg bg-surface-800/50 p-4">
              <p className="text-xs text-surface-400">
                Powered by Ollama
              </p>
              <p className="mt-1 text-xs text-surface-500">
                llama3.2:3b
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

