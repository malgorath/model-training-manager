import { Loader2 } from 'lucide-react';

interface LoadingModalProps {
  isOpen: boolean;
  message?: string;
  subtitle?: string;
}

/**
 * Full-screen loading overlay modal for long-running operations.
 */
export default function LoadingModal({ isOpen, message = 'Loading...', subtitle }: LoadingModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-surface-900 border border-surface-700 rounded-xl p-8 shadow-2xl max-w-md w-full mx-4">
        <div className="flex flex-col items-center text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary-400" />
          <div>
            <h3 className="text-lg font-semibold text-white mb-1">{message}</h3>
            {subtitle && <p className="text-sm text-surface-400">{subtitle}</p>}
          </div>
          <div className="w-full bg-surface-800 rounded-full h-2 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-primary-500 to-accent-500 animate-pulse" style={{ width: '45%' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
