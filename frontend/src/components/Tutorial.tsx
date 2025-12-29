import { useState, useEffect } from 'react';
import { X, ChevronRight, ChevronLeft, HelpCircle, BookOpen } from 'lucide-react';
import { clsx } from 'clsx';
import React from 'react';

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for element to highlight
}

interface TutorialProps {
  steps: TutorialStep[];
  onComplete?: () => void;
  onClose?: () => void;
}

/**
 * Interactive tutorial overlay component with step-by-step guidance.
 */
export default function Tutorial({ steps, onComplete, onClose }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  const current = steps[currentStep];
  const isFirst = currentStep === 0;
  const isLast = currentStep === steps.length - 1;

  const nextStep = () => {
    if (isLast) {
      onComplete?.();
      setIsVisible(false);
    } else {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (!isFirst) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleClose = () => {
    setIsVisible(false);
    onClose?.();
  };

  useEffect(() => {
    if (current?.target) {
      const element = document.querySelector(current.target);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [current]);

  if (!isVisible || !current) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={handleClose} />

      {/* Highlight overlay for target element */}
      {current.target && (
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute border-4 border-primary-500 rounded-lg shadow-2xl shadow-primary-500/50"
            style={{
              // Calculate position from target element
              // This is simplified - in production, calculate actual position
            }}
          />
        </div>
      )}

      {/* Tutorial Card */}
      <div className="relative bg-surface-900 border border-surface-700 rounded-xl shadow-2xl max-w-md w-full mx-4">
        <div className="flex items-center justify-between p-4 border-b border-surface-800">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Tutorial</h3>
          </div>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-surface-800 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-surface-400" />
          </button>
        </div>

        <div className="p-6">
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-lg font-semibold text-white">{current.title}</h4>
              <span className="text-sm text-surface-400">
                {currentStep + 1} / {steps.length}
              </span>
            </div>
            <p className="text-surface-300 leading-relaxed">{current.content}</p>
          </div>

          {/* Progress bar */}
          <div className="mb-6">
            <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all"
                style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
              />
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={prevStep}
              disabled={isFirst}
              className={clsx(
                'btn-secondary flex items-center gap-2',
                isFirst && 'opacity-50 cursor-not-allowed'
              )}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </button>
            <button onClick={nextStep} className="btn-primary flex items-center gap-2">
              {isLast ? 'Complete' : 'Next'}
              {!isLast && <ChevronRight className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Help tooltip component for inline help text.
 */
export function HelpTooltip({ content, children }: { content: string; children: React.ReactNode }) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className="inline-flex items-center"
      >
        {children}
        <HelpCircle className="ml-1 h-4 w-4 text-surface-500 cursor-help" />
      </div>
      {isVisible && (
        <div className="absolute z-50 bottom-full left-0 mb-2 p-3 bg-surface-800 border border-surface-700 rounded-lg shadow-xl max-w-xs">
          <p className="text-sm text-surface-200">{content}</p>
          <div className="absolute top-full left-4 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-surface-800" />
        </div>
      )}
    </div>
  );
}
