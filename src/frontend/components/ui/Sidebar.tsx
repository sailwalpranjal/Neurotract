'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/lib/store';
import Controls from '@/components/viewer/Controls';
import JobStatus from './JobStatus';

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar, currentJob, activeSubject, streamlineBundle, brainMesh } = useAppStore();

  // Close sidebar on escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && sidebarOpen) {
        toggleSidebar();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sidebarOpen, toggleSidebar]);

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay md:hidden"
          onClick={toggleSidebar}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-16 bottom-0 w-80 glass border-r border-white/10 overflow-y-auto z-40 sidebar-slide ${
          sidebarOpen ? '' : 'sidebar-hidden md:hidden'
        }`}
        role="complementary"
        aria-label="Sidebar"
      >
        <div className="p-4 md:p-6 space-y-6">
          {/* Close button on mobile */}
          <button
            className="md:hidden absolute top-3 right-3 p-2 hover:bg-white/10 rounded-lg transition-colors"
            onClick={toggleSidebar}
            aria-label="Close sidebar"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Active Subject */}
          {activeSubject && (
            <div className="bg-primary-600/20 border border-primary-500/30 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-primary-300 mb-1">Active Subject</h3>
              <p className="font-medium">{activeSubject}</p>
              {streamlineBundle && (
                <p className="text-xs text-gray-400 mt-1">
                  {streamlineBundle.metadata.count.toLocaleString()} streamlines loaded
                </p>
              )}
              {brainMesh && (
                <p className="text-xs text-gray-400">
                  Brain mesh: {brainMesh.metadata?.n_vertices?.toLocaleString() || '?'} vertices
                </p>
              )}
            </div>
          )}

          {/* Job Status */}
          {currentJob && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Current Job</h3>
              <JobStatus job={currentJob} compact />
            </div>
          )}

          {/* Viewer Controls */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Viewer Controls</h3>
            <Controls />
          </div>

          {/* Keyboard Shortcuts */}
          <div className="pt-6 border-t border-white/10">
            <h3 className="text-sm font-semibold mb-2 text-gray-400">
              Keyboard Shortcuts
            </h3>
            <div className="space-y-2 text-sm text-gray-300">
              <ShortcutItem keys={['Left Click + Drag']} action="Rotate" />
              <ShortcutItem keys={['Right Click + Drag']} action="Pan" />
              <ShortcutItem keys={['Scroll']} action="Zoom" />
              <ShortcutItem keys={['R']} action="Reset Camera" />
              <ShortcutItem keys={['B']} action="Toggle Brain" />
              <ShortcutItem keys={['T']} action="Toggle Tracts" />
              <ShortcutItem keys={['L']} action="Toggle Labels" />
              <ShortcutItem keys={['S']} action="Toggle Slices" />
              <ShortcutItem keys={['A']} action="Auto-Rotate" />
              <ShortcutItem keys={['W']} action="Wireframe" />
              <ShortcutItem keys={['1', '2', '3']} action="Brain Models" />
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

function ShortcutItem({ keys, action }: { keys: string[]; action: string }) {
  return (
    <div className="flex justify-between items-center">
      <div className="flex gap-1">
        {keys.map((key, i) => (
          <kbd
            key={i}
            className="px-2 py-1 bg-white/10 rounded text-xs font-mono"
          >
            {key}
          </kbd>
        ))}
      </div>
      <span className="text-gray-400">{action}</span>
    </div>
  );
}
