'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAppStore } from '@/lib/store';

const USER_TYPE_LABELS = {
  doctor: 'Clinician',
  student: 'Researcher',
  general: 'Student',
};

export default function Header() {
  const pathname = usePathname();
  const { toggleSidebar, sidebarOpen, userType, activeSubject } = useAppStore();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const isActive = (path: string) => pathname === path;
  const showSidebarToggle = pathname === '/viewer' || pathname === '/analysis';

  return (
    <header className="glass border-b border-white/10 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left: Logo + Mobile menu */}
          <div className="flex items-center gap-2">
            {/* Mobile sidebar toggle */}
            {showSidebarToggle && (
              <button
                onClick={toggleSidebar}
                className="md:hidden p-2 hover:bg-white/10 rounded-lg transition-colors"
                aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}

            <Link href="/" className="flex items-center space-x-2.5">
              <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 via-primary-500 to-primary-700 rounded-lg flex items-center justify-center font-bold text-white shadow-md shadow-cyan-500/20">
                N
              </div>
              <div className="flex items-baseline gap-1.5">
                <span className="text-xl font-bold tracking-tight text-white hidden sm:inline">NeuroTract</span>
                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/80 px-1.5 py-0.5 rounded border border-cyan-800/60 font-semibold">2.0</span>
              </div>
            </Link>
          </div>

          {/* Center: Navigation */}
          <nav className="hidden md:flex space-x-1.5 bg-black/20 p-1 rounded-xl border border-white/[0.04]" role="navigation">
            <NavLink href="/" active={isActive('/')}>
              Workstation
            </NavLink>
            <NavLink href="/viewer" active={isActive('/viewer')}>
              3D Viewer
            </NavLink>
            <NavLink href="/analysis" active={isActive('/analysis')}>
              Connectomics
            </NavLink>
          </nav>

          {/* Right: User type + sidebar toggle */}
          <div className="flex items-center gap-2">
            {/* Active subject indicator */}
            {mounted && activeSubject && (
              <span className="hidden lg:inline-flex items-center gap-1.5 text-xs text-cyan-300 bg-cyan-950/70 border border-cyan-800/60 px-2.5 py-1 rounded-full font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                {activeSubject}
              </span>
            )}

            {/* User type badge - only render after mount to avoid hydration mismatch */}
            {mounted && (
              <span className="hidden sm:inline-flex text-xs text-slate-300 bg-slate-800/80 border border-slate-700/60 px-2.5 py-1 rounded-full font-mono">
                {USER_TYPE_LABELS[userType]}
              </span>
            )}

            {/* Desktop sidebar toggle */}
            {showSidebarToggle && (
              <button
                onClick={toggleSidebar}
                className="hidden md:flex p-2 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors border border-white/[0.05]"
                aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
                aria-expanded={sidebarOpen}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {sidebarOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <nav className="md:hidden border-t border-white/10 px-4 py-2" role="navigation">
        <div className="flex space-x-1">
          <NavLink href="/" active={isActive('/')} mobile>
            Workstation
          </NavLink>
          <NavLink href="/viewer" active={isActive('/viewer')} mobile>
            Viewer
          </NavLink>
          <NavLink href="/analysis" active={isActive('/analysis')} mobile>
            Analysis
          </NavLink>
        </div>
      </nav>
    </header>
  );
}

function NavLink({
  href,
  active,
  mobile,
  children,
}: {
  href: string;
  active: boolean;
  mobile?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={`
        ${mobile ? 'flex-1 text-center text-sm' : ''}
        px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all
        ${
          active
            ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10 font-semibold'
            : 'text-slate-400 hover:bg-white/[0.05] hover:text-slate-200'
        }
      `}
    >
      {children}
    </Link>
  );
}
