'use client';

import Link from 'next/link';
import { useState, useEffect, useCallback, useRef } from 'react';
import FileUpload from '@/components/ui/FileUpload';
import UserTypeSelector from '@/components/ui/UserTypeSelector';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { ProcessedResult } from '@/lib/types';
import { formatFileSize } from '@/lib/utils';

export default function Home() {
  const [uploadComplete, setUploadComplete] = useState(false);
  const {
    availableResults,
    setAvailableResults,
    setActiveSubject,
    activeSubject,
    addNotification,
  } = useAppStore();
  const [loadingResults, setLoadingResults] = useState(true);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    checkServer();
  }, []);

  const checkServer = async () => {
    setLoadingResults(true);
    try {
      await apiClient.health();
      setServerOnline(true);
      const results = await apiClient.getAvailableResults();
      setAvailableResults(results);
    } catch {
      setServerOnline(false);
    } finally {
      setLoadingResults(false);
    }
  };

  const handleLoadSubject = useCallback((subjectId: string) => {
    setActiveSubject(subjectId);
    addNotification({
      type: 'success',
      title: 'Subject Selected',
      message: `${subjectId} is now active. Navigate to Viewer or Analysis.`,
      duration: 4000,
    });
  }, [setActiveSubject, addNotification]);

  const stats = availableResults[0];
  const streamlineCount = stats?.streamline_stats?.bundle_statistics?.n_streamlines || 0;
  const parcels = stats?.connectome_info?.n_parcels || 0;
  const edges = stats?.connectome_info?.n_edges || 0;

  return (
    <div className="flex-1 flex flex-col items-center overflow-auto">
      {/* Hero Section */}
      <section className="w-full py-12 md:py-20 px-4 text-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary-900/20 to-transparent" />
        <div className="relative max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary-400 via-cyan-400 to-primary-600">
              NeuroTract
            </span>
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-2 max-w-2xl mx-auto">
            Brain White Matter Tractography & Connectivity Analysis
          </p>
          <p className="text-sm md:text-base text-gray-500 mb-8 max-w-xl mx-auto">
            Analyze diffusion MRI data with automated tractography, connectome construction, and graph-theoretic network analysis
          </p>

          {/* Server Status */}
          <div className="flex justify-center mb-8">
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${
              serverOnline === null ? 'bg-gray-700 text-gray-300'
                : serverOnline ? 'bg-green-900/50 text-green-300 border border-green-500/30'
                : 'bg-red-900/50 text-red-300 border border-red-500/30'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                serverOnline === null ? 'bg-gray-500'
                  : serverOnline ? 'bg-green-400 animate-pulse-glow'
                  : 'bg-red-400'
              }`} />
              {serverOnline === null ? 'Checking...' : serverOnline ? 'Server Online' : 'Server Offline'}
              {!serverOnline && serverOnline !== null && (
                <button onClick={checkServer} className="ml-2 underline hover:text-red-200 text-xs">Retry</button>
              )}
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-4 justify-center">
            <Link href="/viewer" className="px-8 py-3 bg-primary-600 hover:bg-primary-700 rounded-lg transition-all font-medium text-white shadow-lg shadow-primary-600/20 hover:shadow-primary-600/40">
              Open 3D Viewer
            </Link>
            <Link href="/analysis" className="px-8 py-3 glass hover:bg-white/10 rounded-lg transition-all font-medium">
              Analysis Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      {serverOnline && streamlineCount > 0 && (
        <section className="w-full max-w-5xl mx-auto px-4 -mt-4 mb-8">
          <div className="glass rounded-xl p-6 grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
            <StatItem value={streamlineCount} label="Streamlines Tracked" />
            <StatItem value={parcels} label="Brain Regions" />
            <StatItem value={edges} label="Network Connections" />
            <StatItem value={availableResults.length} label="Subjects Processed" />
          </div>
        </section>
      )}

      <div className="max-w-5xl w-full px-4 space-y-8 pb-12">
        {/* User Type Selector */}
        <UserTypeSelector />

        {/* Available Results */}
        {availableResults.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-2">Processed Results</h2>
            <p className="text-gray-400 text-sm mb-4">Pre-computed pipeline results ready to view</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {availableResults.map((result) => (
                <ResultCard
                  key={result.subject_id}
                  result={result}
                  isActive={activeSubject === result.subject_id}
                  onLoad={() => handleLoadSubject(result.subject_id)}
                />
              ))}
            </div>
          </section>
        )}

        {/* Quick Actions */}
        {activeSubject && (
          <section className="glass rounded-xl p-6 border border-primary-500/30 bg-primary-900/10">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 bg-primary-400 rounded-full animate-pulse-glow" />
              <h3 className="text-lg font-semibold">Active: {activeSubject}</h3>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link href="/viewer" className="px-6 py-3 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors font-medium">
                Open 3D Viewer
              </Link>
              <Link href="/analysis" className="px-6 py-3 glass hover:bg-white/10 rounded-lg transition-colors font-medium">
                View Analysis
              </Link>
            </div>
          </section>
        )}

        {/* Pipeline Overview */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Processing Pipeline</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
            {['Preprocessing', 'DTI', 'CSD/FOD', 'Tractography', 'Connectome', 'Metrics', 'Visualization'].map((step, idx) => (
              <div key={step} className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full bg-primary-600/20 border border-primary-500/30 flex items-center justify-center text-sm font-bold text-primary-400 mb-2">
                  {idx + 1}
                </div>
                <span className="text-xs text-gray-400 text-center">{step}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Features Grid */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Features</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard title="3D Brain Visualization" description="Interactive brain surface with white matter tractography rendered in real-time using WebGL" />
            <FeatureCard title="Graph Network Analysis" description="Comprehensive connectome metrics including clustering, path length, efficiency, and modularity" />
            <FeatureCard title="Community Detection" description="Automated identification of functional brain modules using Louvain and Leiden algorithms" />
            <FeatureCard title="DTI Scalar Maps" description="Fractional anisotropy, mean diffusivity, axial and radial diffusivity maps" />
            <FeatureCard title="Anatomical Labels" description="89-region Desikan-Killiany parcellation with full anatomical names and descriptions" />
            <FeatureCard title="Multi-User Modes" description="Tailored views for clinicians, researchers, and students with adaptive interpretations" />
          </div>
        </section>

        {/* Upload Section */}
        <section className="glass rounded-xl p-6 md:p-8">
          <h2 className="text-2xl font-semibold mb-2">Upload New Data</h2>
          <p className="text-gray-300 text-sm mb-6">Upload diffusion MRI data to begin tractography analysis</p>
          <FileUpload onUploadComplete={() => setUploadComplete(true)} />
          {uploadComplete && (
            <div className="mt-6 p-4 bg-green-500/20 border border-green-500/50 rounded-lg">
              <p className="text-green-300 text-sm">
                Files uploaded. Use the CLI to run the pipeline, then refresh to see results.
              </p>
              <button onClick={checkServer} className="mt-3 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors text-sm">
                Refresh Results
              </button>
            </div>
          )}
        </section>

        {/* Supported Formats */}
        <div className="text-center text-sm text-gray-500 pb-4">
          <p>Supported formats: NIfTI (.nii, .nii.gz), DICOM, TRK, TCK</p>
          <p className="mt-1 text-xs text-gray-600">NeuroTract v0.1.0</p>
        </div>

        {/* Loading */}
        {loadingResults && (
          <div className="flex justify-center py-4">
            <div className="spinner" />
          </div>
        )}
      </div>
    </div>
  );
}

function StatItem({ value, label }: { value: number; label: string }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const duration = 1500;
    const steps = 30;
    const increment = value / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setDisplay(value);
        clearInterval(timer);
      } else {
        setDisplay(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <div ref={ref} className="text-center">
      <div className="text-2xl md:text-3xl font-bold text-primary-400">
        {display.toLocaleString()}
      </div>
      <div className="text-xs text-gray-400 mt-1">{label}</div>
    </div>
  );
}

function ResultCard({ result, isActive, onLoad }: { result: ProcessedResult; isActive: boolean; onLoad: () => void }) {
  const stats = result.streamline_stats?.bundle_statistics;
  const connInfo = result.connectome_info;
  const totalSize = result.files.reduce((sum, f) => sum + f.size_bytes, 0);

  return (
    <div className={`rounded-xl p-4 transition-all ${
      isActive ? 'bg-primary-600/20 border-2 border-primary-500/50' : 'bg-white/5 border-2 border-transparent hover:border-white/20'
    }`}>
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-semibold text-lg truncate">{result.subject_id}</h3>
            {isActive && <span className="text-xs bg-primary-500/30 text-primary-300 px-2 py-0.5 rounded-full flex-shrink-0">Active</span>}
          </div>
          <div className="flex flex-wrap gap-1.5 mb-2">
            {result.has_streamlines && <Badge label="Streamlines" color="blue" />}
            {result.has_metrics && <Badge label="Metrics" color="green" />}
            {result.has_connectome && <Badge label="Connectome" color="purple" />}
            {result.has_dti && <Badge label="DTI" color="yellow" />}
            {result.has_fod && <Badge label="FOD" color="pink" />}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-gray-400">
            {stats && <span>{stats.n_streamlines.toLocaleString()} streamlines</span>}
            {connInfo && <span>{connInfo.n_parcels} regions</span>}
            <span>{formatFileSize(totalSize)}</span>
          </div>
        </div>
        <button onClick={onLoad} className={`flex-shrink-0 ml-3 px-4 py-2 rounded-lg transition-colors font-medium text-sm ${
          isActive ? 'bg-primary-600 text-white' : 'bg-white/10 hover:bg-white/20 text-gray-200'
        }`}>
          {isActive ? 'Selected' : 'Load'}
        </button>
      </div>
    </div>
  );
}

function Badge({ label, color }: { label: string; color: string }) {
  const colors: Record<string, string> = {
    blue: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    green: 'bg-green-500/20 text-green-300 border-green-500/30',
    purple: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    yellow: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    pink: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  };
  return <span className={`text-xs px-2 py-0.5 rounded-full border ${colors[color] || colors.blue}`}>{label}</span>;
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="glass rounded-xl p-5 hover:bg-white/10 transition-all hover:scale-[1.02]">
      <h3 className="text-base font-semibold mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}
