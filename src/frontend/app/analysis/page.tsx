'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { GraphMetrics, ParcellationLabel } from '@/lib/types';
import { generateSummary } from '@/lib/interpretations';
import MetricCard from '@/components/analysis/MetricCard';
import UserTypeSelector from '@/components/ui/UserTypeSelector';
import BrainHealthSummary from '@/components/analysis/BrainHealthSummary';

const GraphMetricsChart = dynamic(() => import('@/components/charts/GraphMetrics'), { ssr: false });
const ConnectomeMatrix = dynamic(() => import('@/components/charts/ConnectomeMatrix'), { ssr: false });
const HubRegions = dynamic(() => import('@/components/charts/HubRegions'), { ssr: false });
const CommunityView = dynamic(() => import('@/components/charts/CommunityView'), { ssr: false });
const AdvancedMetrics = dynamic(() => import('@/components/analysis/AdvancedMetrics'), { ssr: false });
const EducationalPanel = dynamic(() => import('@/components/analysis/EducationalPanel'), { ssr: false });

export default function AnalysisPage() {
  const {
    activeSubject,
    metrics: storeMetrics,
    connectome: storeConnectome,
    setMetrics: setStoreMetrics,
    setConnectome: setStoreConnectome,
    parcellationLabels,
    setParcellationLabels,
    userType,
    addNotification,
    availableResults,
    setAvailableResults,
    setActiveSubject,
  } = useAppStore();

  const [metrics, setMetrics] = useState<GraphMetrics | null>(storeMetrics);
  const [connectome, setConnectome] = useState<number[][] | null>(storeConnectome);
  const [labels, setLabels] = useState<ParcellationLabel[]>(parcellationLabels);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    loadAnalysisData();
  }, [activeSubject]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const loadAnalysisData = useCallback(async () => {
    if (storeMetrics && storeConnectome && activeSubject) {
      setMetrics(storeMetrics);
      setConnectome(storeConnectome);
      if (parcellationLabels.length > 0) setLabels(parcellationLabels);
      return;
    }

    let subjectToLoad = activeSubject;
    if (!subjectToLoad) {
      try {
        let results = availableResults;
        if (results.length === 0) {
          results = await apiClient.getAvailableResults();
          setAvailableResults(results);
        }
        if (results.length > 0) {
          subjectToLoad = results[0].subject_id;
          setActiveSubject(subjectToLoad);
        }
      } catch {
        return;
      }
    }

    if (!subjectToLoad) return;

    setLoading(true);
    setError(null);

    const notifId = addNotification({
      type: 'loading',
      title: 'Loading Analysis Data',
      message: `Fetching metrics, connectome, and labels for ${subjectToLoad}...`,
      duration: 0,
    });

    try {
      const [metricsData, connectomeData, labelsData] = await Promise.all([
        apiClient.getResultMetrics(subjectToLoad).catch(() => null),
        apiClient.getResultConnectome(subjectToLoad).catch(() => null),
        apiClient.getParcellationLabels(subjectToLoad).catch(() => null),
      ]);

      setMetrics(metricsData);
      setConnectome(connectomeData);
      setStoreMetrics(metricsData);
      setStoreConnectome(connectomeData);

      if (labelsData?.labels) {
        setLabels(labelsData.labels);
        setParcellationLabels(labelsData.labels);
      }

      useAppStore.getState().removeNotification(notifId);
      addNotification({
        type: 'success',
        title: 'Analysis Data Loaded',
        message: `Data loaded for ${subjectToLoad}`,
        duration: 4000,
      });
    } catch (err: any) {
      useAppStore.getState().removeNotification(notifId);
      setError(err.message || 'Failed to load analysis data');
      addNotification({
        type: 'error',
        title: 'Failed to Load Analysis',
        message: err.message || 'Could not fetch analysis data',
        duration: 8000,
      });
    } finally {
      setLoading(false);
    }
  }, [activeSubject, storeMetrics, storeConnectome]);

  if (!activeSubject && !loading && !metrics) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 md:p-8">
        <div className="glass rounded-lg p-8 max-w-md text-center">
          <h2 className="text-2xl font-semibold mb-4">No Data Available</h2>
          <p className="text-gray-300 mb-6">
            Select a processed subject from the home page, or run the pipeline first
          </p>
          <Link href="/" className="inline-block px-6 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors">
            Go to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-7xl mx-auto p-4 md:p-6 lg:p-8 space-y-6 md:space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold mb-1">
              {userType === 'doctor' ? 'Clinical Analysis' :
               userType === 'student' ? 'Research Analysis' :
               'Brain Connectivity Report'}
            </h1>
            <p className="text-gray-300 text-sm md:text-base">
              {userType === 'doctor'
                ? 'Comprehensive structural connectome analysis with normative comparison'
                : userType === 'student'
                ? 'Interactive network analysis with educational resources'
                : 'Understanding your brain connectivity results'}
              {activeSubject && <span className="ml-2 text-primary-400">- {activeSubject}</span>}
            </p>
          </div>
          <UserTypeSelector compact />
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex justify-center py-12">
            <div className="text-center">
              <div className="spinner mx-auto mb-4" />
              <p className="text-gray-300">Loading analysis data...</p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass rounded-lg p-6 border border-red-500/50 bg-red-500/10">
            <h3 className="text-xl font-semibold text-red-400 mb-2">Error</h3>
            <p className="text-gray-300">{error}</p>
            <button onClick={loadAnalysisData} className="mt-4 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors text-sm">
              Retry
            </button>
          </div>
        )}

        {metrics && (
          <>
            {/* ==========================================
                GENERAL USER: Simplified Brain Health View
                ========================================== */}
            {userType === 'general' && (
              <BrainHealthSummary metrics={metrics} />
            )}

            {/* ==========================================
                ALL USERS: Summary (adapted text per type)
                ========================================== */}
            <div className="glass rounded-xl p-6 border border-primary-500/20">
              <h2 className="text-lg font-semibold mb-3">
                {userType === 'doctor' ? 'Clinical Summary' : userType === 'student' ? 'Network Overview' : 'Summary'}
              </h2>
              <p className="text-gray-300 text-sm leading-relaxed">
                {generateSummary(metrics, userType)}
              </p>
            </div>

            {/* ==========================================
                ALL USERS: Global Metrics (cards with status)
                ========================================== */}
            <div className="glass rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">
                {userType === 'general' ? 'Key Measurements' : 'Global Network Metrics'}
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                <MetricCard label="Clustering Coefficient" metricKey="clustering_coefficient" value={metrics.global.clustering_coefficient} />
                <MetricCard label="Path Length" metricKey="characteristic_path_length" value={metrics.global.characteristic_path_length} />
                <MetricCard label="Global Efficiency" metricKey="global_efficiency" value={metrics.global.global_efficiency} />
                <MetricCard label="Modularity" metricKey="modularity" value={metrics.global.modularity} />
                {/* Show more metrics for researchers and students */}
                {userType !== 'general' && (
                  <>
                    <MetricCard label="Assortativity" metricKey="assortativity" value={metrics.global.assortativity} />
                    <MetricCard label="Small-Worldness" metricKey="small_worldness" value={metrics.global.small_worldness} />
                    {metrics.global.density !== undefined && (
                      <MetricCard label="Network Density" metricKey="density" value={metrics.global.density} />
                    )}
                    {metrics.global.transitivity !== undefined && (
                      <MetricCard label="Transitivity" metricKey="transitivity" value={metrics.global.transitivity} />
                    )}
                  </>
                )}
              </div>
            </div>

            {/* ==========================================
                RESEARCHER (doctor): Advanced Analysis Tools
                ========================================== */}
            {userType === 'doctor' && labels.length > 0 && (
              <AdvancedMetrics metrics={metrics} labels={labels} connectome={connectome} />
            )}

            {/* ==========================================
                STUDENT: Educational Panel + Correlations
                ========================================== */}
            {userType === 'student' && labels.length > 0 && (
              <div className="glass rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-2">Learning Center</h2>
                <p className="text-gray-400 text-sm mb-4">
                  Interactive educational resources, metric correlations, and network comparisons
                </p>
                <EducationalPanel metrics={metrics} labels={labels} />
              </div>
            )}

            {/* ==========================================
                RESEARCHER + STUDENT: Hub Regions
                ========================================== */}
            {userType !== 'general' && labels.length > 0 && (
              <div className="glass rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-2">Hub Regions</h2>
                <p className="text-gray-400 text-sm mb-4">
                  {userType === 'doctor'
                    ? 'Top network hubs ranked by degree and betweenness centrality'
                    : 'The most connected and influential regions in the brain network'}
                </p>
                <HubRegions
                  degree={metrics.nodal.degree}
                  betweenness={metrics.nodal.betweenness_centrality}
                  labels={labels}
                  topN={userType === 'doctor' ? 15 : 10}
                />
              </div>
            )}

            {/* ==========================================
                RESEARCHER + STUDENT: Nodal Metrics Charts
                ========================================== */}
            {userType !== 'general' && (
              <div className="glass rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-2">Nodal Metrics</h2>
                <p className="text-gray-400 text-sm mb-4">
                  {metrics.nodal.degree.length} brain regions analyzed
                </p>
                <GraphMetricsChart metrics={metrics} labels={labels.length > 0 ? labels : undefined} />
              </div>
            )}

            {/* ==========================================
                ALL USERS: Community Structure (adapted)
                ========================================== */}
            {metrics.communities?.louvain_partition && labels.length > 0 && (
              <div className="glass rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-2">
                  {userType === 'general' ? 'Brain Region Groups' : 'Community Structure'}
                </h2>
                <p className="text-gray-400 text-sm mb-4">
                  {userType === 'doctor'
                    ? 'Louvain community detection partitioning with member regions and node strength'
                    : userType === 'student'
                    ? 'Brain regions grouped into communities that are more densely connected internally'
                    : 'Your brain regions organized into groups that work closely together'}
                </p>
                <CommunityView
                  partition={metrics.communities.louvain_partition}
                  labels={labels}
                  modularity={metrics.communities.louvain_modularity}
                  nodeStrength={metrics.nodal.node_strength}
                />
              </div>
            )}
          </>
        )}

        {/* ==========================================
            RESEARCHER + STUDENT: Connectome Matrix
            ========================================== */}
        {connectome && userType !== 'general' && (
          <div className="glass rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-2">Connectome Matrix</h2>
            <p className="text-gray-400 text-sm mb-4">
              {connectome.length} x {connectome[0]?.length || 0} structural connectivity matrix
              {userType === 'student' && ' — each cell shows the connection strength between two brain regions'}
            </p>
            <ConnectomeMatrix matrix={connectome} labels={labels.length > 0 ? labels : undefined} />
          </div>
        )}

        {/* ==========================================
            ALL USERS: Export (more options for researcher)
            ========================================== */}
        {(metrics || connectome) && activeSubject && (
          <div className="glass rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Export Results</h2>
            <div className="flex flex-wrap gap-3">
              <ExportButton label="Metrics (JSON)" filename="metrics.json" subject={activeSubject} addNotification={addNotification} setError={setError} />
              {userType !== 'general' && (
                <>
                  <ExportButton label="Connectome (CSV)" filename="connectome.csv" subject={activeSubject} addNotification={addNotification} setError={setError} />
                  <ExportButton label="Streamlines (TRK)" filename="streamlines.trk" subject={activeSubject} addNotification={addNotification} setError={setError} />
                </>
              )}
            </div>
            {userType === 'general' && (
              <p className="text-xs text-gray-500 mt-3">
                Switch to Researcher mode for additional export options (connectome matrix, streamline files).
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ExportButton({ label, filename, subject, addNotification, setError }: {
  label: string;
  filename: string;
  subject: string;
  addNotification: any;
  setError: (e: string | null) => void;
}) {
  const handleDownload = async () => {
    try {
      addNotification({ type: 'loading', title: 'Downloading', message: `Preparing ${filename}...`, duration: 3000 });
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/results/${subject}/download/${filename}`
      );
      if (!response.ok) throw new Error('Download failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${subject}_${filename}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      addNotification({ type: 'success', title: 'Download Complete', message: filename, duration: 3000 });
    } catch (err: any) {
      setError(err.message || 'Download failed');
    }
  };

  return (
    <button onClick={handleDownload} className="px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors text-sm">
      {label}
    </button>
  );
}
