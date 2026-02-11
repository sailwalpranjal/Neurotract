'use client';

import { Suspense, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from '@/components/ui/Sidebar';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { Streamline } from '@/lib/types';

// Dynamic import to avoid SSR issues with Three.js
const BrainViewer = dynamic(() => import('@/components/viewer/BrainViewer'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="spinner" />
    </div>
  ),
});

export default function ViewerPage() {
  const {
    sidebarOpen,
    activeSubject,
    streamlineBundle,
    setStreamlineBundle,
    brainMesh,
    setBrainMesh,
    addNotification,
    availableResults,
    setAvailableResults,
    setActiveSubject,
  } = useAppStore();
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingData, setLoadingData] = useState(false);

  // Auto-load data when page mounts or subject changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    loadViewerData();
  }, [activeSubject]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const loadViewerData = useCallback(async () => {
    // If we already have all data for the current subject, skip
    if (streamlineBundle && brainMesh && activeSubject) return;

    // If no subject is set, try to auto-detect from available results
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
        // Server might be offline
      }
    }

    if (!subjectToLoad) return;

    setLoadingData(true);
    const notifId = addNotification({
      type: 'loading',
      title: 'Loading Viewer Data',
      message: `Fetching brain mesh and tractography for ${subjectToLoad}...`,
      duration: 0,
    });

    try {
      // Load streamlines and brain mesh in parallel
      const [streamlineData, meshData] = await Promise.all([
        streamlineBundle ? Promise.resolve(null) : apiClient.getResultStreamlines(subjectToLoad).catch(() => null),
        brainMesh ? Promise.resolve(null) : apiClient.getBrainMesh(subjectToLoad).catch(() => null),
      ]);

      if (streamlineData) {
        // Convert the JSON response to the StreamlineBundle format
        const streamlines: Streamline[] = streamlineData.streamlines.map((sl: any) => ({
          points: new Float32Array(sl.points),
          numPoints: sl.numPoints,
          length: sl.length,
          orientation: sl.orientation as [number, number, number],
        }));

        const bundle = {
          streamlines,
          bounds: streamlineData.bounds,
          metadata: streamlineData.metadata,
        };

        setStreamlineBundle(bundle);
      }

      if (meshData) {
        setBrainMesh(meshData);
      }

      useAppStore.getState().removeNotification(notifId);
      const parts: string[] = [];
      if (streamlineData) parts.push(`${streamlineData.metadata.count.toLocaleString()} streamlines`);
      if (meshData) parts.push(`brain mesh (${meshData.metadata?.n_vertices?.toLocaleString() || '?'} vertices)`);

      addNotification({
        type: 'success',
        title: 'Viewer Data Loaded',
        message: parts.length > 0 ? parts.join(' + ') : 'Using cached data',
        duration: 5000,
      });
    } catch (err: any) {
      useAppStore.getState().removeNotification(notifId);
      addNotification({
        type: 'error',
        title: 'Failed to Load Viewer Data',
        message: err.message || 'Could not fetch data from server',
        duration: 8000,
      });
    } finally {
      setLoadingData(false);
    }
  }, [activeSubject, streamlineBundle, brainMesh]);

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Viewer */}
      <div
        className={`flex-1 transition-all duration-300 ${
          sidebarOpen ? 'ml-80' : 'ml-0'
        }`}
      >
        <div className="w-full h-full relative">
          {/* Loading overlay */}
          {loadingData && (
            <div className="absolute inset-0 z-20 bg-black/50 flex items-center justify-center backdrop-blur-sm">
              <div className="text-center">
                <div className="spinner mx-auto mb-4" />
                <p className="text-gray-200 font-medium">Loading brain data...</p>
                <p className="text-gray-400 text-sm mt-2">Fetching brain mesh and streamlines</p>
              </div>
            </div>
          )}

          {loadError ? (
            <div className="w-full h-full flex items-center justify-center">
              <div className="glass rounded-lg p-8 max-w-md">
                <h3 className="text-xl font-semibold text-red-400 mb-2">
                  Error Loading Viewer
                </h3>
                <p className="text-gray-300">{loadError}</p>
                <button
                  onClick={() => {
                    setLoadError(null);
                    window.location.reload();
                  }}
                  className="mt-4 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors"
                >
                  Reload Page
                </button>
              </div>
            </div>
          ) : (
            <Suspense
              fallback={
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="spinner mx-auto mb-4" />
                    <p className="text-gray-300">Loading 3D Viewer...</p>
                  </div>
                </div>
              }
            >
              <BrainViewer onError={setLoadError} />
            </Suspense>
          )}
        </div>
      </div>
    </div>
  );
}
