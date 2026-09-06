'use client';

import { Suspense, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from '@/components/ui/Sidebar';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { Streamline } from '@/lib/types';
import OrthogonalSliceViewer from '@/components/viewer/OrthogonalSliceViewer';

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
  const [viewportLayout, setViewportLayout] = useState<'split' | '3d' | '2d'>('split');

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
      // Load streamlines, brain mesh, connectome, and labels in parallel
      const [streamlineData, meshData, connectomeData, labelsData] = await Promise.all([
        streamlineBundle ? Promise.resolve(null) : apiClient.getResultStreamlines(subjectToLoad).catch(() => null),
        brainMesh ? Promise.resolve(null) : apiClient.getBrainMesh(subjectToLoad).catch(() => null),
        useAppStore.getState().connectome ? Promise.resolve(null) : apiClient.getResultConnectome(subjectToLoad).catch(() => null),
        useAppStore.getState().parcellationLabels.length > 0 ? Promise.resolve(null) : apiClient.getParcellationLabels(subjectToLoad).catch(() => null),
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

      if (connectomeData) {
        useAppStore.getState().setConnectome(connectomeData);
      }

      if (labelsData?.labels) {
        useAppStore.getState().setParcellationLabels(labelsData.labels);
      }

      useAppStore.getState().removeNotification(notifId);
      const parts: string[] = [];
      if (streamlineData) parts.push(`${streamlineData.metadata.count.toLocaleString()} streamlines`);
      if (meshData) parts.push(`brain surface (${meshData.metadata?.n_vertices?.toLocaleString() || '?'} vertices)`);
      if (connectomeData) parts.push(`3D connectome`);

      addNotification({
        type: 'success',
        title: 'Subject 3D Laboratory Loaded',
        message: parts.length > 0 ? parts.join(' • ') : 'Using cached scientific data',
        duration: 4000,
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
        <div className="w-full h-full relative overflow-hidden flex flex-col">
          {/* Viewport Layout Mode Selector */}
          <div className="absolute top-4 right-4 z-30 flex items-center bg-neutral-900/90 backdrop-blur-md border border-neutral-700/80 rounded-xl p-1 shadow-xl text-xs">
            <button
              onClick={() => setViewportLayout('3d')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                viewportLayout === '3d'
                  ? 'bg-primary-600 text-white shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              3D Only
            </button>
            <button
              onClick={() => setViewportLayout('split')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                viewportLayout === 'split'
                  ? 'bg-primary-600 text-white shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Split MPR (3D + 2D)
            </button>
            <button
              onClick={() => setViewportLayout('2d')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                viewportLayout === '2d'
                  ? 'bg-primary-600 text-white shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              2D Slices Only
            </button>
          </div>

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
            <>
              {/* 3D Viewer Canvas */}
              {viewportLayout !== '2d' && (
                <div className="w-full h-full relative flex-1">
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
                </div>
              )}

              {/* 2D Orthogonal Slice Viewer: Docked at bottom for Split Mode */}
              {viewportLayout === 'split' && (
                <div className="absolute bottom-4 left-4 right-4 z-20 max-h-[46vh] overflow-y-auto">
                  <OrthogonalSliceViewer subjectId={activeSubject || 'SUB1'} />
                </div>
              )}

              {/* 2D Orthogonal Slice Viewer: Full Page Mode */}
              {viewportLayout === '2d' && (
                <div className="w-full h-full p-6 overflow-y-auto">
                  <div className="max-w-6xl mx-auto pt-10">
                    <OrthogonalSliceViewer subjectId={activeSubject || 'SUB1'} isCollapsible={false} />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
