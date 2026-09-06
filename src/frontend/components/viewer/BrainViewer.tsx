'use client';

import { useRef, useEffect, useState, Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stats } from '@react-three/drei';
import { useAppStore } from '@/lib/store';
import StreamlineRenderer from './StreamlineRenderer';
import SliceViewer from './SliceViewer';
import BrainSurface from './BrainSurface';
import ViewPresets from './ViewPresets';
import AnatomicalLabels from './AnatomicalLabels';
import ConnectomeGraph3D from './ConnectomeGraph3D';
import * as THREE from 'three';

interface BrainViewerProps {
  onError?: (error: string) => void;
}

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[20, 16, 16]} />
      <meshBasicMaterial color="#10b981" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

export default function BrainViewer({ onError }: BrainViewerProps) {
  const {
    viewerSettings,
    streamlineBundle,
    brainMesh,
    connectome,
    parcellationLabels,
    hoveredEdge,
    selectedEdge,
    hoveredRegion,
    selectedRegion,
    setSelectedRegion,
    setSelectedEdge,
  } = useAppStore();
  const controlsRef = useRef<any>(null);
  const [showStats, setShowStats] = useState(false);
  const [glError, setGlError] = useState<string | null>(null);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key.toLowerCase()) {
        case 'r':
          if (controlsRef.current) {
            controlsRef.current.reset();
          }
          break;
        case 's':
          useAppStore.getState().updateViewerSettings({
            showSlices: !viewerSettings.showSlices,
          });
          break;
        case 'b':
          useAppStore.getState().updateViewerSettings({
            showBrainSurface: !viewerSettings.showBrainSurface,
          });
          break;
        case 'l':
          useAppStore.getState().updateViewerSettings({
            showLabels: !viewerSettings.showLabels,
          });
          break;
        case 'c':
          useAppStore.getState().updateViewerSettings({
            showConnectomeGraph: !viewerSettings.showConnectomeGraph,
          });
          break;
        case 'g':
          setShowStats(!showStats);
          break;
        case 't':
          useAppStore.getState().updateViewerSettings({
            showStreamlines: !viewerSettings.showStreamlines,
          });
          break;
        case 'a':
          useAppStore.getState().updateViewerSettings({
            autoRotate: !viewerSettings.autoRotate,
          });
          break;
        case 'w':
          useAppStore.getState().updateViewerSettings({
            brainSurfaceWireframe: !viewerSettings.brainSurfaceWireframe,
          });
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [viewerSettings, showStats]);

  // Active region inspector details
  const activeRegionIndex = selectedRegion !== null ? selectedRegion : hoveredRegion;
  const activeRegionLabel = useMemo(() => {
    if (activeRegionIndex === null || !parcellationLabels) return null;
    return parcellationLabels[activeRegionIndex] || null;
  }, [activeRegionIndex, parcellationLabels]);

  const activeRegionDegree = useMemo(() => {
    if (activeRegionIndex === null || !connectome || !connectome[activeRegionIndex]) return 0;
    return connectome[activeRegionIndex].filter((w) => w > 0).length;
  }, [activeRegionIndex, connectome]);

  const activeEdgeInfo = selectedEdge || hoveredEdge;

  if (glError) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 max-w-md text-center">
          <h3 className="text-xl font-semibold text-red-400 mb-3">WebGL Error</h3>
          <p className="text-slate-300 text-sm mb-4">{glError}</p>
          <button onClick={() => setGlError(null)} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm transition-colors">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative canvas-container bg-slate-950">
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          failIfMajorPerformanceCaveat: false,
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color(viewerSettings.backgroundColor || '#090d16'));
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.2;
          const canvas = gl.domElement;
          canvas.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            console.warn('WebGL context lost - will attempt restore');
            setGlError('WebGL context was lost. Click Retry to reload.');
          });
          canvas.addEventListener('webglcontextrestored', () => {
            console.info('WebGL context restored');
            setGlError(null);
          });
        }}
      >
        {/* Camera */}
        <PerspectiveCamera
          makeDefault
          position={viewerSettings.cameraPosition as [number, number, number]}
          fov={50}
          near={0.1}
          far={10000}
        />

        {/* Enhanced Scientific Lighting */}
        <ambientLight intensity={0.4} />
        <hemisphereLight args={['#b1e1ff', '#1e293b', 0.6]} />
        <directionalLight position={[100, 100, 50]} intensity={0.9} castShadow />
        <directionalLight position={[-100, -50, -50]} intensity={0.3} />
        <directionalLight position={[0, 100, -100]} intensity={0.2} />
        <pointLight position={[0, 0, 150]} intensity={0.4} color="#88ccff" />
        <pointLight position={[0, -100, 0]} intensity={0.15} color="#34d399" />

        {/* Controls */}
        <OrbitControls
          ref={controlsRef}
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          panSpeed={0.5}
          zoomSpeed={0.8}
          minDistance={10}
          maxDistance={1000}
          autoRotate={viewerSettings.autoRotate}
          autoRotateSpeed={viewerSettings.autoRotateSpeed}
          makeDefault
        />

        {/* Scene Content */}
        <Suspense fallback={<LoadingFallback />}>
          {/* Strictly Subject-Derived Brain Surface (Marching Cubes) */}
          {viewerSettings.showBrainSurface && brainMesh && (
            <BrainSurface mesh={brainMesh} settings={viewerSettings} />
          )}

          {/* Real Reconstructed Tractography Streamlines */}
          {viewerSettings.showStreamlines && streamlineBundle && (
            <StreamlineRenderer
              bundle={streamlineBundle}
              settings={viewerSettings}
            />
          )}

          {/* 3D Connectome Graph Layer (Nodes & Weighted Edges) */}
          {viewerSettings.showConnectomeGraph && connectome && parcellationLabels.length > 0 && (
            <ConnectomeGraph3D
              connectome={connectome}
              labels={parcellationLabels}
              threshold={viewerSettings.connectomeEdgeThreshold || 1.0}
            />
          )}

          {/* Anatomical Labels */}
          {viewerSettings.showLabels && <AnatomicalLabels />}

          {/* In-scene Orthogonal Slices */}
          {viewerSettings.showSlices && <SliceViewer />}

          {/* Coordinate Axes in RAS+ mm */}
          <axesHelper args={[50]} />
        </Suspense>

        {/* Performance Stats */}
        {showStats && <Stats />}
      </Canvas>

      {/* View Presets Toolbar */}
      <ViewPresets controlsRef={controlsRef} />

      {/* Model & Provenance Indicator */}
      <div className="absolute top-4 right-4 bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300 shadow-lg space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="font-semibold text-white">Subject Laboratory</span>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/60">
            RAS+ mm
          </span>
        </div>
        <p className="text-[10px] text-slate-400">
          Marching Cubes Surface • Probabilistic Tracts • Desikan 89
        </p>
      </div>

      {/* Real Scene Info Overlay */}
      <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-lg p-3 text-xs max-w-xs shadow-lg space-y-1">
        <p className="font-semibold text-slate-100 flex items-center justify-between">
          <span>Scene Telemetry</span>
          <span className="text-[10px] font-mono text-slate-500">Real dMRI</span>
        </p>
        {streamlineBundle ? (
          <div className="space-y-0.5 text-slate-300 font-mono text-[11px]">
            <div>
              Tracts:{' '}
              <span className="text-cyan-300 font-semibold">
                {streamlineBundle.metadata.count.toLocaleString()}
              </span>
              {streamlineBundle.metadata.totalInFile && (
                <span className="text-slate-500"> / {streamlineBundle.metadata.totalInFile.toLocaleString()}</span>
              )}
            </div>
            <div>
              Mean Length:{' '}
              <span className="text-slate-200">
                {streamlineBundle.metadata.meanLength.toFixed(1)} mm
              </span>
            </div>
          </div>
        ) : (
          <p className="text-slate-500">No streamline data loaded</p>
        )}
        {brainMesh && (
          <div className="text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800">
            Surface Mesh: {brainMesh.metadata.n_vertices.toLocaleString()} vertices
          </div>
        )}
        {connectome && (
          <div className="text-[11px] font-mono text-emerald-400">
            Connectome: {connectome.length} parcels
          </div>
        )}
      </div>

      {/* Floating 3D Interactive Inspector (When parcel or edge is hovered/selected) */}
      {(activeRegionLabel || activeEdgeInfo) && (
        <div className="absolute bottom-16 left-4 bg-slate-900/95 backdrop-blur-md border border-slate-700/80 rounded-xl p-3.5 shadow-2xl text-xs font-mono max-w-sm transition-all z-20">
          {activeRegionLabel && (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1.5">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider">
                  {selectedRegion !== null ? 'Selected Parcel' : 'Hovered Parcel'}
                </span>
                <button
                  onClick={() => setSelectedRegion(null)}
                  className="text-[10px] text-slate-500 hover:text-slate-300"
                >
                  close
                </button>
              </div>
              <h4 className="font-semibold text-cyan-300 text-sm">{activeRegionLabel.name}</h4>
              <div className="text-slate-300 text-[11px] space-y-0.5">
                <div>
                  Lobe: <span className="text-white capitalize">{activeRegionLabel.lobe}</span> • Hemi:{' '}
                  <span className="text-white capitalize">{activeRegionLabel.hemisphere}</span>
                </div>
                <div>
                  Centroid: [{activeRegionLabel.centroid?.map((c) => c.toFixed(1)).join(', ')}] mm
                </div>
                <div>
                  Structural Degree: <span className="text-emerald-400 font-bold">{activeRegionDegree}</span> connections
                </div>
              </div>
            </div>
          )}

          {activeEdgeInfo && !activeRegionLabel && (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1.5">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider">Structural Tract Connection</span>
                <button
                  onClick={() => setSelectedEdge(null)}
                  className="text-[10px] text-slate-500 hover:text-slate-300"
                >
                  close
                </button>
              </div>
              <div className="text-slate-200 font-medium">
                {parcellationLabels[activeEdgeInfo.source]?.name || `Node ${activeEdgeInfo.source}`}
                <span className="text-slate-500 mx-1.5">↔</span>
                {parcellationLabels[activeEdgeInfo.target]?.name || `Node ${activeEdgeInfo.target}`}
              </div>
              <div className="text-[11px] text-emerald-400 font-semibold">
                Weight: {activeEdgeInfo.weight.toFixed(0)} reconstructed streamlines
              </div>
            </div>
          )}
        </div>
      )}

      {/* Keyboard Shortcuts Bar */}
      <div className="absolute bottom-4 right-4 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-400 hidden md:block shadow-lg">
        <p>LMB: Rotate | RMB: Pan | Scroll: Zoom</p>
        <p>&apos;R&apos; Reset | &apos;B&apos; Brain Mesh | &apos;T&apos; Tracts | &apos;C&apos; Connectome | &apos;L&apos; Labels | &apos;A&apos; Auto-Rotate</p>
      </div>
    </div>
  );
}

