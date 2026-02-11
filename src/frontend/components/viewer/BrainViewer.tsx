'use client';

import { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stats } from '@react-three/drei';
import { useAppStore } from '@/lib/store';
import StreamlineRenderer from './StreamlineRenderer';
import SliceViewer from './SliceViewer';
import BrainSurface from './BrainSurface';
import BrainModel from './BrainModel';
import ViewPresets from './ViewPresets';
import AnatomicalLabels from './AnatomicalLabels';
import * as THREE from 'three';

interface BrainViewerProps {
  onError?: (error: string) => void;
}

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[20, 16, 16]} />
      <meshBasicMaterial color="#4a9eff" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

export default function BrainViewer({ onError }: BrainViewerProps) {
  const { viewerSettings, streamlineBundle, brainMesh } = useAppStore();
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
        case '1':
          useAppStore.getState().updateViewerSettings({ brainModelType: 'hologram' });
          break;
        case '2':
          useAppStore.getState().updateViewerSettings({ brainModelType: 'point_cloud' });
          break;
        case '3':
          useAppStore.getState().updateViewerSettings({ brainModelType: 'marching_cubes' });
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewerSettings.showSlices, viewerSettings.showBrainSurface, viewerSettings.showLabels, showStats, viewerSettings.showStreamlines, viewerSettings.autoRotate, viewerSettings.brainSurfaceWireframe]);

  if (glError) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="glass rounded-xl p-8 max-w-md text-center">
          <h3 className="text-xl font-semibold text-red-400 mb-3">WebGL Error</h3>
          <p className="text-gray-300 text-sm mb-4">{glError}</p>
          <button onClick={() => setGlError(null)} className="px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm transition-colors">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative canvas-container">
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          failIfMajorPerformanceCaveat: false,
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color(viewerSettings.backgroundColor));
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.2;
          const canvas = gl.domElement;
          canvas.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            console.warn('WebGL context lost - will attempt restore');
            setGlError('WebGL context was lost. This may happen with large models. Click Retry to reload.');
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

        {/* Enhanced Lighting */}
        <ambientLight intensity={0.35} />
        <hemisphereLight args={['#b1e1ff', '#b97a20', 0.5]} />
        <directionalLight position={[100, 100, 50]} intensity={0.9} castShadow />
        <directionalLight position={[-100, -50, -50]} intensity={0.3} />
        <directionalLight position={[0, 100, -100]} intensity={0.2} />
        <pointLight position={[0, 0, 150]} intensity={0.4} color="#88ccff" />
        <pointLight position={[0, -100, 0]} intensity={0.15} color="#ff8866" />

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
          {/* Brain Surface - GLB models or marching cubes */}
          {viewerSettings.showBrainSurface && (
            <>
              {viewerSettings.brainModelType === 'marching_cubes' && brainMesh && (
                <BrainSurface mesh={brainMesh} settings={viewerSettings} />
              )}
              {viewerSettings.brainModelType === 'hologram' && (
                <BrainModel modelType="hologram" settings={viewerSettings} streamlineBounds={streamlineBundle?.bounds} />
              )}
              {viewerSettings.brainModelType === 'point_cloud' && (
                <BrainModel modelType="point_cloud" settings={viewerSettings} streamlineBounds={streamlineBundle?.bounds} />
              )}
            </>
          )}

          {/* Streamlines */}
          {viewerSettings.showStreamlines && streamlineBundle && (
            <StreamlineRenderer
              bundle={streamlineBundle}
              settings={viewerSettings}
            />
          )}

          {/* Anatomical Labels */}
          {viewerSettings.showLabels && <AnatomicalLabels />}

          {/* Slices */}
          {viewerSettings.showSlices && <SliceViewer />}

          {/* Coordinate Axes */}
          <axesHelper args={[50]} />
        </Suspense>

        {/* Performance Stats */}
        {showStats && <Stats />}
      </Canvas>

      {/* View Presets Toolbar */}
      <ViewPresets controlsRef={controlsRef} />

      {/* Model Indicator */}
      <div className="absolute top-4 right-4 glass rounded-lg px-3 py-2 text-xs text-gray-300">
        <span className="text-primary-400 font-medium">
          {viewerSettings.brainModelType === 'hologram' ? 'Hologram' :
           viewerSettings.brainModelType === 'point_cloud' ? 'Point Cloud' : 'MRI Mesh'}
        </span>
        {viewerSettings.autoRotate && <span className="ml-2 text-cyan-400">Rotating</span>}
      </div>

      {/* Info Overlay */}
      <div className="absolute top-4 left-4 glass rounded-lg p-3 text-sm max-w-xs">
        <p className="font-semibold mb-1">Scene Info</p>
        {streamlineBundle ? (
          <>
            <p className="text-gray-300">
              Streamlines: {streamlineBundle.metadata.count.toLocaleString()}
              {streamlineBundle.metadata.totalInFile && (
                <span className="text-gray-500"> / {streamlineBundle.metadata.totalInFile.toLocaleString()}</span>
              )}
            </p>
            <p className="text-gray-300">
              Points: {streamlineBundle.metadata.totalPoints.toLocaleString()}
            </p>
            <p className="text-gray-300">
              Mean Length: {streamlineBundle.metadata.meanLength.toFixed(1)} mm
            </p>
          </>
        ) : (
          <p className="text-gray-400">No streamline data</p>
        )}
        {brainMesh && (
          <p className="text-gray-300 mt-1">
            MRI Mesh: {brainMesh.metadata.n_vertices.toLocaleString()} vertices
          </p>
        )}
      </div>

      {/* Controls Help */}
      <div className="absolute bottom-4 right-4 glass rounded-lg p-3 text-xs text-gray-400 hidden md:block">
        <p>LMB: Rotate | RMB: Pan | Scroll: Zoom</p>
        <p>&apos;R&apos; Reset | &apos;B&apos; Brain | &apos;T&apos; Tracts | &apos;L&apos; Labels</p>
        <p>&apos;A&apos; Auto-Rotate | &apos;W&apos; Wireframe | &apos;1/2/3&apos; Models</p>
      </div>
    </div>
  );
}
