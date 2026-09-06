'use client';

import { useCallback } from 'react';
import { ViewPreset } from '@/lib/types';

const BRAIN_CENTROID: [number, number, number] = [0, -12, 18];

const VIEW_PRESETS: ViewPreset[] = [
  { name: 'anterior', label: 'Anterior', cameraPosition: [0, -260, 18], cameraTarget: BRAIN_CENTROID },
  { name: 'posterior', label: 'Posterior', cameraPosition: [0, 240, 18], cameraTarget: BRAIN_CENTROID },
  { name: 'left', label: 'Left', cameraPosition: [-260, -12, 18], cameraTarget: BRAIN_CENTROID },
  { name: 'right', label: 'Right', cameraPosition: [260, -12, 18], cameraTarget: BRAIN_CENTROID },
  { name: 'superior', label: 'Superior', cameraPosition: [0, -12, 280], cameraTarget: BRAIN_CENTROID },
  { name: 'inferior', label: 'Inferior', cameraPosition: [0, -12, -250], cameraTarget: BRAIN_CENTROID },
  { name: 'default', label: '3/4 Perspective', cameraPosition: [160, -200, 110], cameraTarget: BRAIN_CENTROID },
];

interface ViewPresetsProps {
  controlsRef: React.RefObject<any>;
}

export default function ViewPresets({ controlsRef }: ViewPresetsProps) {
  const handlePreset = useCallback((preset: ViewPreset) => {
    if (!controlsRef.current) return;

    const controls = controlsRef.current;
    controls.target.set(...preset.cameraTarget);
    controls.object.position.set(...preset.cameraPosition);
    controls.update();
  }, [controlsRef]);

  const handleZoom = useCallback((factor: number) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    const camera = controls.object;
    const target = controls.target;
    // Scale distance to target
    camera.position.sub(target).multiplyScalar(factor).add(target);
    controls.update();
  }, [controlsRef]);

  const handleReset = useCallback(() => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    controls.target.set(BRAIN_CENTROID[0], BRAIN_CENTROID[1], BRAIN_CENTROID[2]);
    controls.object.position.set(0, -220, 80);
    controls.update();
  }, [controlsRef]);

  return (
    <div className="absolute top-20 right-4 z-10 flex flex-col gap-2">
      <div className="bg-slate-900/85 backdrop-blur-xl border border-white/[0.08] rounded-xl p-2.5 shadow-2xl text-xs">
        <div className="flex items-center justify-between mb-2 px-1">
          <p className="text-[10px] font-mono tracking-wider uppercase text-slate-400 font-semibold">
            Anatomical View
          </p>
          <button
            onClick={handleReset}
            className="text-[10px] text-cyan-400 hover:text-cyan-300 transition-colors"
            title="Reset Camera Target & Zoom"
          >
            Reset
          </button>
        </div>
        <div className="grid grid-cols-2 gap-1.5 mb-2">
          {VIEW_PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => handlePreset(preset)}
              className="px-2.5 py-1.5 text-xs font-medium rounded-lg bg-white/[0.04] hover:bg-white/[0.1] border border-white/[0.05] hover:border-white/[0.15] transition-all text-slate-200 hover:text-white active:scale-95 shadow-sm"
              title={`Orient camera to ${preset.label}`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Zoom Controls */}
        <div className="flex gap-1 pt-1.5 border-t border-white/[0.06]">
          <button
            onClick={() => handleZoom(0.8)}
            className="flex-1 py-1 px-2 text-xs font-semibold rounded bg-white/[0.04] hover:bg-white/[0.1] border border-white/[0.05] text-slate-300 hover:text-white"
            title="Zoom In"
          >
            + Zoom In
          </button>
          <button
            onClick={() => handleZoom(1.25)}
            className="flex-1 py-1 px-2 text-xs font-semibold rounded bg-white/[0.04] hover:bg-white/[0.1] border border-white/[0.05] text-slate-300 hover:text-white"
            title="Zoom Out"
          >
            - Zoom Out
          </button>
        </div>
      </div>
    </div>
  );
}
