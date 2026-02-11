'use client';

import { useCallback } from 'react';
import { ViewPreset } from '@/lib/types';

const VIEW_PRESETS: ViewPreset[] = [
  { name: 'anterior', label: 'Anterior', cameraPosition: [0, 300, 0], cameraTarget: [0, 0, 0] },
  { name: 'posterior', label: 'Posterior', cameraPosition: [0, -300, 0], cameraTarget: [0, 0, 0] },
  { name: 'left', label: 'Left', cameraPosition: [-300, 0, 0], cameraTarget: [0, 0, 0] },
  { name: 'right', label: 'Right', cameraPosition: [300, 0, 0], cameraTarget: [0, 0, 0] },
  { name: 'superior', label: 'Superior', cameraPosition: [0, 0, 300], cameraTarget: [0, 0, 0] },
  { name: 'inferior', label: 'Inferior', cameraPosition: [0, 0, -300], cameraTarget: [0, 0, 0] },
  { name: 'default', label: '3/4 View', cameraPosition: [150, 150, 200], cameraTarget: [0, 0, 0] },
];

interface ViewPresetsProps {
  controlsRef: React.RefObject<any>;
}

export default function ViewPresets({ controlsRef }: ViewPresetsProps) {
  const handlePreset = useCallback((preset: ViewPreset) => {
    if (!controlsRef.current) return;

    const controls = controlsRef.current;

    // Set target
    controls.target.set(...preset.cameraTarget);

    // Set camera position
    controls.object.position.set(...preset.cameraPosition);

    controls.update();
  }, [controlsRef]);

  return (
    <div className="absolute top-4 right-4 z-10 flex flex-col gap-1">
      <div className="glass rounded-lg p-2">
        <p className="text-xs text-gray-400 mb-2 font-medium px-1">Views</p>
        <div className="grid grid-cols-2 gap-1">
          {VIEW_PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => handlePreset(preset)}
              className="px-2 py-1.5 text-xs rounded bg-white/5 hover:bg-white/15 transition-colors text-gray-300 hover:text-white"
              title={preset.label}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
