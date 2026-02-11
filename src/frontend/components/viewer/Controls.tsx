'use client';

import { useAppStore } from '@/lib/store';
import { COLORMAPS } from '@/lib/utils';
import { BrainModelType } from '@/lib/types';

const BRAIN_MODEL_OPTIONS: { value: BrainModelType; label: string; desc: string }[] = [
  { value: 'hologram', label: 'Hologram', desc: 'Detailed 3D brain model' },
  { value: 'point_cloud', label: 'Point Cloud', desc: 'Volumetric point representation' },
  { value: 'marching_cubes', label: 'MRI Mesh', desc: 'Generated from brain mask data' },
];

export default function Controls() {
  const { viewerSettings, updateViewerSettings, brainMesh } = useAppStore();

  return (
    <div className="space-y-6">
      {/* Brain Model Section */}
      <div className="pb-4 border-b border-white/10">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Brain Model</h3>

        <Toggle
          label="Show Brain"
          checked={viewerSettings.showBrainSurface}
          onChange={(checked) => updateViewerSettings({ showBrainSurface: checked })}
        />

        {viewerSettings.showBrainSurface && (
          <>
            {/* Model Type Selector */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">Model Type</label>
              <div className="space-y-1.5">
                {BRAIN_MODEL_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => updateViewerSettings({ brainModelType: opt.value })}
                    disabled={opt.value === 'marching_cubes' && !brainMesh}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                      viewerSettings.brainModelType === opt.value
                        ? 'bg-primary-600/40 border border-primary-500/50 text-white'
                        : opt.value === 'marching_cubes' && !brainMesh
                        ? 'bg-white/5 border border-white/5 text-gray-500 cursor-not-allowed'
                        : 'bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10'
                    }`}
                  >
                    <span className="font-medium">{opt.label}</span>
                    <span className="block text-xs text-gray-400 mt-0.5">{opt.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Opacity */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">
                Opacity: {(viewerSettings.brainSurfaceOpacity * 100).toFixed(0)}%
              </label>
              <input
                type="range" min="0.02" max="1" step="0.01"
                value={viewerSettings.brainSurfaceOpacity}
                onChange={(e) => updateViewerSettings({ brainSurfaceOpacity: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Surface Color */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">Surface Color</label>
              <div className="flex gap-2 items-center">
                <input
                  type="color"
                  value={viewerSettings.brainSurfaceColor}
                  onChange={(e) => updateViewerSettings({ brainSurfaceColor: e.target.value })}
                  className="w-10 h-8 rounded cursor-pointer bg-transparent"
                />
                <button
                  onClick={() => updateViewerSettings({ brainSurfaceColor: '#e8d5cf' })}
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Wireframe */}
            <div className="mt-3">
              <Toggle
                label="Wireframe"
                checked={viewerSettings.brainSurfaceWireframe}
                onChange={(checked) => updateViewerSettings({ brainSurfaceWireframe: checked })}
              />
            </div>

            {/* Material Properties */}
            <div className="mt-3 space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Emissive: {(viewerSettings.brainEmissiveIntensity * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min="0" max="1" step="0.01"
                  value={viewerSettings.brainEmissiveIntensity}
                  onChange={(e) => updateViewerSettings({ brainEmissiveIntensity: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Metalness: {(viewerSettings.brainMetalness * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min="0" max="1" step="0.01"
                  value={viewerSettings.brainMetalness}
                  onChange={(e) => updateViewerSettings({ brainMetalness: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Roughness: {(viewerSettings.brainRoughness * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min="0" max="1" step="0.01"
                  value={viewerSettings.brainRoughness}
                  onChange={(e) => updateViewerSettings({ brainRoughness: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>
            </div>
          </>
        )}
      </div>

      {/* Streamlines Section */}
      <div className="pb-4 border-b border-white/10">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Streamlines</h3>

        <Toggle
          label="Show Streamlines"
          checked={viewerSettings.showStreamlines}
          onChange={(checked) => updateViewerSettings({ showStreamlines: checked })}
        />

        {viewerSettings.showStreamlines && (
          <>
            {/* Color Mapping */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">Color By</label>
              <select
                value={viewerSettings.colorMapping.type}
                onChange={(e) =>
                  updateViewerSettings({
                    colorMapping: { ...viewerSettings.colorMapping, type: e.target.value as any },
                  })
                }
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="orientation">Orientation (RGB)</option>
                <option value="length">Length</option>
                <option value="fa">Fractional Anisotropy</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            {/* Colormap */}
            {viewerSettings.colorMapping.type !== 'orientation' && (
              <div className="mt-3">
                <label className="block text-sm font-medium mb-2">Colormap</label>
                <select
                  value={viewerSettings.colorMapping.colormap}
                  onChange={(e) =>
                    updateViewerSettings({
                      colorMapping: { ...viewerSettings.colorMapping, colormap: e.target.value },
                    })
                  }
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  {COLORMAPS.map((map) => (
                    <option key={map} value={map}>
                      {map.charAt(0).toUpperCase() + map.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Opacity */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">
                Opacity: {(viewerSettings.streamlineOpacity * 100).toFixed(0)}%
              </label>
              <input
                type="range" min="0" max="1" step="0.01"
                value={viewerSettings.streamlineOpacity}
                onChange={(e) => updateViewerSettings({ streamlineOpacity: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Line Width */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">
                Line Width: {viewerSettings.streamlineWidth.toFixed(1)}
              </label>
              <input
                type="range" min="0.1" max="5" step="0.1"
                value={viewerSettings.streamlineWidth}
                onChange={(e) => updateViewerSettings({ streamlineWidth: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Level of Detail */}
            <div className="mt-3">
              <label className="block text-sm font-medium mb-2">Level of Detail</label>
              <select
                value={viewerSettings.levelOfDetail}
                onChange={(e) => updateViewerSettings({ levelOfDetail: e.target.value as any })}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="low">Low (20%)</option>
                <option value="medium">Medium (Dynamic)</option>
                <option value="high">High (100%)</option>
              </select>
            </div>
          </>
        )}
      </div>

      {/* Animation Section */}
      <div className="pb-4 border-b border-white/10">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Animation</h3>

        <Toggle
          label="Auto-Rotate"
          checked={viewerSettings.autoRotate}
          onChange={(checked) => updateViewerSettings({ autoRotate: checked })}
        />

        {viewerSettings.autoRotate && (
          <div className="mt-3">
            <label className="block text-sm font-medium mb-2">
              Speed: {viewerSettings.autoRotateSpeed.toFixed(1)}x
            </label>
            <input
              type="range" min="0.1" max="5" step="0.1"
              value={viewerSettings.autoRotateSpeed}
              onChange={(e) => updateViewerSettings({ autoRotateSpeed: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>
        )}
      </div>

      {/* Labels & Overlays */}
      <div className="pb-4 border-b border-white/10">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Overlays</h3>

        <div className="space-y-3">
          <Toggle
            label="Anatomical Labels"
            checked={viewerSettings.showLabels}
            onChange={(checked) => updateViewerSettings({ showLabels: checked })}
          />
          <Toggle
            label="Show Slices"
            checked={viewerSettings.showSlices}
            onChange={(checked) => updateViewerSettings({ showSlices: checked })}
          />
        </div>
      </div>

      {/* Slice Positions */}
      {viewerSettings.showSlices && (
        <div className="pb-4 border-b border-white/10">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Slice Position</h3>
          <div className="space-y-3">
            <SliceControl
              label="Axial"
              value={viewerSettings.slicePosition.axial}
              onChange={(value) =>
                updateViewerSettings({ slicePosition: { ...viewerSettings.slicePosition, axial: value } })
              }
            />
            <SliceControl
              label="Coronal"
              value={viewerSettings.slicePosition.coronal}
              onChange={(value) =>
                updateViewerSettings({ slicePosition: { ...viewerSettings.slicePosition, coronal: value } })
              }
            />
            <SliceControl
              label="Sagittal"
              value={viewerSettings.slicePosition.sagittal}
              onChange={(value) =>
                updateViewerSettings({ slicePosition: { ...viewerSettings.slicePosition, sagittal: value } })
              }
            />
          </div>
        </div>
      )}

      {/* Background */}
      <div>
        <label className="block text-sm font-medium mb-2">Background Color</label>
        <div className="flex gap-2 items-center">
          <input
            type="color"
            value={viewerSettings.backgroundColor}
            onChange={(e) => updateViewerSettings({ backgroundColor: e.target.value })}
            className="w-10 h-8 rounded cursor-pointer bg-transparent"
          />
          <div className="flex gap-1">
            {['#1a1a2e', '#000000', '#0a0a1a', '#1a2a1a'].map((c) => (
              <button
                key={c}
                onClick={() => updateViewerSettings({ backgroundColor: c })}
                className={`w-6 h-6 rounded border ${viewerSettings.backgroundColor === c ? 'border-primary-400' : 'border-white/20'}`}
                style={{ backgroundColor: c }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex items-center justify-between cursor-pointer">
      <span className="text-sm">{label}</span>
      <div className="relative">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="sr-only" />
        <div className={`w-11 h-6 rounded-full transition-colors ${checked ? 'bg-primary-600' : 'bg-gray-600'}`}>
          <div className={`absolute left-1 top-1 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'transform translate-x-5' : ''}`} />
        </div>
      </div>
    </label>
  );
}

function SliceControl({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-2">
        {label}: {(value * 100).toFixed(0)}%
      </label>
      <input
        type="range" min="0" max="1" step="0.01"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
