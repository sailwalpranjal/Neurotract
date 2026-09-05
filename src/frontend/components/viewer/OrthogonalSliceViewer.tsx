'use client';

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { OrthogonalSlicesData } from '@/lib/types';

interface OrthogonalSliceViewerProps {
  subjectId?: string;
  isCollapsible?: boolean;
}

export default function OrthogonalSliceViewer({
  subjectId: propSubjectId,
  isCollapsible = true,
}: OrthogonalSliceViewerProps) {
  const {
    activeSubject,
    slicesData,
    setSlicesData,
    sliceIndices,
    setSliceIndices,
    activeSliceType,
    setActiveSliceType,
  } = useAppStore();

  const currentSubject = propSubjectId || activeSubject || 'SUB1';

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number; z: number; plane: string } | null>(null);

  // Default coordinate if null
  const currentCoord = sliceIndices || {
    x: slicesData?.indices.sagittal ?? 70,
    y: slicesData?.indices.coronal ?? 70,
    z: slicesData?.indices.axial ?? 46,
  };

  const dims = slicesData?.volume_shape || [140, 140, 92];
  const voxelSize = slicesData?.voxel_size_mm || [1.79, 1.79, 2.0];

  const getImageUrl = (b64?: string) => {
    if (!b64) return '';
    return b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
  };

  const fetchSlices = useCallback(
    async (coords?: { x: number; y: number; z: number }, volType?: 'b0' | 'fa' | 'md' | 't1') => {
      if (!currentSubject) return;
      setLoading(true);
      setError(null);

      const targetVol = volType || activeSliceType;
      const targetX = coords ? coords.x : currentCoord.x;
      const targetY = coords ? coords.y : currentCoord.y;
      const targetZ = coords ? coords.z : currentCoord.z;

      try {
        const data = await apiClient.getOrthogonalSlices(currentSubject, {
          volume_type: targetVol,
          x: targetX,
          y: targetY,
          z: targetZ,
        });
        setSlicesData(data);
        setSliceIndices({
          x: data.indices.sagittal,
          y: data.indices.coronal,
          z: data.indices.axial,
        });
      } catch (err: any) {
        setError(err.message || 'Failed to load orthogonal slice planes');
      } finally {
        setLoading(false);
      }
    },
    [currentSubject, activeSliceType, setSlicesData, setSliceIndices, currentCoord.x, currentCoord.y, currentCoord.z]
  );

  // Fetch initial slices when subject or volume type changes
  useEffect(() => {
    fetchSlices(undefined, activeSliceType);
  }, [currentSubject, activeSliceType]);

  // Click on Axial plane: (X, Y) clicked -> update X and Y, keep Z
  const handleAxialClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));

    const newX = Math.round(normX * (dims[0] - 1));
    const newY = Math.round((1 - normY) * (dims[1] - 1)); // Invert Y for radiological orientation

    const nextCoord = { x: newX, y: newY, z: currentCoord.z };
    setSliceIndices(nextCoord);
    fetchSlices(nextCoord);
  };

  // Click on Coronal plane: (X, Z) clicked -> update X and Z, keep Y
  const handleCoronalClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normZ = Math.max(0, Math.min(1, clickY / rect.height));

    const newX = Math.round(normX * (dims[0] - 1));
    const newZ = Math.round((1 - normZ) * (dims[2] - 1)); // Top is Superior

    const nextCoord = { x: newX, y: currentCoord.y, z: newZ };
    setSliceIndices(nextCoord);
    fetchSlices(nextCoord);
  };

  // Click on Sagittal plane: (Y, Z) clicked -> update Y and Z, keep X
  const handleSagittalClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickY = e.clientX - rect.left;
    const clickZ = e.clientY - rect.top;

    const normY = Math.max(0, Math.min(1, clickY / rect.width));
    const normZ = Math.max(0, Math.min(1, clickZ / rect.height));

    const newY = Math.round(normY * (dims[1] - 1));
    const newZ = Math.round((1 - normZ) * (dims[2] - 1)); // Top is Superior

    const nextCoord = { x: currentCoord.x, y: newY, z: newZ };
    setSliceIndices(nextCoord);
    fetchSlices(nextCoord);
  };

  const handleAxialHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));
    const hX = Math.round(normX * (dims[0] - 1));
    const hY = Math.round((1 - normY) * (dims[1] - 1));
    setHoverCoord({ x: hX, y: hY, z: currentCoord.z, plane: 'Axial' });
  };

  const handleCoronalHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickZ = e.clientY - rect.top;
    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normZ = Math.max(0, Math.min(1, clickZ / rect.height));
    const hX = Math.round(normX * (dims[0] - 1));
    const hZ = Math.round((1 - normZ) * (dims[2] - 1));
    setHoverCoord({ x: hX, y: currentCoord.y, z: hZ, plane: 'Coronal' });
  };

  const handleSagittalHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickY = e.clientX - rect.left;
    const clickZ = e.clientY - rect.top;
    const normY = Math.max(0, Math.min(1, clickY / rect.width));
    const normZ = Math.max(0, Math.min(1, clickZ / rect.height));
    const hY = Math.round(normY * (dims[1] - 1));
    const hZ = Math.round((1 - normZ) * (dims[2] - 1));
    setHoverCoord({ x: currentCoord.x, y: hY, z: hZ, plane: 'Sagittal' });
  };

  const handleHoverLeave = () => {
    setHoverCoord(null);
  };

  return (
    <div className="bg-neutral-900/95 backdrop-blur-md border border-neutral-800 rounded-2xl p-4 text-gray-100 shadow-2xl transition-all">
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-neutral-800">
        <div className="flex items-center space-x-2.5">
          <span className="p-1.5 rounded-lg bg-primary-500/20 text-primary-400 border border-primary-500/30">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
          </span>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Synchronized Orthogonal Anatomical Slices
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-neutral-700">
                NIfTI 3D
              </span>
            </h3>
            <p className="text-[11px] text-neutral-400">
              Interactive Axial, Coronal, and Sagittal crosshair navigation
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Volume Type Select */}
          <div className="flex bg-neutral-950 p-0.5 rounded-lg border border-neutral-800 text-xs">
            <button
              onClick={() => setActiveSliceType('b0')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeSliceType === 'b0'
                  ? 'bg-primary-600 text-white font-medium shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              B0 Vol
            </button>
            <button
              onClick={() => setActiveSliceType('fa')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeSliceType === 'fa'
                  ? 'bg-primary-600 text-white font-medium shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              FA Map
            </button>
            <button
              onClick={() => setActiveSliceType('md')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeSliceType === 'md'
                  ? 'bg-primary-600 text-white font-medium shadow-sm'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              MD Map
            </button>
          </div>

          {isCollapsible && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
              title={isExpanded ? 'Collapse slices' : 'Expand slices'}
            >
              <svg
                className={`w-4 h-4 transform transition-transform ${
                  isExpanded ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-3 p-3 rounded-lg bg-red-950/40 border border-red-800 text-red-300 text-xs">
          {error}
        </div>
      )}

      {isExpanded && (
        <div className="mt-3 space-y-3">
          {/* Slices 3-Up Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* 1. Axial View (XY) */}
            <div className="relative bg-black rounded-xl border border-neutral-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-neutral-950/90 text-xs border-b border-neutral-800 z-10">
                <span className="font-bold text-white font-mono text-[11px]">
                  Axial (Transverse)
                </span>
                <span className="text-[10px] text-primary-400 font-mono">
                  Z = {currentCoord.z} / {dims[2] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-black cursor-crosshair overflow-hidden select-none"
                onClick={handleAxialClick}
                onMouseMove={handleAxialHover}
                onMouseLeave={handleHoverLeave}
              >
                {slicesData?.slices.axial.image ? (
                  <img
                    src={getImageUrl(slicesData.slices.axial.image)}
                    alt="Axial slice"
                    className="w-full h-full object-contain pointer-events-none"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-neutral-600">
                    {loading ? 'Extracting slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: X line (vertical), Y line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/80 pointer-events-none"
                  style={{ left: `${(currentCoord.x / (dims[0] - 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/80 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.y / (dims[1] - 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  A
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  P
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  R
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  L
                </span>
              </div>

              {/* Slider for Axial Z */}
              <div className="p-2 bg-neutral-950/90 border-t border-neutral-800">
                <input
                  type="range"
                  min="0"
                  max={dims[2] - 1}
                  value={currentCoord.z}
                  onChange={(e) => {
                    const z = parseInt(e.target.value, 10);
                    const next = { ...currentCoord, z };
                    setSliceIndices(next);
                    fetchSlices(next);
                  }}
                  className="w-full accent-primary-500 h-1.5 bg-neutral-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* 2. Coronal View (XZ) */}
            <div className="relative bg-black rounded-xl border border-neutral-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-neutral-950/90 text-xs border-b border-neutral-800 z-10">
                <span className="font-bold text-white font-mono text-[11px]">
                  Coronal (Frontal)
                </span>
                <span className="text-[10px] text-primary-400 font-mono">
                  Y = {currentCoord.y} / {dims[1] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-black cursor-crosshair overflow-hidden select-none"
                onClick={handleCoronalClick}
                onMouseMove={handleCoronalHover}
                onMouseLeave={handleHoverLeave}
              >
                {slicesData?.slices.coronal.image ? (
                  <img
                    src={getImageUrl(slicesData.slices.coronal.image)}
                    alt="Coronal slice"
                    className="w-full h-full object-contain pointer-events-none"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-neutral-600">
                    {loading ? 'Extracting slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: X line (vertical), Z line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/80 pointer-events-none"
                  style={{ left: `${(currentCoord.x / (dims[0] - 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/80 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.z / (dims[2] - 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  S
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  I
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  R
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  L
                </span>
              </div>

              {/* Slider for Coronal Y */}
              <div className="p-2 bg-neutral-950/90 border-t border-neutral-800">
                <input
                  type="range"
                  min="0"
                  max={dims[1] - 1}
                  value={currentCoord.y}
                  onChange={(e) => {
                    const y = parseInt(e.target.value, 10);
                    const next = { ...currentCoord, y };
                    setSliceIndices(next);
                    fetchSlices(next);
                  }}
                  className="w-full accent-primary-500 h-1.5 bg-neutral-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* 3. Sagittal View (YZ) */}
            <div className="relative bg-black rounded-xl border border-neutral-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-neutral-950/90 text-xs border-b border-neutral-800 z-10">
                <span className="font-bold text-white font-mono text-[11px]">
                  Sagittal (Lateral)
                </span>
                <span className="text-[10px] text-primary-400 font-mono">
                  X = {currentCoord.x} / {dims[0] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-black cursor-crosshair overflow-hidden select-none"
                onClick={handleSagittalClick}
                onMouseMove={handleSagittalHover}
                onMouseLeave={handleHoverLeave}
              >
                {slicesData?.slices.sagittal.image ? (
                  <img
                    src={getImageUrl(slicesData.slices.sagittal.image)}
                    alt="Sagittal slice"
                    className="w-full h-full object-contain pointer-events-none"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-neutral-600">
                    {loading ? 'Extracting slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: Y line (vertical), Z line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/80 pointer-events-none"
                  style={{ left: `${(currentCoord.y / (dims[1] - 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/80 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.z / (dims[2] - 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  S
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  I
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  A
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-neutral-500 pointer-events-none">
                  P
                </span>
              </div>

              {/* Slider for Sagittal X */}
              <div className="p-2 bg-neutral-950/90 border-t border-neutral-800">
                <input
                  type="range"
                  min="0"
                  max={dims[0] - 1}
                  value={currentCoord.x}
                  onChange={(e) => {
                    const x = parseInt(e.target.value, 10);
                    const next = { ...currentCoord, x };
                    setSliceIndices(next);
                    fetchSlices(next);
                  }}
                  className="w-full accent-primary-500 h-1.5 bg-neutral-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Coordinate & Physical Dimension Telemetry Bar */}
          <div className="flex flex-wrap items-center justify-between text-xs font-mono text-neutral-400 bg-neutral-950/80 px-4 py-2 rounded-xl border border-neutral-800/80 gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-white font-semibold">
                Slice: [{currentCoord.x}, {currentCoord.y}, {currentCoord.z}]
              </span>
              <span className="text-neutral-600">|</span>
              <span className="text-primary-300">
                Physical: [{(currentCoord.x * voxelSize[0]).toFixed(1)},{' '}
                {(currentCoord.y * voxelSize[1]).toFixed(1)},{' '}
                {(currentCoord.z * voxelSize[2]).toFixed(1)}] mm
              </span>
              {hoverCoord && (
                <>
                  <span className="text-neutral-600">|</span>
                  <span className="text-cyan-300 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/50">
                    Pointer ({hoverCoord.plane}): [{hoverCoord.x}, {hoverCoord.y}, {hoverCoord.z}] (
                    {(hoverCoord.x * voxelSize[0]).toFixed(1)}, {(hoverCoord.y * voxelSize[1]).toFixed(1)}, {(hoverCoord.z * voxelSize[2]).toFixed(1)} mm)
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center gap-3 text-[11px] text-neutral-500">
              <span>Dim: {dims.join(' × ')}</span>
              <span>•</span>
              <span>Vox: {voxelSize.map((v) => v.toFixed(2)).join(' × ')} mm</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
