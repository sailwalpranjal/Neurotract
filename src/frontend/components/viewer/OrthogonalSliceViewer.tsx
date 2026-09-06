'use client';
/* eslint-disable @next/next/no-img-element */

import React, { useEffect, useState, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';

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
  const [hoverCoord, setHoverCoord] = useState<{
    x: number;
    y: number;
    z: number;
    plane: string;
    value?: number;
  } | null>(null);

  // Default coordinate if null
  const currentCoord = sliceIndices || {
    x: slicesData?.indices.sagittal ?? 40,
    y: slicesData?.indices.coronal ?? 53,
    z: slicesData?.indices.axial ?? 38,
  };

  const dims = slicesData?.volume_shape || [81, 106, 76];
  const voxelSize = slicesData?.voxel_size_mm || [2.0, 2.0, 2.0];

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

  const handleVolumeSwitch = (type: 'b0' | 'fa' | 'md') => {
    setActiveSliceType(type);
    fetchSlices(currentCoord, type);
  };

  // Click on Axial plane: (X, Y) clicked -> update X and Y, keep Z
  const handleAxialClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));

    const newX = Math.round(normX * (dims[0] - 1));
    const newY = Math.round((1 - normY) * (dims[1] - 1)); // Invert Y for neurological display

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

    const row = Math.min(Math.round(normY * (dims[1] - 1)), (slicesData?.slices.axial.dims[0] ?? dims[1]) - 1);
    const col = Math.min(Math.round(normX * (dims[0] - 1)), (slicesData?.slices.axial.dims[1] ?? dims[0]) - 1);
    const val = slicesData?.slices.axial.scalar_matrix?.[row]?.[col];

    setHoverCoord({ x: hX, y: hY, z: currentCoord.z, plane: 'Axial', value: val });
  };

  const handleCoronalHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normZ = Math.max(0, Math.min(1, clickY / rect.height));
    const hX = Math.round(normX * (dims[0] - 1));
    const hZ = Math.round((1 - normZ) * (dims[2] - 1));

    const row = Math.min(Math.round(normZ * (dims[2] - 1)), (slicesData?.slices.coronal.dims[0] ?? dims[2]) - 1);
    const col = Math.min(Math.round(normX * (dims[0] - 1)), (slicesData?.slices.coronal.dims[1] ?? dims[0]) - 1);
    const val = slicesData?.slices.coronal.scalar_matrix?.[row]?.[col];

    setHoverCoord({ x: hX, y: currentCoord.y, z: hZ, plane: 'Coronal', value: val });
  };

  const handleSagittalHover = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickY = e.clientX - rect.left;
    const clickZ = e.clientY - rect.top;
    const normY = Math.max(0, Math.min(1, clickY / rect.width));
    const normZ = Math.max(0, Math.min(1, clickZ / rect.height));
    const hY = Math.round(normY * (dims[1] - 1));
    const hZ = Math.round((1 - normZ) * (dims[2] - 1));

    const row = Math.min(Math.round(normZ * (dims[2] - 1)), (slicesData?.slices.sagittal.dims[0] ?? dims[2]) - 1);
    const col = Math.min(Math.round(normY * (dims[1] - 1)), (slicesData?.slices.sagittal.dims[1] ?? dims[1]) - 1);
    const val = slicesData?.slices.sagittal.scalar_matrix?.[row]?.[col];

    setHoverCoord({ x: currentCoord.x, y: hY, z: hZ, plane: 'Sagittal', value: val });
  };

  const handleHoverLeave = () => {
    setHoverCoord(null);
  };

  // Physical coordinates formatted (RAS+)
  const physX = slicesData?.physical_mm?.x ?? Number(((currentCoord.x - dims[0] / 2) * voxelSize[0]).toFixed(1));
  const physY = slicesData?.physical_mm?.y ?? Number(((currentCoord.y - dims[1] / 2) * voxelSize[1]).toFixed(1));
  const physZ = slicesData?.physical_mm?.z ?? Number(((currentCoord.z - dims[2] / 2) * voxelSize[2]).toFixed(1));

  return (
    <div className="bg-slate-900/95 backdrop-blur-md border border-slate-800 rounded-xl p-4 text-slate-100 shadow-2xl transition-all">
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
          </span>
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              Synchronized Multi-Planar Reconstruction (MPR)
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {slicesData?.modality_name || activeSliceType.toUpperCase()}
              </span>
            </h3>
            <p className="text-[11px] text-slate-400">
              Interactive 3-plane orthogonal crosshair navigation in scanner space (RAS+)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Volume Type Select */}
          <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => handleVolumeSwitch('b0')}
              className={`px-3 py-1 rounded-md transition-colors ${
                activeSliceType === 'b0'
                  ? 'bg-emerald-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              B0 Vol
            </button>
            <button
              onClick={() => handleVolumeSwitch('fa')}
              className={`px-3 py-1 rounded-md transition-colors ${
                activeSliceType === 'fa'
                  ? 'bg-emerald-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              FA Map
            </button>
            <button
              onClick={() => handleVolumeSwitch('md')}
              className={`px-3 py-1 rounded-md transition-colors ${
                activeSliceType === 'md'
                  ? 'bg-emerald-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              MD Map
            </button>
          </div>

          {isCollapsible && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
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
            <div className="relative bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 text-xs border-b border-slate-800 z-10">
                <span className="font-semibold text-slate-200 font-mono text-[11px]">
                  Axial (Transverse)
                </span>
                <span className="text-[10px] text-emerald-400 font-mono">
                  Z = {currentCoord.z} / {dims[2] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-slate-950 cursor-crosshair overflow-hidden select-none"
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
                  <div className="w-full h-full flex items-center justify-center text-xs text-slate-600">
                    {loading ? 'Extracting axial slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: X line (vertical), Y line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/70 pointer-events-none"
                  style={{ left: `${(currentCoord.x / Math.max(dims[0] - 1, 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/70 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.y / Math.max(dims[1] - 1, 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  A
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  P
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  R
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  L
                </span>
              </div>

              {/* Slider for Axial Z */}
              <div className="p-2 bg-slate-900/90 border-t border-slate-800">
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
                  className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* 2. Coronal View (XZ) */}
            <div className="relative bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 text-xs border-b border-slate-800 z-10">
                <span className="font-semibold text-slate-200 font-mono text-[11px]">
                  Coronal (Frontal)
                </span>
                <span className="text-[10px] text-emerald-400 font-mono">
                  Y = {currentCoord.y} / {dims[1] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-slate-950 cursor-crosshair overflow-hidden select-none"
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
                  <div className="w-full h-full flex items-center justify-center text-xs text-slate-600">
                    {loading ? 'Extracting coronal slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: X line (vertical), Z line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/70 pointer-events-none"
                  style={{ left: `${(currentCoord.x / Math.max(dims[0] - 1, 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/70 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.z / Math.max(dims[2] - 1, 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  S
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  I
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  R
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  L
                </span>
              </div>

              {/* Slider for Coronal Y */}
              <div className="p-2 bg-slate-900/90 border-t border-slate-800">
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
                  className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* 3. Sagittal View (YZ) */}
            <div className="relative bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex flex-col group">
              <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 text-xs border-b border-slate-800 z-10">
                <span className="font-semibold text-slate-200 font-mono text-[11px]">
                  Sagittal (Lateral)
                </span>
                <span className="text-[10px] text-emerald-400 font-mono">
                  X = {currentCoord.x} / {dims[0] - 1}
                </span>
              </div>

              <div
                className="relative aspect-square w-full bg-slate-950 cursor-crosshair overflow-hidden select-none"
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
                  <div className="w-full h-full flex items-center justify-center text-xs text-slate-600">
                    {loading ? 'Extracting sagittal slice...' : 'No slice'}
                  </div>
                )}

                {/* Crosshairs: Y line (vertical), Z line (horizontal) */}
                <div
                  className="absolute top-0 bottom-0 border-l border-emerald-400/70 pointer-events-none"
                  style={{ left: `${(currentCoord.y / Math.max(dims[1] - 1, 1)) * 100}%` }}
                />
                <div
                  className="absolute left-0 right-0 border-t border-emerald-400/70 pointer-events-none"
                  style={{ top: `${(1 - currentCoord.z / Math.max(dims[2] - 1, 1)) * 100}%` }}
                />

                {/* Orientation Markers */}
                <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  S
                </span>
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  I
                </span>
                <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  A
                </span>
                <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-500 pointer-events-none">
                  P
                </span>
              </div>

              {/* Slider for Sagittal X */}
              <div className="p-2 bg-slate-900/90 border-t border-slate-800">
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
                  className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Real-time Scientific Telemetry & Inspector Bar */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2 text-xs font-mono bg-slate-950 p-3 rounded-xl border border-slate-800">
            {/* 1. Crosshair Voxel & Physical RAS+ */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Crosshair Position</span>
              <div className="text-slate-200 font-semibold">
                Voxel: [{currentCoord.x}, {currentCoord.y}, {currentCoord.z}]
              </div>
              <div className="text-emerald-400 text-[11px]">
                RAS+: [{physX}, {physY}, {physZ}] mm
              </div>
            </div>

            {/* 2. Scalar Intensity Value */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block">
                {slicesData?.modality_name || 'Scalar Intensity'}
              </span>
              <div className="text-slate-200 font-semibold">
                {slicesData?.current_voxel_value !== undefined
                  ? `${slicesData.current_voxel_value} ${slicesData.unit || ''}`
                  : 'N/A'}
              </div>
              <div className="text-slate-500 text-[11px]">
                Window: [{slicesData?.intensity_range[0]?.toFixed(2) ?? '0.00'},{' '}
                {slicesData?.intensity_range[1]?.toFixed(2) ?? '1.00'}]
              </div>
            </div>

            {/* 3. Anatomical Parcellation */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Anatomical Region</span>
              <div className="text-cyan-300 font-medium truncate" title={slicesData?.parcellation?.name}>
                {slicesData?.parcellation?.name || 'Subcortical White Matter'}
              </div>
              <div className="text-slate-500 text-[11px]">
                {slicesData?.parcellation?.lobe && slicesData.parcellation.lobe !== 'none'
                  ? `${slicesData.parcellation.lobe.toUpperCase()} • ${slicesData.parcellation.hemisphere || 'bilateral'}`
                  : 'Deep Parenchyma'}
              </div>
            </div>

            {/* 4. Live Hover Cursor Telemetry */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Pointer Inspector</span>
              {hoverCoord ? (
                <div>
                  <div className="text-amber-300 font-semibold">
                    {hoverCoord.plane}: [{hoverCoord.x}, {hoverCoord.y}, {hoverCoord.z}]
                  </div>
                  <div className="text-amber-200 text-[11px]">
                    Val: {hoverCoord.value !== undefined ? `${hoverCoord.value} ${slicesData?.unit || ''}` : '---'}
                  </div>
                </div>
              ) : (
                <div className="text-slate-600 text-[11px] pt-1">
                  Hover over slice to probe voxels
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

