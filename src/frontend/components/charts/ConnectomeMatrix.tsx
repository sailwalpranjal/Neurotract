'use client';

import React, { useState, useMemo, useRef } from 'react';
import { useAppStore } from '@/lib/store';
import { ParcellationLabel, ConnectomeEdge } from '@/lib/types';

interface ConnectomeMatrixProps {
  matrix: number[][];
  labels?: ParcellationLabel[];
}

export default function ConnectomeMatrix({ matrix, labels }: ConnectomeMatrixProps) {
  const {
    hoveredEdge,
    selectedEdge,
    hoveredRegion,
    selectedRegion,
    connectomeThreshold,
    setHoveredEdge,
    setSelectedEdge,
    setHoveredRegion,
    setSelectedRegion,
    setConnectomeThreshold,
    setSliceIndices,
  } = useAppStore();

  const [useLogScale, setUseLogScale] = useState(false);
  const [filterQuery, setFilterQuery] = useState('');
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    val: number;
    x: number;
    y: number;
  } | null>(null);

  const numNodes = matrix.length;
  const nodeLabels: ParcellationLabel[] = useMemo(() => {
    if (labels && labels.length >= numNodes) {
      return labels.slice(0, numNodes);
    }
    return Array.from({ length: numNodes }, (_, i) => ({
      index: i,
      generic_name: `parcel_${i}`,
      anatomical_name: `Region ${i}`,
      name: `Region ${i}`,
      abbreviation: `P${i}`,
      hemisphere: i % 2 === 0 ? 'left' : 'right',
      lobe: 'cortical',
      description: '',
      centroid: undefined,
    }));
  }, [labels, numNodes]);

  // Compute maximum weight in matrix
  const maxVal = useMemo(() => {
    let m = 0;
    for (let i = 0; i < numNodes; i++) {
      for (let j = 0; j < numNodes; j++) {
        if (matrix[i][j] > m) m = matrix[i][j];
      }
    }
    return m;
  }, [matrix, numNodes]);

  // Network stats at current threshold
  const stats = useMemo(() => {
    let edgesCount = 0;
    let totalWeight = 0;
    const degrees = new Array(numNodes).fill(0);
    const strengths = new Array(numNodes).fill(0);

    for (let i = 0; i < numNodes; i++) {
      for (let j = i + 1; j < numNodes; j++) {
        const w = matrix[i][j];
        if (w >= connectomeThreshold) {
          edgesCount++;
          totalWeight += w;
          degrees[i]++;
          degrees[j]++;
          strengths[i] += w;
          strengths[j] += w;
        }
      }
    }

    const maxPossibleEdges = (numNodes * (numNodes - 1)) / 2;
    const density = maxPossibleEdges > 0 ? edgesCount / maxPossibleEdges : 0;

    // Top 5 hub regions
    const hubs = nodeLabels
      .map((label, idx) => ({
        index: idx,
        name: label.anatomical_name || label.generic_name,
        abbr: label.abbreviation || `P${idx}`,
        degree: degrees[idx],
        strength: strengths[idx],
        lobe: label.lobe,
      }))
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 6);

    return { edgesCount, totalWeight, density, hubs };
  }, [matrix, numNodes, connectomeThreshold, nodeLabels]);

  // Handle cell click -> sync edge and 2D slice
  const handleCellClick = (row: number, col: number, val: number) => {
    const edgeData: ConnectomeEdge = {
      source: row,
      target: col,
      weight: val,
      sourceName: nodeLabels[row]?.anatomical_name || `P${row}`,
      targetName: nodeLabels[col]?.anatomical_name || `P${col}`,
    };
    setSelectedEdge(edgeData);
    setSelectedRegion(row);

    // Jump 2D slices to source parcel centroid if available
    const src = nodeLabels[row];
    if (src && src.centroid) {
      const voxX = Math.round((src.centroid[0] - (-80.0)) / 2.0);
      const voxY = Math.round((src.centroid[1] - (-120.0)) / 2.0);
      const voxZ = Math.round((src.centroid[2] - (-60.0)) / 2.0);
      setSliceIndices({
        x: Math.max(0, Math.min(80, voxX)),
        y: Math.max(0, Math.min(105, voxY)),
        z: Math.max(0, Math.min(75, voxZ)),
      });
    }
  };

  // Color mapping function
  const getCellColor = (val: number) => {
    if (val < connectomeThreshold || val === 0) {
      return '#0f172a'; // Deep background
    }
    const ratio = useLogScale
      ? Math.min(1.0, Math.log10(val + 1) / Math.log10(maxVal + 1))
      : Math.min(1.0, val / (maxVal * 0.5));

    // Cyan to Emerald to Gold gradient
    if (ratio < 0.33) {
      const t = ratio / 0.33;
      return `rgb(${Math.round(14 + t * 14)}, ${Math.round(116 + t * 50)}, ${Math.round(144 + t * 50)})`;
    } else if (ratio < 0.66) {
      const t = (ratio - 0.33) / 0.33;
      return `rgb(${Math.round(28 + t * 200)}, ${Math.round(166 + t * 20)}, ${Math.round(194 - t * 100)})`;
    } else {
      const t = (ratio - 0.66) / 0.34;
      return `rgb(${Math.round(228 + t * 27)}, ${Math.round(186 + t * 30)}, ${Math.round(94 - t * 40)})`;
    }
  };

  return (
    <div className="space-y-4">
      {/* Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-neutral-900/90 rounded-xl border border-neutral-800 text-xs">
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={useLogScale}
              onChange={(e) => setUseLogScale(e.target.checked)}
              className="rounded bg-neutral-800 border-neutral-700 text-cyan-500 focus:ring-0"
            />
            <span className="font-mono">Log₁₀ Scaling</span>
          </label>

          <div className="flex items-center gap-2">
            <span className="text-gray-400 font-mono">Threshold (τ):</span>
            <input
              type="range"
              min="0"
              max="25"
              step="1"
              value={connectomeThreshold}
              onChange={(e) => setConnectomeThreshold(parseFloat(e.target.value))}
              className="w-24 accent-cyan-500"
            />
            <span className="text-cyan-400 font-mono font-bold w-7 text-right">
              {connectomeThreshold}
            </span>
          </div>
        </div>

        {/* Live Network Statistics */}
        <div className="flex items-center gap-4 text-neutral-300 font-mono text-[11px]">
          <div>
            Active Edges: <strong className="text-white">{stats.edgesCount}</strong>
          </div>
          <div>
            Density: <strong className="text-cyan-400">{(stats.density * 100).toFixed(2)}%</strong>
          </div>
          <div>
            Total Streamlines: <strong className="text-emerald-400">{stats.totalWeight.toLocaleString()}</strong>
          </div>
        </div>
      </div>

      {/* Synchronized Hover Detail Banner */}
      <div className="p-3 rounded-lg bg-neutral-950/80 border border-neutral-800 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-neutral-400 font-mono">Focused Connection:</span>
          {hoveredEdge || selectedEdge ? (
            <span className="text-white font-mono font-semibold">
              {(hoveredEdge || selectedEdge)?.sourceName} ⇄ {(hoveredEdge || selectedEdge)?.targetName}
            </span>
          ) : (
            <span className="text-neutral-500 italic">Hover any cell or 3D node to inspect</span>
          )}
        </div>
        {(hoveredEdge || selectedEdge) && (
          <div className="flex items-center gap-3 font-mono text-[11px]">
            <span className="text-neutral-400">Weight:</span>
            <span className="text-emerald-400 font-bold px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40">
              {(hoveredEdge || selectedEdge)?.weight} streamlines
            </span>
          </div>
        )}
      </div>

      {/* Interactive Heatmap Matrix Grid */}
      <div className="relative overflow-x-auto border border-neutral-800 rounded-xl bg-neutral-950 p-2">
        <div
          className="grid gap-[1px] min-w-[700px] select-none"
          style={{
            gridTemplateColumns: `repeat(${numNodes}, minmax(0, 1fr))`,
          }}
          onMouseLeave={() => {
            setHoveredCell(null);
            setHoveredEdge(null);
          }}
        >
          {matrix.map((row, rIdx) =>
            row.map((val, cIdx) => {
              const isSelected =
                selectedEdge &&
                ((selectedEdge.source === rIdx && selectedEdge.target === cIdx) ||
                  (selectedEdge.source === cIdx && selectedEdge.target === rIdx));
              const isHovered =
                hoveredCell && (hoveredCell.row === rIdx || hoveredCell.col === cIdx);
              const isRegionHighlighted =
                selectedRegion === rIdx || selectedRegion === cIdx;

              const bg = getCellColor(val);

              return (
                <div
                  key={`${rIdx}-${cIdx}`}
                  onClick={() => handleCellClick(rIdx, cIdx, val)}
                  onMouseEnter={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setHoveredCell({ row: rIdx, col: cIdx, val, x: rect.left, y: rect.top });
                    if (val >= connectomeThreshold) {
                      setHoveredEdge({
                        source: rIdx,
                        target: cIdx,
                        weight: val,
                        sourceName: nodeLabels[rIdx]?.anatomical_name || `P${rIdx}`,
                        targetName: nodeLabels[cIdx]?.anatomical_name || `P${cIdx}`,
                      });
                    }
                  }}
                  className={`aspect-square transition-all cursor-pointer ${
                    isSelected
                      ? 'ring-2 ring-cyan-400 z-10 scale-125'
                      : isRegionHighlighted
                      ? 'brightness-125 ring-1 ring-amber-400/60'
                      : isHovered
                      ? 'brightness-150'
                      : 'hover:brightness-125'
                  }`}
                  style={{ backgroundColor: bg }}
                  title={`${nodeLabels[rIdx]?.abbreviation || rIdx} ⇄ ${nodeLabels[cIdx]?.abbreviation || cIdx}: ${val} streamlines`}
                />
              );
            })
          )}
        </div>
      </div>

      {/* Hub Regions Ranking (Top Connected Brain Parcels) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold uppercase tracking-wider text-neutral-400">
            Top Connected Anatomical Hubs (Degree &amp; Streamline Strength)
          </h4>
          <span className="text-[10px] text-neutral-500 font-mono">Click hub to center 3D &amp; MPR</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          {stats.hubs.map((hub) => {
            const isSelected = selectedRegion === hub.index;
            return (
              <button
                key={hub.index}
                onClick={() => {
                  setSelectedRegion(hub.index);
                  const p = nodeLabels[hub.index];
                  if (p && p.centroid) {
                    const voxX = Math.round((p.centroid[0] - (-80.0)) / 2.0);
                    const voxY = Math.round((p.centroid[1] - (-120.0)) / 2.0);
                    const voxZ = Math.round((p.centroid[2] - (-60.0)) / 2.0);
                    setSliceIndices({
                      x: Math.max(0, Math.min(80, voxX)),
                      y: Math.max(0, Math.min(105, voxY)),
                      z: Math.max(0, Math.min(75, voxZ)),
                    });
                  }
                }}
                className={`p-2.5 rounded-lg border text-left transition-all font-mono ${
                  isSelected
                    ? 'bg-cyan-950/80 border-cyan-500/80 ring-1 ring-cyan-400'
                    : 'bg-neutral-900/80 border-neutral-800 hover:border-neutral-700 hover:bg-neutral-800/60'
                }`}
              >
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-cyan-400 font-bold">{hub.abbr}</span>
                  <span className="text-neutral-400">{hub.degree} deg</span>
                </div>
                <div className="text-[11px] font-semibold text-white truncate mt-1">
                  {hub.name}
                </div>
                <div className="text-[10px] text-emerald-400 mt-0.5">
                  {hub.strength.toLocaleString()} tracts
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
