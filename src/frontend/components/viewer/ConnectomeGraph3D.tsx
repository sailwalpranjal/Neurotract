'use client';

import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { useAppStore } from '@/lib/store';
import { ParcellationLabel } from '@/lib/types';

interface ConnectomeGraph3DProps {
  connectome: number[][];
  labels: ParcellationLabel[];
  threshold?: number;
}

const LOBE_COLORS: Record<string, string> = {
  frontal: '#3b82f6',     // Blue
  temporal: '#10b981',    // Emerald
  parietal: '#f59e0b',    // Amber
  occipital: '#ec4899',   // Pink
  subcortical: '#8b5cf6', // Purple
  none: '#64748b',        // Slate
};

export default function ConnectomeGraph3D({
  connectome,
  labels,
  threshold = 1.0,
}: ConnectomeGraph3DProps) {
  const {
    hoveredEdge,
    selectedEdge,
    hoveredRegion,
    selectedRegion,
    setHoveredEdge,
    setSelectedEdge,
    setHoveredRegion,
    setSelectedRegion,
    setSliceIndices,
  } = useAppStore();

  const groupRef = useRef<THREE.Group>(null);

  // Map label centroids to valid 3D vectors
  const nodePositions = useMemo(() => {
    return labels.map((l, idx) => {
      if (l.centroid && Array.isArray(l.centroid) && l.centroid.length === 3) {
        return new THREE.Vector3(l.centroid[0], l.centroid[1], l.centroid[2]);
      }
      return new THREE.Vector3(0, 0, 0);
    });
  }, [labels]);

  // Extract edges that meet or exceed threshold
  const activeEdges = useMemo(() => {
    if (!connectome || connectome.length === 0) return [];
    const n = Math.min(connectome.length, labels.length);
    const edges: {
      source: number;
      target: number;
      weight: number;
      sourcePos: THREE.Vector3;
      targetPos: THREE.Vector3;
    }[] = [];

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = connectome[i][j];
        if (w >= threshold && nodePositions[i] && nodePositions[j]) {
          edges.push({
            source: i,
            target: j,
            weight: w,
            sourcePos: nodePositions[i],
            targetPos: nodePositions[j],
          });
        }
      }
    }
    return edges;
  }, [connectome, labels, threshold, nodePositions]);

  // Build LineSegments geometry for all active edges
  const { lineGeometry } = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const colorObj = new THREE.Color();

    for (const edge of activeEdges) {
      positions.push(
        edge.sourcePos.x, edge.sourcePos.y, edge.sourcePos.z,
        edge.targetPos.x, edge.targetPos.y, edge.targetPos.z
      );

      // Edge intensity based on log weight
      const alpha = Math.min(1.0, Math.log10(edge.weight + 1) / 3.0);
      colorObj.setHSL(0.55, 0.8, 0.2 + alpha * 0.5);
      colors.push(colorObj.r, colorObj.g, colorObj.b);
      colors.push(colorObj.r, colorObj.g, colorObj.b);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    return { lineGeometry: geo };
  }, [activeEdges]);

  // Handle clicking a node -> sync with 2D slice viewer
  const handleNodeClick = (nodeIdx: number, e: any) => {
    e.stopPropagation();
    setSelectedRegion(selectedRegion === nodeIdx ? null : nodeIdx);

    const pos = nodePositions[nodeIdx];
    if (pos) {
      // Stanford dataset affine: aff[0,3]=-80, aff[1,3]=-120, aff[2,3]=-60, voxel=2.0mm
      const voxX = Math.round((pos.x - (-80.0)) / 2.0);
      const voxY = Math.round((pos.y - (-120.0)) / 2.0);
      const voxZ = Math.round((pos.z - (-60.0)) / 2.0);
      setSliceIndices({
        x: Math.max(0, Math.min(80, voxX)),
        y: Math.max(0, Math.min(105, voxY)),
        z: Math.max(0, Math.min(75, voxZ)),
      });
    }
  };

  // Determine highlighted edge (hovered or selected)
  const activeFocusEdge = hoveredEdge || selectedEdge;

  return (
    <group ref={groupRef}>
      {/* 3D Connectome Edges */}
      {activeEdges.length > 0 && (
        <lineSegments geometry={lineGeometry}>
          <lineBasicMaterial
            vertexColors
            transparent
            opacity={0.35}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </lineSegments>
      )}

      {/* Highlighted Edge (when hovered or selected) */}
      {activeFocusEdge && nodePositions[activeFocusEdge.source] && nodePositions[activeFocusEdge.target] && (
        <group>
          {/* Label tooltip at midpoint */}
          <Html
            position={[
              (nodePositions[activeFocusEdge.source].x + nodePositions[activeFocusEdge.target].x) / 2,
              (nodePositions[activeFocusEdge.source].y + nodePositions[activeFocusEdge.target].y) / 2 + 5,
              (nodePositions[activeFocusEdge.source].z + nodePositions[activeFocusEdge.target].z) / 2,
            ]}
            center
            distanceFactor={180}
          >
            <div className="bg-neutral-950/90 text-cyan-300 px-3 py-1.5 rounded-lg border border-cyan-500/50 text-[11px] font-mono shadow-2xl backdrop-blur-md pointer-events-none whitespace-nowrap">
              <strong>{activeFocusEdge.sourceName || `P${activeFocusEdge.source}`}</strong>
              <span className="text-neutral-400 mx-1.5">⇄</span>
              <strong>{activeFocusEdge.targetName || `P${activeFocusEdge.target}`}</strong>
              <div className="text-[10px] text-neutral-400 mt-0.5">
                Weight: <span className="text-emerald-400 font-bold">{activeFocusEdge.weight}</span> streamlines
              </div>
            </div>
          </Html>
        </group>
      )}

      {/* 3D Parcellation Nodes */}
      {labels.map((label, idx) => {
        const pos = nodePositions[idx];
        if (!pos || (pos.x === 0 && pos.y === 0 && pos.z === 0)) return null;

        const isHovered = hoveredRegion === idx;
        const isSelected = selectedRegion === idx;
        const isEdgeConnected =
          activeFocusEdge && (activeFocusEdge.source === idx || activeFocusEdge.target === idx);

        const nodeColor = isSelected
          ? '#38bdf8'
          : isHovered || isEdgeConnected
          ? '#fbbf24'
          : LOBE_COLORS[label.lobe?.toLowerCase()] || '#60a5fa';

        const scale = isSelected || isEdgeConnected ? 2.4 : isHovered ? 2.0 : 1.2;

        return (
          <mesh
            key={label.index ?? idx}
            position={[pos.x, pos.y, pos.z]}
            scale={[scale, scale, scale]}
            onClick={(e) => handleNodeClick(idx, e)}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredRegion(idx);
            }}
            onPointerOut={() => setHoveredRegion(null)}
          >
            <sphereGeometry args={[1.5, 16, 16]} />
            <meshStandardMaterial
              color={nodeColor}
              emissive={nodeColor}
              emissiveIntensity={isSelected || isEdgeConnected ? 0.8 : isHovered ? 0.5 : 0.2}
              roughness={0.3}
              metalness={0.2}
            />

            {(isHovered || isSelected || isEdgeConnected) && (
              <Html distanceFactor={160} center position={[0, 4, 0]}>
                <div className="bg-neutral-900/95 text-white px-2 py-1 rounded border border-neutral-700 text-[10px] font-mono shadow-xl whitespace-nowrap pointer-events-none">
                  <span className="text-cyan-400 font-bold">{label.abbreviation || `P${idx}`}</span>:{' '}
                  {label.anatomical_name || label.generic_name}
                </div>
              </Html>
            )}
          </mesh>
        );
      })}
    </group>
  );
}
