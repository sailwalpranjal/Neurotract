'use client';

import { useMemo } from 'react';
import * as THREE from 'three';
import { StreamlineBundle, ViewerSettings } from '@/lib/types';

interface StreamlineRendererProps {
  bundle: StreamlineBundle;
  settings: ViewerSettings;
}

export default function StreamlineRenderer({
  bundle,
  settings,
}: StreamlineRendererProps) {
  // Batch all streamlines into a single LineSegments BufferGeometry (1 draw call)
  const geometry = useMemo(() => {
    if (!bundle?.streamlines || bundle.streamlines.length === 0) {
      return null;
    }

    // Count total segments
    let totalSegments = 0;
    for (let s = 0; s < bundle.streamlines.length; s++) {
      const nPts = bundle.streamlines[s].numPoints;
      if (nPts > 1) {
        totalSegments += (nPts - 1);
      }
    }

    if (totalSegments === 0) return null;

    const positions = new Float32Array(totalSegments * 2 * 3);
    const colors = new Float32Array(totalSegments * 2 * 3);

    let posIdx = 0;
    let colIdx = 0;

    for (let s = 0; s < bundle.streamlines.length; s++) {
      const sl = bundle.streamlines[s];
      const pts = sl.points;
      const nPts = sl.numPoints;
      if (nPts < 2) continue;

      for (let i = 0; i < nPts - 1; i++) {
        const x1 = pts[i * 3];
        const y1 = pts[i * 3 + 1];
        const z1 = pts[i * 3 + 2];

        const x2 = pts[(i + 1) * 3];
        const y2 = pts[(i + 1) * 3 + 1];
        const z2 = pts[(i + 1) * 3 + 2];

        // Segment start
        positions[posIdx++] = x1;
        positions[posIdx++] = y1;
        positions[posIdx++] = z1;

        // Segment end
        positions[posIdx++] = x2;
        positions[posIdx++] = y2;
        positions[posIdx++] = z2;

        // Directional RGB (tangent of this segment)
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dz = z2 - z1;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;

        // Canonical diffusion-MRI color mapping:
        // Red = Left-Right (X), Green = Anterior-Posterior (Y), Blue = Superior-Inferior (Z)
        const r = Math.min(1.0, Math.max(0.05, Math.abs(dx / len)));
        const g = Math.min(1.0, Math.max(0.05, Math.abs(dy / len)));
        const b = Math.min(1.0, Math.max(0.05, Math.abs(dz / len)));

        // Both segment vertices receive the segment directional color
        colors[colIdx++] = r;
        colors[colIdx++] = g;
        colors[colIdx++] = b;

        colors[colIdx++] = r;
        colors[colIdx++] = g;
        colors[colIdx++] = b;
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeBoundingSphere();
    geo.computeBoundingBox();

    return geo;
  }, [bundle]);

  const material = useMemo(() => {
    return new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: Math.max(0.2, settings.streamlineOpacity ?? 0.85),
      depthWrite: false,
    });
  }, [settings.streamlineOpacity]);

  if (!geometry) return null;

  return (
    <group>
      <lineSegments geometry={geometry} material={material} />
    </group>
  );
}
