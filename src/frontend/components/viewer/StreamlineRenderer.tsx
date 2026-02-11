'use client';

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { StreamlineBundle, ViewerSettings } from '@/lib/types';
import {
  getColorFromMap,
  calculateStreamlineLength,
  calculateMeanOrientation,
  Colormap,
} from '@/lib/utils';

interface StreamlineRendererProps {
  bundle: StreamlineBundle;
  settings: ViewerSettings;
}

export default function StreamlineRenderer({
  bundle,
  settings,
}: StreamlineRendererProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Generate geometry for all streamlines
  const { geometries, colors } = useMemo(() => {
    const geometries: THREE.BufferGeometry[] = [];
    const colors: THREE.Color[] = [];

    const colorMapping = settings.colorMapping;
    let valueMin = Infinity;
    let valueMax = -Infinity;

    // First pass: calculate min/max for normalization
    bundle.streamlines.forEach((streamline) => {
      let value = 0;

      switch (colorMapping.type) {
        case 'length':
          value = streamline.length || calculateStreamlineLength(streamline.points);
          break;
        case 'fa':
          value = streamline.fa || 0;
          break;
        case 'orientation':
          // Will handle separately
          break;
      }

      if (colorMapping.type !== 'orientation') {
        valueMin = Math.min(valueMin, value);
        valueMax = Math.max(valueMax, value);
      }
    });

    // Use custom range if provided
    if (colorMapping.range) {
      [valueMin, valueMax] = colorMapping.range;
    }

    // Second pass: create geometries and colors
    bundle.streamlines.forEach((streamline) => {
      const points = streamline.points;
      const numPoints = streamline.numPoints;

      // Create line geometry
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(numPoints * 3);

      for (let i = 0; i < numPoints; i++) {
        positions[i * 3] = points[i * 3];
        positions[i * 3 + 1] = points[i * 3 + 1];
        positions[i * 3 + 2] = points[i * 3 + 2];
      }

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      // Determine color
      let color: [number, number, number];

      switch (colorMapping.type) {
        case 'length': {
          const length = streamline.length || calculateStreamlineLength(points);
          color = getColorFromMap(length, valueMin, valueMax, colorMapping.colormap as Colormap);
          break;
        }
        case 'fa': {
          const fa = streamline.fa || 0;
          color = getColorFromMap(fa, valueMin, valueMax, colorMapping.colormap as Colormap);
          break;
        }
        case 'orientation': {
          const orientation = streamline.orientation || calculateMeanOrientation(points);
          // RGB encoding: abs(x), abs(y), abs(z)
          color = [
            Math.abs(orientation[0]),
            Math.abs(orientation[1]),
            Math.abs(orientation[2]),
          ];
          break;
        }
        case 'custom': {
          color = streamline.color || [1, 1, 1];
          break;
        }
        default:
          color = [1, 1, 1];
      }

      geometries.push(geometry);
      colors.push(new THREE.Color(color[0], color[1], color[2]));
    });

    return { geometries, colors };
  }, [bundle, settings.colorMapping]);

  // Level of Detail - reduce streamlines based on camera distance
  useFrame(({ camera }) => {
    if (!groupRef.current) return;

    const distance = camera.position.length();
    let visibleCount = geometries.length;

    switch (settings.levelOfDetail) {
      case 'low':
        visibleCount = Math.floor(geometries.length * 0.2);
        break;
      case 'medium':
        if (distance > 300) {
          visibleCount = Math.floor(geometries.length * 0.5);
        } else if (distance > 150) {
          visibleCount = Math.floor(geometries.length * 0.75);
        }
        break;
      case 'high':
        // Show all
        break;
    }

    // Update visibility
    groupRef.current.children.forEach((child, index) => {
      child.visible = index < visibleCount;
    });
  });

  const lineObjects = useMemo(() => {
    return geometries.map((geometry, index) => {
      const material = new THREE.LineBasicMaterial({
        color: colors[index],
        opacity: settings.streamlineOpacity,
        transparent: settings.streamlineOpacity < 1,
        linewidth: settings.streamlineWidth,
      });
      return new THREE.Line(geometry, material);
    });
  }, [geometries, colors, settings.streamlineOpacity, settings.streamlineWidth]);

  return (
    <group ref={groupRef}>
      {lineObjects.map((lineObj, index) => (
        <primitive key={index} object={lineObj} />
      ))}
    </group>
  );
}
