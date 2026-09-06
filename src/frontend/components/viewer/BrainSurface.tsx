'use client';

import { useMemo } from 'react';
import * as THREE from 'three';
import { BrainMeshData, ViewerSettings } from '@/lib/types';

interface BrainSurfaceProps {
  mesh: BrainMeshData;
  settings: ViewerSettings;
}

export default function BrainSurface({ mesh, settings }: BrainSurfaceProps) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();

    // Create typed arrays from flat data
    const vertices = new Float32Array(mesh.vertices);
    const normals = new Float32Array(mesh.normals);
    const indices = new Uint32Array(mesh.faces);

    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // Ensure smooth continuous vertex normals for gyral curvature
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    geo.computeBoundingBox();

    return geo;
  }, [mesh]);

  const material = useMemo(() => {
    const baseColor = new THREE.Color(settings.brainSurfaceColor || '#cbd5e1');
    const opacity = settings.brainSurfaceOpacity ?? 0.35;
    const isSolid = opacity >= 0.95 && !settings.brainSurfaceWireframe;

    return new THREE.MeshPhysicalMaterial({
      color: baseColor,
      transparent: !isSolid,
      opacity: Math.max(0.05, opacity),
      side: THREE.DoubleSide,
      wireframe: settings.brainSurfaceWireframe || false,
      roughness: settings.brainRoughness ?? 0.35,
      metalness: settings.brainMetalness ?? 0.1,
      clearcoat: isSolid ? 0.2 : 0.85,
      clearcoatRoughness: 0.25,
      reflectivity: 0.5,
      depthWrite: isSolid,
    });
  }, [settings.brainSurfaceColor, settings.brainSurfaceOpacity, settings.brainSurfaceWireframe, settings.brainRoughness, settings.brainMetalness]);

  return (
    <group>
      <mesh geometry={geometry} material={material} receiveShadow castShadow />
    </group>
  );
}
