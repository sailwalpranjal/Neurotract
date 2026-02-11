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

    geo.computeBoundingSphere();

    return geo;
  }, [mesh]);

  const material = useMemo(() => {
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(settings.brainSurfaceColor),
      transparent: true,
      opacity: settings.brainSurfaceOpacity,
      side: THREE.DoubleSide,
      wireframe: settings.brainSurfaceWireframe,
      roughness: 0.7,
      metalness: 0.0,
      clearcoat: 0.3,
      clearcoatRoughness: 0.4,
      depthWrite: false,
    });
  }, [settings.brainSurfaceColor, settings.brainSurfaceOpacity, settings.brainSurfaceWireframe]);

  return <mesh geometry={geometry} material={material} />;
}
