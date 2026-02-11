'use client';

import { useMemo } from 'react';
import * as THREE from 'three';
import { useAppStore } from '@/lib/store';

export default function SliceViewer() {
  const { volumes, viewerSettings } = useAppStore();
  const faVolume = volumes['fa'];

  // Create slice textures
  const slices = useMemo(() => {
    if (!faVolume) return null;

    const { data, dimensions, voxelSize, dataRange } = faVolume;
    const [nx, ny, nz] = dimensions;
    const [dx, dy, dz] = voxelSize;

    // Normalize data to 0-255 for texture
    const normalizeValue = (value: number) => {
      const normalized = (value - dataRange[0]) / (dataRange[1] - dataRange[0]);
      return Math.floor(normalized * 255);
    };

    // Axial slice (XY plane)
    const axialIndex = Math.floor(nz * viewerSettings.slicePosition.axial);
    const axialData = new Uint8Array(nx * ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + axialIndex * nx * ny;
        axialData[x + y * nx] = normalizeValue(data[idx]);
      }
    }
    const axialTexture = new THREE.DataTexture(
      axialData,
      nx,
      ny,
      THREE.RedFormat,
      THREE.UnsignedByteType
    );
    axialTexture.needsUpdate = true;

    // Coronal slice (XZ plane)
    const coronalIndex = Math.floor(ny * viewerSettings.slicePosition.coronal);
    const coronalData = new Uint8Array(nx * nz);
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + coronalIndex * nx + z * nx * ny;
        coronalData[x + z * nx] = normalizeValue(data[idx]);
      }
    }
    const coronalTexture = new THREE.DataTexture(
      coronalData,
      nx,
      nz,
      THREE.RedFormat,
      THREE.UnsignedByteType
    );
    coronalTexture.needsUpdate = true;

    // Sagittal slice (YZ plane)
    const sagittalIndex = Math.floor(nx * viewerSettings.slicePosition.sagittal);
    const sagittalData = new Uint8Array(ny * nz);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const idx = sagittalIndex + y * nx + z * nx * ny;
        sagittalData[y + z * ny] = normalizeValue(data[idx]);
      }
    }
    const sagittalTexture = new THREE.DataTexture(
      sagittalData,
      ny,
      nz,
      THREE.RedFormat,
      THREE.UnsignedByteType
    );
    sagittalTexture.needsUpdate = true;

    return {
      axial: {
        texture: axialTexture,
        position: [0, 0, (axialIndex - nz / 2) * dz] as [number, number, number],
        size: [nx * dx, ny * dy] as [number, number],
        rotation: [0, 0, 0] as [number, number, number],
      },
      coronal: {
        texture: coronalTexture,
        position: [0, (coronalIndex - ny / 2) * dy, 0] as [number, number, number],
        size: [nx * dx, nz * dz] as [number, number],
        rotation: [Math.PI / 2, 0, 0] as [number, number, number],
      },
      sagittal: {
        texture: sagittalTexture,
        position: [(sagittalIndex - nx / 2) * dx, 0, 0] as [number, number, number],
        size: [ny * dy, nz * dz] as [number, number],
        rotation: [0, Math.PI / 2, 0] as [number, number, number],
      },
    };
  }, [faVolume, viewerSettings.slicePosition]);

  if (!slices) return null;

  return (
    <group>
      {/* Axial */}
      <mesh position={slices.axial.position} rotation={slices.axial.rotation}>
        <planeGeometry args={slices.axial.size} />
        <meshBasicMaterial map={slices.axial.texture} transparent opacity={0.8} />
      </mesh>

      {/* Coronal */}
      <mesh position={slices.coronal.position} rotation={slices.coronal.rotation}>
        <planeGeometry args={slices.coronal.size} />
        <meshBasicMaterial map={slices.coronal.texture} transparent opacity={0.8} />
      </mesh>

      {/* Sagittal */}
      <mesh position={slices.sagittal.position} rotation={slices.sagittal.rotation}>
        <planeGeometry args={slices.sagittal.size} />
        <meshBasicMaterial map={slices.sagittal.texture} transparent opacity={0.8} />
      </mesh>
    </group>
  );
}
