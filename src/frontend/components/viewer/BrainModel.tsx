'use client';

import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { ViewerSettings, BrainModelType } from '@/lib/types';

const MODEL_PATHS: Record<Exclude<BrainModelType, 'marching_cubes'>, string> = {
  hologram: '/models/brain_hologram.glb',
  point_cloud: '/models/brain_point_cloud.glb',
};

interface BrainModelProps {
  modelType: Exclude<BrainModelType, 'marching_cubes'>;
  settings: ViewerSettings;
  streamlineBounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

export default function BrainModel({ modelType, settings, streamlineBounds }: BrainModelProps) {
  const groupRef = useRef<THREE.Group>(null);
  const { scene } = useGLTF(MODEL_PATHS[modelType]);

  // Clone and align the model to streamline coordinate space
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);

    // Compute the GLB model's bounding box
    const modelBox = new THREE.Box3().setFromObject(clone);
    const modelCenter = modelBox.getCenter(new THREE.Vector3());
    const modelSize = modelBox.getSize(new THREE.Vector3());

    if (streamlineBounds) {
      // Align GLB model to match the streamline data bounds
      const targetMin = new THREE.Vector3(...streamlineBounds.min);
      const targetMax = new THREE.Vector3(...streamlineBounds.max);
      const targetSize = new THREE.Vector3().subVectors(targetMax, targetMin);
      const targetCenter = new THREE.Vector3().addVectors(targetMin, targetMax).multiplyScalar(0.5);

      // Scale model to fit inside streamline bounds (use the smallest ratio to avoid overflow)
      // Apply slight shrink (0.85) so the brain surface sits inside the streamlines
      const scaleX = modelSize.x > 0 ? (targetSize.x / modelSize.x) : 1;
      const scaleY = modelSize.y > 0 ? (targetSize.y / modelSize.y) : 1;
      const scaleZ = modelSize.z > 0 ? (targetSize.z / modelSize.z) : 1;
      const uniformScale = Math.min(scaleX, scaleY, scaleZ) * 0.85;

      clone.scale.setScalar(uniformScale);

      // Position: move model center to target center
      clone.position.set(
        targetCenter.x - modelCenter.x * uniformScale,
        targetCenter.y - modelCenter.y * uniformScale,
        targetCenter.z - modelCenter.z * uniformScale
      );
    } else {
      // Fallback: center at origin with reasonable scale
      const maxDim = Math.max(modelSize.x, modelSize.y, modelSize.z);
      const targetSize = 150;
      const scale = maxDim > 0 ? targetSize / maxDim : 1;
      clone.scale.setScalar(scale);
      clone.position.set(
        -modelCenter.x * scale,
        -modelCenter.y * scale,
        -modelCenter.z * scale
      );
    }

    return clone;
  }, [scene, streamlineBounds]);

  // Update materials when settings change
  useEffect(() => {
    if (!clonedScene) return;

    clonedScene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
          if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
            mat.transparent = true;
            mat.opacity = settings.brainSurfaceOpacity;
            mat.wireframe = settings.brainSurfaceWireframe;
            mat.depthWrite = settings.brainSurfaceOpacity > 0.5;
            mat.side = THREE.DoubleSide;

            if (settings.brainSurfaceColor !== '#e8d5cf') {
              mat.color.set(settings.brainSurfaceColor);
            }
            mat.emissiveIntensity = settings.brainEmissiveIntensity;
            mat.metalness = settings.brainMetalness;
            mat.roughness = settings.brainRoughness;
            mat.needsUpdate = true;
          } else if (mat instanceof THREE.MeshBasicMaterial) {
            mat.transparent = true;
            mat.opacity = settings.brainSurfaceOpacity;
            mat.wireframe = settings.brainSurfaceWireframe;
            mat.depthWrite = settings.brainSurfaceOpacity > 0.5;
            mat.side = THREE.DoubleSide;
            if (settings.brainSurfaceColor !== '#e8d5cf') {
              mat.color.set(settings.brainSurfaceColor);
            }
            mat.needsUpdate = true;
          } else if (mat instanceof THREE.PointsMaterial) {
            mat.transparent = true;
            mat.opacity = settings.brainSurfaceOpacity;
            if (settings.brainSurfaceColor !== '#e8d5cf') {
              mat.color.set(settings.brainSurfaceColor);
            }
            mat.needsUpdate = true;
          }
        });
      }

      // Handle Points objects (for point cloud model)
      if (child instanceof THREE.Points && child.material) {
        const mat = child.material as THREE.PointsMaterial;
        mat.transparent = true;
        mat.opacity = settings.brainSurfaceOpacity;
        mat.size = settings.streamlineWidth * 0.5;
        if (settings.brainSurfaceColor !== '#e8d5cf') {
          mat.color.set(settings.brainSurfaceColor);
        }
        mat.needsUpdate = true;
      }
    });
  }, [
    clonedScene,
    settings.brainSurfaceOpacity,
    settings.brainSurfaceWireframe,
    settings.brainSurfaceColor,
    settings.brainEmissiveIntensity,
    settings.brainMetalness,
    settings.brainRoughness,
    settings.streamlineWidth,
  ]);

  // Auto-rotate
  useFrame((_, delta) => {
    if (settings.autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * settings.autoRotateSpeed * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={clonedScene} />
    </group>
  );
}

// Preload models for faster switching
useGLTF.preload('/models/brain_hologram.glb');
useGLTF.preload('/models/brain_point_cloud.glb');
