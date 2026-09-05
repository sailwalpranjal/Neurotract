// Utility functions

import { type ClassValue, clsx } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

// Format file size
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Format duration
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

// Format date
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 60) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return date.toLocaleDateString();
}

// Debounce
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

// Throttle
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

// Download blob as file
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Parse TRK file header (simplified)
export function parseTRKHeader(buffer: ArrayBuffer): {
  numStreamlines: number;
  dimensions: [number, number, number];
  voxelSize: [number, number, number];
} {
  const view = new DataView(buffer);

  // TRK format: first 6 bytes are magic number
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5)
  );

  if (magic !== 'TRACK') {
    throw new Error('Invalid TRK file format');
  }

  // Dimensions at offset 6 (3 shorts)
  const dimensions: [number, number, number] = [
    view.getInt16(6, true),
    view.getInt16(8, true),
    view.getInt16(10, true),
  ];

  // Voxel size at offset 12 (3 floats)
  const voxelSize: [number, number, number] = [
    view.getFloat32(12, true),
    view.getFloat32(16, true),
    view.getFloat32(20, true),
  ];

  // Number of streamlines at offset 988
  const numStreamlines = view.getInt32(988, true);

  return {
    numStreamlines,
    dimensions,
    voxelSize,
  };
}

// Colormap utilities
export const COLORMAPS = [
  'viridis',
  'plasma',
  'inferno',
  'magma',
  'jet',
  'rainbow',
  'cool',
  'warm',
] as const;

export type Colormap = typeof COLORMAPS[number];

// Get color from colormap
export function getColorFromMap(
  value: number,
  min: number,
  max: number,
  colormap: Colormap = 'viridis'
): [number, number, number] {
  const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));

  // Simple viridis-like colormap implementation
  switch (colormap) {
    case 'viridis':
      return viridisColor(normalized);
    case 'plasma':
      return plasmaColor(normalized);
    case 'jet':
      return jetColor(normalized);
    case 'rainbow':
      return rainbowColor(normalized);
    default:
      return viridisColor(normalized);
  }
}

function viridisColor(t: number): [number, number, number] {
  const r = Math.max(0, Math.min(1, 0.282 + 0.718 * t - 0.382 * t * t));
  const g = Math.max(0, Math.min(1, -0.071 + 1.618 * t - 0.547 * t * t));
  const b = Math.max(0, Math.min(1, 0.219 + 0.781 * t));
  return [r, g, b];
}

function plasmaColor(t: number): [number, number, number] {
  const r = Math.max(0, Math.min(1, 0.549 + 0.451 * t));
  const g = Math.max(0, Math.min(1, 0.071 + 0.929 * t - 0.929 * t * t));
  const b = Math.max(0, Math.min(1, 0.821 - 0.821 * t));
  return [r, g, b];
}

function jetColor(t: number): [number, number, number] {
  let r, g, b;
  if (t < 0.25) {
    r = 0;
    g = 4 * t;
    b = 1;
  } else if (t < 0.5) {
    r = 0;
    g = 1;
    b = 1 - 4 * (t - 0.25);
  } else if (t < 0.75) {
    r = 4 * (t - 0.5);
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - 4 * (t - 0.75);
    b = 0;
  }
  return [r, g, b];
}

function rainbowColor(t: number): [number, number, number] {
  const h = t * 360;
  const s = 1;
  const v = 1;

  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;

  let r = 0, g = 0, b = 0;

  if (h >= 0 && h < 60) {
    r = c; g = x; b = 0;
  } else if (h >= 60 && h < 120) {
    r = x; g = c; b = 0;
  } else if (h >= 120 && h < 180) {
    r = 0; g = c; b = x;
  } else if (h >= 180 && h < 240) {
    r = 0; g = x; b = c;
  } else if (h >= 240 && h < 300) {
    r = x; g = 0; b = c;
  } else {
    r = c; g = 0; b = x;
  }

  return [r + m, g + m, b + m];
}

// Calculate streamline length
export function calculateStreamlineLength(points: Float32Array): number {
  let length = 0;
  for (let i = 3; i < points.length; i += 3) {
    const dx = points[i] - points[i - 3];
    const dy = points[i + 1] - points[i - 2];
    const dz = points[i + 2] - points[i - 1];
    length += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return length;
}

// Calculate mean orientation
export function calculateMeanOrientation(
  points: Float32Array
): [number, number, number] {
  let dx = 0, dy = 0, dz = 0;
  let count = 0;

  for (let i = 3; i < points.length; i += 3) {
    dx += points[i] - points[i - 3];
    dy += points[i + 1] - points[i - 2];
    dz += points[i + 2] - points[i - 1];
    count++;
  }

  const mag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (mag === 0) return [0, 0, 1];

  return [dx / mag, dy / mag, dz / mag];
}
