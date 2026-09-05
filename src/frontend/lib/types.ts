// Core data types for NeuroTract 2.0

export type UserType = 'doctor' | 'student' | 'general';

// Parcellation label with anatomical mapping
export interface ParcellationLabel {
  index: number;
  generic_name: string;
  anatomical_name: string;
  abbreviation: string;
  hemisphere: string;
  lobe: string;
  description: string;
}

// Brain mesh data from marching cubes endpoint
export interface BrainMeshData {
  vertices: number[];   // flat [x,y,z,x,y,z,...]
  faces: number[];      // flat [i,j,k,i,j,k,...]
  normals: number[];    // flat [nx,ny,nz,...]
  metadata: {
    n_vertices: number;
    n_faces: number;
    bounds: { min: number[]; max: number[] };
    spacing: number[];
    source: string;
    step_size: number;
    smooth_sigma: number;
  };
}

// View preset for anatomical views
export interface ViewPreset {
  name: string;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  label: string;
}

// Metric interpretation for different user perspectives
export interface MetricInterpretation {
  doctor: string;
  student: string;
  general: string;
  normalRange?: [number, number];
  status?: 'normal' | 'elevated' | 'reduced' | 'abnormal';
}

export interface Streamline {
  points: Float32Array; // Flattened [x,y,z,x,y,z,...]
  numPoints: number;
  color?: [number, number, number];
  fa?: number; // Fractional anisotropy
  length?: number;
  orientation?: [number, number, number]; // Mean tangent vector
}

export interface StreamlineBundle {
  streamlines: Streamline[];
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  metadata: {
    count: number;
    totalPoints: number;
    meanLength: number;
    maxLength: number;
    minLength: number;
    totalInFile?: number;
  };
}

export interface NiftiVolume {
  data: Float32Array;
  dimensions: [number, number, number];
  affine: number[][]; // 4x4 affine matrix
  voxelSize: [number, number, number];
  dataRange: [number, number];
}

export interface ColorMapping {
  type: 'length' | 'orientation' | 'fa' | 'custom';
  colormap: string; // 'viridis', 'plasma', 'jet', etc.
  range?: [number, number];
}

export type BrainModelType = 'marching_cubes' | 'hologram' | 'point_cloud';

export interface ViewerSettings {
  colorMapping: ColorMapping;
  streamlineOpacity: number;
  streamlineWidth: number;
  showBrainSurface: boolean;
  showStreamlines: boolean;
  showSlices: boolean;
  showLabels: boolean;
  brainModelType: BrainModelType;
  brainSurfaceOpacity: number;
  brainSurfaceColor: string;
  brainSurfaceWireframe: boolean;
  brainEmissiveIntensity: number;
  brainMetalness: number;
  brainRoughness: number;
  slicePosition: {
    axial: number;
    coronal: number;
    sagittal: number;
  };
  backgroundColor: string;
  cameraPosition: [number, number, number];
  levelOfDetail: 'low' | 'medium' | 'high';
  selectedRegion: number | null;
  autoRotate: boolean;
  autoRotateSpeed: number;
}

export interface Job {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  task: string;
  created_at: string;
  updated_at: string;
  result?: JobResult;
  error?: string;
}

export interface JobResult {
  streamlines?: string;
  metrics?: GraphMetrics;
  connectome?: number[][];
  volumes?: {
    fa?: string;
    md?: string;
    ad?: string;
    rd?: string;
  };
}

export interface GraphMetrics {
  global: {
    clustering_coefficient: number;
    characteristic_path_length: number;
    global_efficiency: number;
    modularity: number;
    assortativity: number;
    small_worldness: number;
    density?: number;
    transitivity?: number;
  };
  nodal: {
    degree: number[];
    betweenness_centrality: number[];
    closeness_centrality: number[];
    local_efficiency: number[];
    node_strength?: number[];
    eigenvector_centrality?: number[];
  };
  rich_club?: {
    coefficients: number[];
    k_values: number[];
  };
  communities?: {
    louvain_partition?: number[];
    louvain_modularity?: number;
    leiden_partition?: number[];
    leiden_modularity?: number;
  };
}

export interface ProcessedResult {
  subject_id: string;
  has_streamlines: boolean;
  has_metrics: boolean;
  has_connectome: boolean;
  has_dti: boolean;
  has_fod: boolean;
  has_report?: boolean;
  files: ResultFile[];
  streamline_stats?: {
    bundle_statistics: {
      n_streamlines: number;
      mean_length: number;
      min_length: number;
      max_length: number;
    };
    tracking_metadata: {
      n_seeds: number;
      seeds_per_voxel: number;
      step_size: number;
      max_angle: number;
      fa_threshold?: number;
    };
  };
  connectome_info?: {
    n_parcels: number;
    n_edges: number;
    density: number;
    n_streamlines: number;
  };
}

export interface ResultFile {
  name: string;
  size_bytes: number;
  type: string;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'loading';
  title: string;
  message?: string;
  timestamp: number;
  duration?: number;
}

export interface APIError {
  message: string;
  code: string;
  details?: any;
}

export interface UploadedFile {
  id?: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: string;
  status: 'uploading' | 'uploaded' | 'completed' | 'error';
  progress: number;
  error?: string;
}

// ── Scientific Provenance & Validation Types ──

export interface DatasetValidationReport {
  is_valid: boolean;
  dataset_format: string;
  file_paths: Record<string, string>;
  checksums_sha256: Record<string, string>;
  dimensions?: number[];
  voxel_size_mm?: number[];
  num_volumes?: number;
  coordinate_system?: string;
  orientation_codes?: string;
  gradient_summary?: {
    n_total: number;
    n_b0: number;
    n_dwi: number;
    unique_bvals: number[];
    shell_distribution: Record<string, number>;
    is_unit_normalized: boolean;
    max_norm_deviation: number;
    b0_indices: number[];
  };
  metadata?: Record<string, any>;
  warnings: string[];
  errors: string[];
  transformations_applied: Array<{
    type: string;
    original_shape?: number[];
    transformed_shape?: number[];
    reason: string;
  }>;
  source_info?: {
    name?: string;
    license?: string;
  };
}

export interface MetricProvenance {
  metric_id: string;
  name: string;
  value: any;
  units: string;
  formula: string;
  description: string;
  reference_citation: string;
  input_properties: Record<string, any>;
  execution_id: string;
  timestamp: string;
  software_versions: Record<string, string>;
}

export interface ExecutionEvent {
  job_id: string;
  event_type: string;
  stage?: string;
  status?: string;
  progress?: number;
  elapsed_seconds?: number;
  message?: string;
  telemetry?: Record<string, any>;
  timestamp: string;
  results?: any;
  error?: string;
}

export interface OrthogonalSlicePlane {
  image: string; // base64 PNG data URL
  index: number;
  max_index: number;
  percentage: number;
  plane: string;
  dims: [number, number];
}

export interface OrthogonalSlicesData {
  volume_shape: [number, number, number];
  voxel_size_mm: [number, number, number];
  intensity_range: [number, number];
  indices: {
    axial: number;
    coronal: number;
    sagittal: number;
  };
  slices: {
    axial: OrthogonalSlicePlane;
    coronal: OrthogonalSlicePlane;
    sagittal: OrthogonalSlicePlane;
  };
}

export interface ValidationBenchmarkResult {
  status: string;
  all_passed: boolean;
  dti_benchmark: {
    benchmark_name: string;
    reference_toolkit: string;
    timestamp: string;
    software_versions: Record<string, string>;
    parameters: Record<string, any>;
    metrics: {
      fractional_anisotropy: {
        pearson_r: number;
        mean_absolute_error: number;
        max_absolute_error: number;
        rmse: number;
        tolerance_threshold: number;
        passed: boolean;
      };
      mean_diffusivity: {
        pearson_r: number;
        mean_absolute_error_mm2_s: number;
        max_absolute_error_mm2_s: number;
        rmse_mm2_s: number;
        tolerance_threshold: number;
        passed: boolean;
      };
    };
    summary: string;
  };
  graph_benchmark: {
    benchmark_name: string;
    reference_toolkit: string;
    matrix_nodes: number;
    matrix_edges: number;
    comparisons: Array<{
      metric: string;
      neurotract_value: number;
      reference_nx_value: number;
      absolute_diff: number;
      passed: boolean;
    }>;
    all_metrics_passed: boolean;
    summary: string;
  };
}

export interface SensitivityResult {
  analysis_type: string;
  parameter_varied: string;
  parameter_values: number[];
  n_nodes: number;
  total_possible_edges: number;
  summary: {
    stable_edge_count: number;
    variable_edge_count: number;
    union_edge_count: number;
    stability_ratio: number;
    mean_pairwise_jaccard: number;
  };
  pairwise_jaccard_matrix: number[][];
  runs: Array<{
    run_index: number;
    parameter_value: number;
    edge_count: number;
    density: number;
    global_efficiency: number;
    clustering_coefficient: number;
    characteristic_path_length: number;
    modularity: number;
    gained_edges_vs_baseline: number;
    lost_edges_vs_baseline: number;
  }>;
  methodology: string;
}
