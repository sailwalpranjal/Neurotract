// Zustand store for global state management

import { create } from 'zustand';
import {
  Job,
  ViewerSettings,
  StreamlineBundle,
  NiftiVolume,
  Notification,
  ProcessedResult,
  GraphMetrics,
  BrainMeshData,
  ParcellationLabel,
  UserType,
  OrthogonalSlicesData,
  ExecutionEvent,
  ValidationBenchmarkResult,
  SensitivityResult,
  ConnectomeEdge,
} from './types';

interface AppState {
  // Jobs
  jobs: Job[];
  currentJob: Job | null;
  addJob: (job: Job) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  setCurrentJob: (job: Job | null) => void;

  // Viewer
  viewerSettings: ViewerSettings;
  updateViewerSettings: (updates: Partial<ViewerSettings>) => void;

  // Data
  streamlineBundle: StreamlineBundle | null;
  volumes: Record<string, NiftiVolume>;
  setStreamlineBundle: (bundle: StreamlineBundle | null) => void;
  setVolume: (name: string, volume: NiftiVolume) => void;

  // Brain mesh
  brainMesh: BrainMeshData | null;
  setBrainMesh: (mesh: BrainMeshData | null) => void;

  // Parcellation labels
  parcellationLabels: ParcellationLabel[];
  setParcellationLabels: (labels: ParcellationLabel[]) => void;

  // User type (doctor, student, general)
  userType: UserType;
  setUserType: (type: UserType) => void;

  // Analysis data (direct load without requiring currentJob)
  metrics: GraphMetrics | null;
  connectome: number[][] | null;
  setMetrics: (metrics: GraphMetrics | null) => void;
  setConnectome: (connectome: number[][] | null) => void;

  // Active subject
  activeSubject: string | null;
  setActiveSubject: (subject: string | null) => void;

  // Available results
  availableResults: ProcessedResult[];
  setAvailableResults: (results: ProcessedResult[]) => void;

  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // UI
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
  statusMessage: string | null;
  setStatusMessage: (msg: string | null) => void;

  // ── Slices & Anatomical Plane Synchronization ──
  slicesData: OrthogonalSlicesData | null;
  sliceIndices: { x: number; y: number; z: number } | null;
  activeSliceType: 'b0' | 'fa' | 'md' | 't1';
  isSliceCrosshairSynced: boolean;
  setSlicesData: (data: OrthogonalSlicesData | null) => void;
  setSliceIndices: (indices: { x: number; y: number; z: number } | null) => void;
  setActiveSliceType: (type: 'b0' | 'fa' | 'md' | 't1') => void;
  setIsSliceCrosshairSynced: (synced: boolean) => void;

  // ── Parcellation & Selection ──
  selectedParcel: ParcellationLabel | null;
  setSelectedParcel: (parcel: ParcellationLabel | null) => void;

  // ── Execution Events (SSE) ──
  executionEvents: ExecutionEvent[];
  addExecutionEvent: (event: ExecutionEvent) => void;
  clearExecutionEvents: () => void;

  // ── Provenance Inspection ──
  provenanceModalKey: string | null;
  setProvenanceModalKey: (key: string | null) => void;

  // ── Validation Benchmark & Sensitivity Results ──
  validationBenchmark: ValidationBenchmarkResult | null;
  setValidationBenchmark: (res: ValidationBenchmarkResult | null) => void;
  sensitivityResult: SensitivityResult | null;
  setSensitivityResult: (res: SensitivityResult | null) => void;

  // ── Synchronized Connectome & Analytics State ──
  hoveredEdge: ConnectomeEdge | null;
  selectedEdge: ConnectomeEdge | null;
  hoveredRegion: number | null;
  selectedRegion: number | null;
  connectomeThreshold: number;
  setHoveredEdge: (edge: ConnectomeEdge | null) => void;
  setSelectedEdge: (edge: ConnectomeEdge | null) => void;
  setHoveredRegion: (idx: number | null) => void;
  setSelectedRegion: (idx: number | null) => void;
  setConnectomeThreshold: (th: number) => void;
}

let notificationCounter = 0;

// Load user type from localStorage
function getInitialUserType(): UserType {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('neurotract_user_type');
    if (saved === 'doctor' || saved === 'student' || saved === 'general') return saved;
  }
  return 'general';
}

export const useAppStore = create<AppState>((set) => ({
  // Jobs
  jobs: [],
  currentJob: null,
  addJob: (job) =>
    set((state) => ({
      jobs: [job, ...state.jobs],
    })),
  updateJob: (jobId, updates) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === jobId ? { ...job, ...updates } : job
      ),
      currentJob:
        state.currentJob?.id === jobId
          ? { ...state.currentJob, ...updates }
          : state.currentJob,
    })),
  setCurrentJob: (job) => set({ currentJob: job }),

  // Viewer
  viewerSettings: {
    colorMapping: {
      type: 'orientation',
      colormap: 'viridis',
    },
    streamlineOpacity: 0.8,
    streamlineWidth: 1.0,
    showBrainSurface: true,
    showStreamlines: true,
    showSlices: false,
    showLabels: false,
    brainModelType: 'marching_cubes',
    brainSurfaceOpacity: 0.15,
    brainSurfaceColor: '#e8d5cf',
    brainSurfaceWireframe: false,
    brainEmissiveIntensity: 0.3,
    brainMetalness: 0.1,
    brainRoughness: 0.7,
    slicePosition: {
      axial: 0.5,
      coronal: 0.5,
      sagittal: 0.5,
    },
    backgroundColor: '#1a1a2e',
    cameraPosition: [0, 0, 300],
    levelOfDetail: 'medium',
    selectedRegion: null,
    autoRotate: false,
    autoRotateSpeed: 1.0,
    showConnectomeGraph: true,
    showParcellationNodes: true,
    connectomeEdgeThreshold: 1.0,
  },
  updateViewerSettings: (updates) =>
    set((state) => ({
      viewerSettings: {
        ...state.viewerSettings,
        ...updates,
      },
    })),

  // Data
  streamlineBundle: null,
  volumes: {},
  setStreamlineBundle: (bundle) => set({ streamlineBundle: bundle }),
  setVolume: (name, volume) =>
    set((state) => ({
      volumes: {
        ...state.volumes,
        [name]: volume,
      },
    })),

  // Brain mesh
  brainMesh: null,
  setBrainMesh: (mesh) => set({ brainMesh: mesh }),

  // Parcellation labels
  parcellationLabels: [],
  setParcellationLabels: (labels) => set({ parcellationLabels: labels }),

  // User type
  userType: getInitialUserType(),
  setUserType: (type) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('neurotract_user_type', type);
    }
    set({ userType: type });
  },

  // Analysis data
  metrics: null,
  connectome: null,
  setMetrics: (metrics) => set({ metrics }),
  setConnectome: (connectome) => set({ connectome }),

  // Active subject
  activeSubject: null,
  setActiveSubject: (subject) => set({ activeSubject: subject }),

  // Available results
  availableResults: [],
  setAvailableResults: (results) => set({ availableResults: results }),

  // Notifications
  notifications: [],
  addNotification: (notification) => {
    const id = `notif-${++notificationCounter}-${Date.now()}`;
    const full: Notification = {
      ...notification,
      id,
      timestamp: Date.now(),
      duration: notification.duration ?? 5000,
    };
    set((state) => ({
      notifications: [...state.notifications, full],
    }));
    // Auto-remove after duration (if not persistent)
    if (full.duration && full.duration > 0) {
      setTimeout(() => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        }));
      }, full.duration);
    }
    return id;
  },
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
  clearNotifications: () => set({ notifications: [] }),

  // UI
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  loading: false,
  setLoading: (loading) => set({ loading }),
  error: null,
  setError: (error) => set({ error }),
  statusMessage: null,
  setStatusMessage: (msg) => set({ statusMessage: msg }),

  // ── Slices & Anatomical Plane Synchronization ──
  slicesData: null,
  sliceIndices: null,
  activeSliceType: 'b0',
  isSliceCrosshairSynced: true,
  setSlicesData: (slicesData) => set({ slicesData }),
  setSliceIndices: (sliceIndices) => set({ sliceIndices }),
  setActiveSliceType: (activeSliceType) => set({ activeSliceType }),
  setIsSliceCrosshairSynced: (isSliceCrosshairSynced) => set({ isSliceCrosshairSynced }),

  // ── Parcellation & Selection ──
  selectedParcel: null,
  setSelectedParcel: (selectedParcel) => set({ selectedParcel }),

  // ── Execution Events (SSE) ──
  executionEvents: [],
  addExecutionEvent: (event) =>
    set((state) => ({
      executionEvents: [...state.executionEvents, event],
    })),
  clearExecutionEvents: () => set({ executionEvents: [] }),

  // ── Provenance Inspection ──
  provenanceModalKey: null,
  setProvenanceModalKey: (provenanceModalKey) => set({ provenanceModalKey }),

  // ── Validation Benchmark & Sensitivity Results ──
  validationBenchmark: null,
  setValidationBenchmark: (validationBenchmark) => set({ validationBenchmark }),
  sensitivityResult: null,
  setSensitivityResult: (sensitivityResult) => set({ sensitivityResult }),

  // ── Synchronized Connectome & Analytics State ──
  hoveredEdge: null,
  selectedEdge: null,
  hoveredRegion: null,
  selectedRegion: null,
  connectomeThreshold: 1.0,
  setHoveredEdge: (hoveredEdge) => set({ hoveredEdge }),
  setSelectedEdge: (selectedEdge) => set({ selectedEdge }),
  setHoveredRegion: (hoveredRegion) => set({ hoveredRegion }),
  setSelectedRegion: (selectedRegion) => set({ selectedRegion }),
  setConnectomeThreshold: (connectomeThreshold) => set({ connectomeThreshold }),
}));
