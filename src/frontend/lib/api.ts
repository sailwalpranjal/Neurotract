// API client for FastAPI backend

import axios, { AxiosInstance, AxiosProgressEvent } from 'axios';
import {
  Job,
  JobResult,
  GraphMetrics,
  APIError,
  ProcessedResult,
  StreamlineBundle,
  BrainMeshData,
  ParcellationLabel,
  DatasetValidationReport,
  MetricProvenance,
  ExecutionEvent,
  OrthogonalSlicesData,
  ValidationBenchmarkResult,
  SensitivityResult,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 300000, // 5 minutes for long-running jobs
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        let message = 'An error occurred';
        const rawDetail = error.response?.data?.detail;
        if (typeof rawDetail === 'string') {
          message = rawDetail;
        } else if (Array.isArray(rawDetail)) {
          message = rawDetail
            .map((item: any) =>
              typeof item === 'object' && item?.msg
                ? `${item.loc?.filter((l: any) => l !== 'body').join('.') || 'parameter'}: ${item.msg}`
                : String(item)
            )
            .join('; ');
        } else if (typeof rawDetail === 'object' && rawDetail !== null) {
          message = rawDetail.message || JSON.stringify(rawDetail);
        } else if (error.message) {
          message = error.message;
        }

        const apiError: APIError = {
          message,
          code: error.response?.status?.toString() || 'UNKNOWN',
          details: error.response?.data,
        };
        return Promise.reject(apiError);
      }
    );
  }

  // Health check
  async health(): Promise<{ status: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // File upload
  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ file_id: string; filename: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  }

  // Submit full analysis pipeline job
  async submitJob(config: {
    subject_id: string;
    mode?: string;
    preprocessing?: Record<string, any>;
    tractography?: Record<string, any>;
    connectome?: Record<string, any>;
    rng_seed?: number;
  }): Promise<Job> {
    const response = await this.client.post('/jobs/submit', config);
    return response.data;
  }

  // Tractography
  async runTractography(params: {
    dwi_file: string;
    bval_file?: string;
    bvec_file?: string;
    mask_file?: string;
    algorithm?: string;
    step_size?: number;
    fa_threshold?: number;
    max_angle?: number;
    seeds_per_voxel?: number;
  }): Promise<Job> {
    const response = await this.client.post('/tractography', params);
    return response.data;
  }

  // Graph analysis
  async runGraphAnalysis(params: {
    tractogram_file: string;
    atlas_file?: string;
    parcellation_scheme?: string;
    threshold?: number;
  }): Promise<Job> {
    const response = await this.client.post('/graph-analysis', params);
    return response.data;
  }

  // Job status
  async getJobStatus(jobId: string): Promise<Job> {
    const response = await this.client.get(`/jobs/${jobId}`);
    return response.data;
  }

  // Get all jobs
  async getJobs(limit = 50): Promise<Job[]> {
    const response = await this.client.get(`/jobs?limit=${limit}`);
    return response.data;
  }

  // Cancel job
  async cancelJob(jobId: string): Promise<void> {
    await this.client.post(`/jobs/${jobId}/cancel`);
  }

  // Download results
  async downloadResult(jobId: string, resultType: string): Promise<Blob> {
    const response = await this.client.get(
      `/jobs/${jobId}/results/${resultType}`,
      {
        responseType: 'blob',
      }
    );
    return response.data;
  }

  // Get graph metrics
  async getGraphMetrics(jobId: string): Promise<GraphMetrics> {
    const response = await this.client.get(`/jobs/${jobId}/metrics`);
    return response.data;
  }

  // Get connectome matrix
  async getConnectome(jobId: string): Promise<number[][]> {
    const response = await this.client.get(`/jobs/${jobId}/connectome`);
    return response.data;
  }

  // Fetch streamlines data (binary)
  async fetchStreamlines(url: string): Promise<ArrayBuffer> {
    const response = await this.client.get(url, {
      responseType: 'arraybuffer',
    });
    return response.data;
  }

  // Fetch NIfTI volume
  async fetchVolume(url: string): Promise<ArrayBuffer> {
    const response = await this.client.get(url, {
      responseType: 'arraybuffer',
    });
    return response.data;
  }

  // Get available atlases
  async getAtlases(): Promise<{ name: string; description: string }[]> {
    const response = await this.client.get('/atlases');
    return response.data;
  }

  // Get available algorithms
  async getAlgorithms(): Promise<{ name: string; description: string }[]> {
    const response = await this.client.get('/algorithms');
    return response.data;
  }

  // ── Pre-computed results endpoints ──

  // Get available processed results
  async getAvailableResults(): Promise<ProcessedResult[]> {
    const response = await this.client.get('/results/available');
    return response.data;
  }

  // Get streamlines for a subject (returns StreamlineBundle-compatible data)
  async getResultStreamlines(subjectId: string, maxStreamlines = 3000): Promise<any> {
    const response = await this.client.get(
      `/results/${subjectId}/streamlines?max_streamlines=${maxStreamlines}`,
      { timeout: 120000 } // 2 min for large files
    );
    return response.data;
  }

  // Get metrics for a subject
  async getResultMetrics(subjectId: string): Promise<GraphMetrics> {
    const response = await this.client.get(`/results/${subjectId}/metrics`);
    return response.data;
  }

  // Get connectome for a subject
  async getResultConnectome(subjectId: string): Promise<number[][]> {
    const response = await this.client.get(`/results/${subjectId}/connectome`);
    return response.data;
  }

  // Get all info for a subject
  async getResultInfo(subjectId: string): Promise<any> {
    const response = await this.client.get(`/results/${subjectId}/info`);
    return response.data;
  }

  // Get brain surface mesh (marching cubes from brain mask)
  async getBrainMesh(subjectId: string, stepSize = 1): Promise<BrainMeshData> {
    const response = await this.client.get(
      `/results/${subjectId}/brain-mesh?step_size=${stepSize}`,
      { timeout: 60000 }
    );
    return response.data;
  }

  // Get parcellation labels with anatomical names
  async getParcellationLabels(subjectId: string): Promise<{
    labels: ParcellationLabel[];
    atlas: string;
    n_parcels: number;
    lobe_centroids: Record<string, number[]>;
  }> {
    const response = await this.client.get(`/results/${subjectId}/parcellation-labels`);
    return response.data;
  }

  // ── NeuroTract 2.0 Scientific & Laboratory Endpoints ──

  // Validate dataset files against strict scientific standards
  async validateDataset(params: {
    dwi_path: string;
    bval_path?: string;
    bvec_path?: string;
    mask_path?: string;
  }): Promise<DatasetValidationReport> {
    const response = await this.client.post('/api/datasets/validate', params);
    return response.data;
  }

  // Get metric provenance metadata and formula
  async getMetricProvenance(metricKey: string, executionId?: string): Promise<MetricProvenance> {
    const query = executionId ? `?execution_id=${encodeURIComponent(executionId)}` : '';
    const response = await this.client.get(`/api/provenance/${encodeURIComponent(metricKey)}${query}`);
    return response.data;
  }

  // Get all registered metric provenances
  async getAllProvenance(executionId?: string): Promise<Record<string, MetricProvenance>> {
    const query = executionId ? `?execution_id=${encodeURIComponent(executionId)}` : '';
    const response = await this.client.get(`/api/provenance${query}`);
    return response.data;
  }

  // Run reference validation benchmark (against DIPY and NetworkX baselines)
  async runValidationBenchmark(subjectId = 'SUB1'): Promise<ValidationBenchmarkResult> {
    const response = await this.client.post('/api/validation/benchmark', {
      subject_id: subjectId,
    }, { timeout: 120000 });
    return response.data;
  }

  // Run connectome parameter sensitivity lab
  async runSensitivityAnalysis(params?: {
    subject_id?: string;
    thresholds?: number[];
  }): Promise<SensitivityResult> {
    const response = await this.client.post('/api/sensitivity/run', params || {}, {
      timeout: 120000,
    });
    return response.data;
  }

  // Get orthogonal anatomical slices (Axial, Coronal, Sagittal) from real NIfTI volumes
  async getOrthogonalSlices(
    subjectId: string,
    options?: {
      volume_type?: 'b0' | 'fa' | 'md' | 't1';
      x?: number;
      y?: number;
      z?: number;
      colormap?: string;
    }
  ): Promise<OrthogonalSlicesData> {
    const queryParams = new URLSearchParams();
    if (options?.volume_type) queryParams.append('volume_type', options.volume_type);
    if (options?.x !== undefined) queryParams.append('x', options.x.toString());
    if (options?.y !== undefined) queryParams.append('y', options.y.toString());
    if (options?.z !== undefined) queryParams.append('z', options.z.toString());
    if (options?.colormap) queryParams.append('colormap', options.colormap);

    const queryStr = queryParams.toString() ? `?${queryParams.toString()}` : '';
    const response = await this.client.get(`/results/${subjectId}/slices${queryStr}`);
    return response.data;
  }

  // Export reproducible standalone HTML report
  getReportExportUrl(subjectId: string): string {
    return `${API_BASE_URL}/api/report/${subjectId}/export`;
  }

  // Subscribe to live Server-Sent Events (SSE) for real-time stage progress
  subscribeJobEvents(
    jobId: string,
    onEvent: (event: ExecutionEvent) => void,
    onError?: (err: Event) => void
  ): () => void {
    const eventSource = new EventSource(`${API_BASE_URL}/jobs/${jobId}/events`);

    eventSource.onmessage = (event) => {
      try {
        const parsed: ExecutionEvent = JSON.parse(event.data);
        onEvent(parsed);
      } catch (e) {
        console.error('Error parsing SSE event data:', e);
      }
    };

    eventSource.onerror = (err) => {
      console.warn('SSE EventSource error:', err);
      if (onError) onError(err);
    };

    return () => {
      eventSource.close();
    };
  }
}

// Singleton instance
export const apiClient = new APIClient();

// Job polling utility
export async function pollJobUntilComplete(
  jobId: string,
  onProgress?: (job: Job) => void,
  interval = 2000
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const job = await apiClient.getJobStatus(jobId);

        if (onProgress) {
          onProgress(job);
        }

        if (job.status === 'completed') {
          resolve(job);
        } else if (job.status === 'failed') {
          reject(new Error(job.error || 'Job failed'));
        } else {
          setTimeout(poll, interval);
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
}

export default apiClient;
