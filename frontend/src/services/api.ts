/**
 * API client service for communicating with the backend.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  Dataset,
  DatasetCreate,
  DatasetUpdate,
  DatasetListResponse,
  TrainingJob,
  TrainingJobCreate,
  TrainingJobUpdate,
  TrainingJobListResponse,
  TrainingJobStatus,
  TrainingConfig,
  TrainingConfigUpdate,
  WorkerPoolStatus,
  WorkerCommand,
  HealthStatus,
  TrainingStatus,
  HFSearchResponse,
  HFDataset,
  HFDownloadRequest,
  HFDownloadResponse,
  Project,
  ProjectCreate,
  ProjectUpdate,
  ProjectListResponse,
  GPUInfo,
} from '../types';

const API_BASE_URL = '/api/v1';

/**
 * Create configured axios instance.
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      if (error.response) {
        const message = (error.response.data as { detail?: string })?.detail || 'An error occurred';
        return Promise.reject(new Error(message));
      }
      return Promise.reject(error);
    }
  );

  return client;
};

const api = createApiClient();

// Health API
export const healthApi = {
  check: async (): Promise<HealthStatus> => {
    const response = await axios.get('/health');
    return response.data;
  },
};

// Dataset API
export const datasetApi = {
  list: async (page = 1, pageSize = 10): Promise<DatasetListResponse> => {
    const response = await api.get('/datasets/', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (id: number): Promise<Dataset> => {
    const response = await api.get(`/datasets/${id}`);
    return response.data;
  },

  upload: async (file: File, data: DatasetCreate): Promise<Dataset> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', data.name);
    if (data.description) {
      formData.append('description', data.description);
    }

    const response = await api.post('/datasets/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  update: async (id: number, data: DatasetUpdate): Promise<Dataset> => {
    const response = await api.patch(`/datasets/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/datasets/${id}`);
  },

  /**
   * Scan the data directory structure and auto-add valid datasets to the database.
   */
  scan: async (): Promise<{
    scanned: number;
    added: number;
    skipped: number;
    added_datasets: string[];
    skipped_paths: string[];
  }> => {
    const response = await api.post('/datasets/scan');
    return response.data;
  },
};

/**
 * Model file information from the backend.
 */
interface ModelFileInfo {
  type: 'file' | 'directory';
  name: string;
  size?: number;
  total_size?: number;
  file_count?: number;
  files?: Array<{ name: string; size: number }>;
  path: string;
}

// Training Job API
export const trainingJobApi = {
  list: async (
    page = 1,
    pageSize = 10,
    status?: TrainingStatus
  ): Promise<TrainingJobListResponse> => {
    const response = await api.get('/jobs/', {
      params: { page, page_size: pageSize, status },
    });
    return response.data;
  },

  get: async (id: number): Promise<TrainingJob> => {
    const response = await api.get(`/jobs/${id}`);
    return response.data;
  },

  create: async (data: TrainingJobCreate): Promise<TrainingJob> => {
    const response = await api.post('/jobs/', data);
    return response.data;
  },

  update: async (id: number, data: TrainingJobUpdate): Promise<TrainingJob> => {
    const response = await api.patch(`/jobs/${id}`, data);
    return response.data;
  },

  start: async (id: number): Promise<TrainingJob> => {
    const response = await api.post(`/jobs/${id}/start`);
    return response.data;
  },

  cancel: async (id: number): Promise<TrainingJob> => {
    const response = await api.post(`/jobs/${id}/cancel`);
    return response.data;
  },

  getStatus: async (id: number): Promise<TrainingJobStatus> => {
    const response = await api.get(`/jobs/${id}/status`);
    return response.data;
  },

  /**
   * Download the trained model for a completed job.
   */
  download: async (id: number): Promise<Blob> => {
    const response = await api.get(`/jobs/${id}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },

  /**
   * Get the download URL for a job's model.
   */
  getDownloadUrl: (id: number): string => {
    return `${API_BASE_URL}/jobs/${id}/download`;
  },

  /**
   * Get information about the model files.
   */
  getModelInfo: async (id: number): Promise<ModelFileInfo> => {
    const response = await api.get(`/jobs/${id}/model-info`);
    return response.data;
  },
};

// Config API
export const configApi = {
  get: async (): Promise<TrainingConfig> => {
    const response = await api.get('/config/');
    return response.data;
  },

  update: async (data: TrainingConfigUpdate): Promise<TrainingConfig> => {
    const response = await api.patch('/config/', data);
    return response.data;
  },

  getGPUs: async (): Promise<GPUInfo[]> => {
    const response = await api.get('/config/gpus');
    return response.data;
  },
};

// Worker API
export const workerApi = {
  getStatus: async (): Promise<WorkerPoolStatus> => {
    const response = await api.get('/workers/');
    return response.data;
  },

  control: async (command: WorkerCommand): Promise<WorkerPoolStatus> => {
    const response = await api.post('/workers/', command);
    return response.data;
  },

  start: async (count: number): Promise<WorkerPoolStatus> => {
    return workerApi.control({ action: 'start', worker_count: count });
  },

  stop: async (): Promise<WorkerPoolStatus> => {
    return workerApi.control({ action: 'stop' });
  },

  restart: async (): Promise<WorkerPoolStatus> => {
    return workerApi.control({ action: 'restart' });
  },
};

// Models API
export const modelsApi = {
  /**
   * Search for models on HuggingFace Hub.
   */
  search: async (query: string, limit: number = 20, offset: number = 0): Promise<ModelSearchResponse> => {
    const response = await api.get('/models/search', {
      params: { query, limit, offset },
    });
    return response.data;
  },

  /**
   * Get detailed information about a model from HuggingFace Hub.
   */
  getInfo: async (modelId: string): Promise<HuggingFaceModel> => {
    const response = await api.get(`/models/${encodeURIComponent(modelId)}`);
    return response.data;
  },

  /**
   * Download a model from HuggingFace Hub.
   */
  download: async (modelId: string): Promise<ModelDownloadResponse> => {
    const response = await api.post('/models/download', { model_id: modelId } as ModelDownloadRequest);
    return response.data;
  },

  /**
   * List all locally downloaded models.
   */
  listLocal: async (): Promise<LocalModel[]> => {
    const response = await api.get('/models/');
    return response.data;
  },

  /**
   * Get information about a locally downloaded model.
   */
  getLocal: async (modelId: string): Promise<LocalModel> => {
    const response = await api.get(`/models/local/${encodeURIComponent(modelId)}`);
    return response.data;
  },

  /**
   * Delete a locally downloaded model.
   */
  deleteLocal: async (modelId: string): Promise<void> => {
    await api.delete(`/models/local/${encodeURIComponent(modelId)}`);
  },

  /**
   * Scan the model directory structure and auto-add valid models to the database.
   */
  scan: async (): Promise<{
    scanned: number;
    added: number;
    skipped: number;
    added_models: string[];
    skipped_paths: string[];
  }> => {
    const response = await api.post('/models/scan');
    return response.data;
  },
};

// Hugging Face API
export const huggingfaceApi = {
  search: async (
    query: string,
    limit = 20,
    offset = 0
  ): Promise<HFSearchResponse> => {
    const response = await api.get('/huggingface/search', {
      params: { query, limit, offset },
    });
    return response.data;
  },

  getDatasetInfo: async (datasetId: string): Promise<HFDataset> => {
    const response = await api.get(`/huggingface/datasets/${encodeURIComponent(datasetId)}`);
    return response.data;
  },

  getConfigs: async (datasetId: string): Promise<{ configs: string[] }> => {
    const response = await api.get(`/huggingface/datasets/${encodeURIComponent(datasetId)}/configs`);
    return response.data;
  },

  download: async (request: HFDownloadRequest): Promise<HFDownloadResponse> => {
    const response = await api.post('/huggingface/download', request);
    return response.data;
  },
};

// Project API
export const projectApi = {
  list: async (page = 1, pageSize = 50): Promise<ProjectListResponse> => {
    const response = await api.get('/projects/', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (id: number): Promise<Project> => {
    const response = await api.get(`/projects/${id}`);
    return response.data;
  },

  create: async (data: ProjectCreate): Promise<Project> => {
    const response = await api.post('/projects/', data);
    return response.data;
  },

  update: async (id: number, data: ProjectUpdate): Promise<Project> => {
    const response = await api.patch(`/projects/${id}`, data);
    return response.data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/projects/${id}`);
  },

  start: async (id: number): Promise<Project> => {
    const response = await api.post(`/projects/${id}/start`);
    return response.data;
  },

  cancel: async (id: number): Promise<Project> => {
    const response = await api.post(`/projects/${id}/cancel`);
    return response.data;
  },

  retry: async (id: number): Promise<Project> => {
    const response = await api.post(`/projects/${id}/retry`);
    return response.data;
  },

  validate: async (id: number): Promise<any> => {
    const response = await api.post(`/projects/${id}/validate`);
    return response.data;
  },

  validateOutputDir: async (output_directory: string): Promise<{ valid: boolean; writable: boolean; path: string; errors: string[] }> => {
    const response = await api.post('/projects/validate-output-dir', { output_directory });
    return response.data;
  },

  validateModel: async (model_name: string): Promise<{ available: boolean; model_name: string; path?: string; errors: string[] }> => {
    const response = await api.post('/projects/validate-model', { model_name });
    return response.data;
  },

  listAvailableModels: async (): Promise<string[]> => {
    const response = await api.get('/projects/models/available');
    return response.data;
  },

  getModelTypes: async (modelName: string): Promise<{
    model_type: string;
    available_types: string[];
    recommended: string | null;
    model_name: string;
  }> => {
    const response = await api.get(`/projects/models/${encodeURIComponent(modelName)}/types`);
    return response.data;
  },
};

export default api;

