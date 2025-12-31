/**
 * TypeScript type definitions for the Model Training Manager.
 */

// Dataset types
export interface Dataset {
  id: number;
  name: string;
  description: string | null;
  filename: string;
  file_path: string;
  file_type: 'csv' | 'json';
  file_size: number;
  row_count: number;
  column_count: number;
  columns: string | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetCreate {
  name: string;
  description?: string;
}

export interface DatasetUpdate {
  name?: string;
  description?: string;
}

export interface DatasetListResponse {
  items: Dataset[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

// Training job types
export type TrainingStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
export type TrainingType = 'qlora' | 'unsloth' | 'rag' | 'standard';

export interface TrainingJob {
  id: number;
  name: string;
  description: string | null;
  status: TrainingStatus;
  training_type: TrainingType;
  model_name: string;
  dataset_id: number;
  batch_size: number;
  learning_rate: number;
  epochs: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  progress: number;
  current_epoch: number;
  current_loss: number | null;
  error_message: string | null;
  log: string | null;
  model_path: string | null;
  worker_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface TrainingJobCreate {
  name: string;
  description?: string;
  dataset_id: number;
  training_type?: TrainingType;
  model_name?: string;
  batch_size?: number;
  learning_rate?: number;
  epochs?: number;
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
}

export interface TrainingJobUpdate {
  name?: string;
  description?: string;
}

export interface TrainingJobListResponse {
  items: TrainingJob[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface TrainingJobStatus {
  id: number;
  status: TrainingStatus;
  progress: number;
  current_epoch: number;
  current_loss: number | null;
  error_message: string | null;
}

// Training config types
export type ModelProvider = 'ollama' | 'lm_studio';

export interface TrainingConfig {
  id: number;
  max_concurrent_workers: number;
  active_workers: number;
  default_model: string;
  default_training_type: TrainingType;
  default_batch_size: number;
  default_learning_rate: number;
  default_epochs: number;
  default_lora_r: number;
  default_lora_alpha: number;
  default_lora_dropout: number;
  auto_start_workers: boolean;
  model_provider: ModelProvider;
  model_api_url: string;
  output_directory_base?: string | null;
  model_cache_path?: string | null;
  hf_token?: string | null;
  selected_gpus?: number[] | null;
  gpu_auto_detect: boolean;
  created_at: string;
  updated_at: string;
}

export interface TrainingConfigUpdate {
  max_concurrent_workers?: number;
  default_model?: string;
  default_training_type?: TrainingType;
  default_batch_size?: number;
  default_learning_rate?: number;
  default_epochs?: number;
  default_lora_r?: number;
  default_lora_alpha?: number;
  default_lora_dropout?: number;
  auto_start_workers?: boolean;
  model_provider?: ModelProvider;
  model_api_url?: string;
  output_directory_base?: string | null;
  model_cache_path?: string | null;
  hf_token?: string | null;
  selected_gpus?: number[];
  gpu_auto_detect?: boolean;
}

// Model types
export interface HuggingFaceModel {
  id: string;
  name: string;
  author: string;
  description: string;
  downloads: number;
  likes: number;
  tags: string[];
  model_type: string;
  private: boolean;
  last_modified?: string;
}

export interface ModelSearchResponse {
  items: HuggingFaceModel[];
  query: string;
  limit: number;
  offset: number;
}

export interface ModelDownloadRequest {
  model_id: string;
}

export interface ModelDownloadResponse {
  id: number;
  model_id: string;
  name: string;
  author: string;
  local_path: string;
  file_size: number;
  message: string;
}

export interface LocalModel {
  id: number;
  model_id: string;
  name: string;
  author: string;
  description: string | null;
  local_path: string;
  file_size: number;
  downloaded_at: string;
  tags: string[];
  model_type: string | null;
  is_private: boolean;
  created_at: string;
  updated_at: string;
}

// Worker types
export type WorkerStatus = 'idle' | 'busy' | 'stopped' | 'error';

export interface WorkerInfo {
  id: string;
  status: WorkerStatus;
  current_job_id: number | null;
  jobs_completed: number;
  started_at: string;
  last_activity: string | null;
}

export interface WorkerPoolStatus {
  total_workers: number;
  active_workers: number;
  idle_workers: number;
  busy_workers: number;
  max_workers: number;
  workers: WorkerInfo[];
  jobs_in_queue: number;
}

export interface WorkerCommand {
  action: 'start' | 'stop' | 'restart';
  worker_count?: number;
}

// API response types
export interface ApiError {
  detail: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
}

export interface GPUInfo {
  id: number;
  name: string;
  memory_total?: number | null;
  memory_free?: number | null;
  memory_used?: number | null;
}

// Hugging Face types
export interface HFDataset {
  id: string;
  name: string;
  author: string;
  description: string | null;
  downloads: number;
  likes: number;
  tags: string[];
  last_modified: string | null;
  private: boolean;
}

export interface HFSearchResponse {
  items: HFDataset[];
  query: string;
  limit: number;
  offset: number;
}

export interface HFDownloadRequest {
  dataset_id: string;
  name?: string;
  split: string;
  config?: string;
  max_rows?: number;
}

export interface HFDownloadResponse {
  id: number;
  name: string;
  dataset_id: string;
  row_count: number;
  columns: string[];
  file_path: string;
  message: string;
}

// Project types
export type TraitType = 'reasoning' | 'coding' | 'general_tools';
export type ProjectStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface DatasetAllocation {
  dataset_id: number;
  /** Percentage of the dataset file to use (0-100). For example, 50% means use half of the dataset's rows. */
  percentage: number;
}

export interface TraitConfiguration {
  trait_type: TraitType;
  datasets: DatasetAllocation[];
}

export interface ProjectTrait {
  id: number;
  trait_type: TraitType;
  datasets: DatasetAllocation[];
}

export interface Project {
  id: number;
  name: string;
  description: string | null;
  base_model: string;
  model_type: string | null;
  training_type: TrainingType;
  max_rows: number | null;
  output_directory: string;
  status: ProjectStatus;
  progress: number;
  current_epoch: number;
  current_loss: number | null;
  error_message: string | null;
  model_path: string | null;
  worker_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  traits: ProjectTrait[];
}

export interface ProjectCreate {
  name: string;
  description?: string;
  base_model: string;
  model_type?: string;
  training_type: TrainingType;
  max_rows?: number;
  output_directory: string;
  traits: TraitConfiguration[];
}

export interface ProjectUpdate {
  name?: string;
  description?: string;
  status?: ProjectStatus;
}

export interface ProjectListResponse {
  items: Project[];
  total: number;
  page: number;
  page_size: number;
}