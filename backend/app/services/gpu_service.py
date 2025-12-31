"""
GPU detection service.

Provides functionality to detect and query available GPUs using PyTorch CUDA.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy-load torch to avoid import errors if not available
_TORCH_AVAILABLE = None
_TORCH_CUDA_AVAILABLE = None


def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_cuda_available() -> bool:
    """Check if CUDA is available in PyTorch."""
    global _TORCH_CUDA_AVAILABLE
    if _TORCH_CUDA_AVAILABLE is None:
        if not _check_torch_available():
            _TORCH_CUDA_AVAILABLE = False
        else:
            try:
                import torch
                _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
            except Exception:
                _TORCH_CUDA_AVAILABLE = False
    return _TORCH_CUDA_AVAILABLE


class GPUService:
    """
    Service for detecting and managing GPU information.
    
    Provides methods to detect available GPUs, get their properties,
    and manage GPU assignments for training workers.
    """
    
    def detect_gpus(self) -> List[Dict[str, Any]]:
        """
        Detect all available GPUs on the system.
        
        Returns:
            List of dictionaries containing GPU information:
                - id: GPU ID (integer)
                - name: GPU name/model
                - memory_total: Total memory in bytes
                - memory_free: Free memory in bytes (optional, may be None)
                - memory_used: Used memory in bytes (optional, may be None)
        """
        if not _check_cuda_available():
            logger.info("CUDA not available, no GPUs detected")
            return []
        
        try:
            import torch
            
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return []
            
            gpus = []
            for gpu_id in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(gpu_id)
                    
                    # Try to get memory info if available
                    memory_total = props.total_memory if hasattr(props, 'total_memory') else None
                    memory_free = None
                    memory_used = None
                    
                    try:
                        # Allocate temporary tensor to get current memory usage
                        torch.cuda.set_device(gpu_id)
                        memory_allocated = torch.cuda.memory_allocated(gpu_id)
                        memory_reserved = torch.cuda.memory_reserved(gpu_id)
                        if memory_total:
                            memory_free = memory_total - memory_reserved
                            memory_used = memory_allocated
                    except Exception:
                        # If we can't get memory info, continue without it
                        pass
                    
                    gpu_info = {
                        "id": gpu_id,
                        "name": props.name if hasattr(props, 'name') else f"GPU {gpu_id}",
                        "memory_total": memory_total,
                        "memory_free": memory_free,
                        "memory_used": memory_used,
                    }
                    
                    gpus.append(gpu_info)
                    logger.debug(f"Detected GPU {gpu_id}: {gpu_info['name']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to get properties for GPU {gpu_id}: {e}")
                    continue
            
            logger.info(f"Detected {len(gpus)} GPU(s)")
            return gpus
            
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
            return []
    
    def get_gpu_count(self) -> int:
        """
        Get the number of available GPUs.
        
        Returns:
            Number of GPUs available, or 0 if CUDA is not available.
        """
        if not _check_cuda_available():
            return 0
        
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0
    
    def is_gpu_available(self, gpu_id: int) -> bool:
        """
        Check if a specific GPU is available.
        
        Args:
            gpu_id: GPU ID to check.
            
        Returns:
            True if GPU is available, False otherwise.
        """
        if not _check_cuda_available():
            return False
        
        try:
            import torch
            return 0 <= gpu_id < torch.cuda.device_count()
        except Exception:
            return False
