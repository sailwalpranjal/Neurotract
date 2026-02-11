"""
Memory Management Utilities

Handles memory-mapped arrays, streaming processing, and memory monitoring
for operations within 16GB RAM constraint.
"""

import numpy as np
import psutil
import h5py
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory usage and provides memory-efficient operations"""

    def __init__(self, max_memory_gb: float = 10.0):
        """
        Initialize memory manager

        Args:
            max_memory_gb: Maximum GB to use for in-memory operations (default 10GB)
        """
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.temp_files = []

    def get_available_memory(self) -> int:
        """Get currently available system memory in bytes"""
        return psutil.virtual_memory().available

    def get_memory_usage(self) -> float:
        """Get current process memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    def can_fit_in_memory(self, array_shape: Tuple, dtype: np.dtype) -> bool:
        """
        Check if an array of given shape and dtype can fit in available memory

        Args:
            array_shape: Shape tuple
            dtype: NumPy dtype

        Returns:
            True if array fits in memory budget
        """
        array_bytes = np.prod(array_shape) * np.dtype(dtype).itemsize
        available = self.get_available_memory()
        return array_bytes < min(self.max_memory_bytes, available * 0.8)

    def create_memory_mapped_array(
        self,
        shape: Tuple,
        dtype: np.dtype,
        filename: Optional[str] = None,
        mode: str = 'w+'
    ) -> np.memmap:
        """
        Create a memory-mapped array for large data

        Args:
            shape: Array shape
            dtype: Data type
            filename: Path to memmap file (auto-generated if None)
            mode: File mode ('r+', 'w+', 'r', etc.)

        Returns:
            Memory-mapped array
        """
        if filename is None:
            import tempfile
            fd, filename = tempfile.mkstemp(suffix='.dat', prefix='neurotract_')
            import os
            os.close(fd)
            self.temp_files.append(filename)

        logger.info(f"Creating memory-mapped array: shape={shape}, dtype={dtype}, file={filename}")

        mmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        return mmap

    def create_hdf5_dataset(
        self,
        filename: str,
        dataset_name: str,
        shape: Tuple,
        dtype: np.dtype,
        chunks: Optional[Tuple] = None,
        compression: str = 'gzip'
    ) -> h5py.Dataset:
        """
        Create an HDF5 dataset for streaming I/O

        Args:
            filename: HDF5 file path
            dataset_name: Name of dataset within file
            shape: Dataset shape
            dtype: Data type
            chunks: Chunk shape for efficient I/O
            compression: Compression method ('gzip', 'lzf', None)

        Returns:
            HDF5 dataset handle
        """
        logger.info(f"Creating HDF5 dataset: {filename}/{dataset_name}")

        # Auto-determine chunking if not provided
        if chunks is None and len(shape) > 2:
            # Chunk along spatial dimensions for typical neuroimaging access patterns
            chunk_size = list(shape)
            chunk_size[0] = min(16, shape[0])  # Limit first dimension
            chunks = tuple(chunk_size)

        f = h5py.File(filename, 'a')
        dataset = f.create_dataset(
            dataset_name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression if compression else None
        )

        return dataset

    def process_in_chunks(
        self,
        data: np.ndarray,
        func,
        chunk_size: int,
        axis: int = 0,
        **kwargs
    ) -> np.ndarray:
        """
        Process large array in chunks along specified axis

        Args:
            data: Input array
            func: Function to apply to each chunk
            chunk_size: Size of chunks
            axis: Axis to chunk along
            **kwargs: Additional arguments to func

        Returns:
            Processed array
        """
        num_chunks = int(np.ceil(data.shape[axis] / chunk_size))
        results = []

        logger.info(f"Processing {data.shape} array in {num_chunks} chunks")

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, data.shape[axis])

            # Extract chunk
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(start, end)
            chunk = data[tuple(slices)]

            # Process chunk
            result = func(chunk, **kwargs)
            results.append(result)

        # Concatenate results
        output = np.concatenate(results, axis=axis)
        return output

    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink()
                logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(max_memory_gb: float = 10.0) -> MemoryManager:
    """Get or create global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(max_memory_gb)
    return _memory_manager
