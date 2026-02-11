'use client';

import { useState, useRef, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { formatFileSize } from '@/lib/utils';
import { UploadedFile } from '@/lib/types';
import { useAppStore } from '@/lib/store';

interface FileUploadProps {
  onUploadComplete?: () => void;
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addNotification } = useAppStore();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files ? Array.from(e.target.files) : [];
    handleFiles(selectedFiles);
  };

  const handleFiles = async (newFiles: File[]) => {
    // Filter valid files
    const validFiles = newFiles.filter((file) => {
      const validExtensions = ['.nii', '.nii.gz', '.trk', '.tck', '.dcm'];
      return validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));
    });

    if (validFiles.length === 0) {
      addNotification({
        type: 'warning',
        title: 'Invalid File Format',
        message: 'Please upload neuroimaging files (.nii, .nii.gz, .trk, .tck, .dcm)',
        duration: 5000,
      });
      return;
    }

    addNotification({
      type: 'info',
      title: 'Upload Started',
      message: `Uploading ${validFiles.length} file(s)...`,
      duration: 3000,
    });

    // Add to state
    const uploadedFiles: UploadedFile[] = validFiles.map((file) => ({
      name: file.name,
      size: file.size,
      type: file.type || 'application/octet-stream',
      uploadedAt: new Date().toISOString(),
      status: 'uploading',
      progress: 0,
    }));

    setFiles((prev) => [...prev, ...uploadedFiles]);

    // Upload each file
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      const fileIndex = files.length + i;

      try {
        await apiClient.uploadFile(file, (progress) => {
          setFiles((prev) =>
            prev.map((f, idx) =>
              idx === fileIndex ? { ...f, progress } : f
            )
          );
        });

        // Mark as uploaded
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === fileIndex
              ? { ...f, status: 'uploaded', progress: 100 }
              : f
          )
        );
        addNotification({
          type: 'success',
          title: 'File Uploaded',
          message: `${file.name} (${formatFileSize(file.size)})`,
          duration: 3000,
        });
      } catch (error: any) {
        // Mark as error
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === fileIndex ? { ...f, status: 'error', progress: 0 } : f
          )
        );
        addNotification({
          type: 'error',
          title: 'Upload Failed',
          message: `${file.name}: ${error.message || 'Unknown error'}`,
          duration: 6000,
        });
        console.error('Upload failed:', error);
      }
    }

    // Check if all uploads complete
    const allUploaded = files.every((f) => f.status === 'uploaded');
    if (allUploaded && onUploadComplete) {
      onUploadComplete();
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
          ${
            isDragging
              ? 'border-primary-500 bg-primary-500/10'
              : 'border-gray-600 hover:border-gray-500'
          }
        `}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload files"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            fileInputRef.current?.click();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".nii,.nii.gz,.trk,.tck,.dcm"
          onChange={handleFileSelect}
          className="hidden"
          aria-label="File input"
        />

        <svg
          className="mx-auto h-12 w-12 text-gray-400 mb-4"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
          aria-hidden="true"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <p className="text-lg mb-2">
          {isDragging ? 'Drop files here' : 'Drag and drop files here'}
        </p>
        <p className="text-sm text-gray-400">
          or click to browse
        </p>
        <p className="text-xs text-gray-500 mt-2">
          Supported: .nii, .nii.gz, .trk, .tck, .dcm
        </p>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-300">Uploaded Files</h4>
          {files.map((file, index) => (
            <FileItem
              key={index}
              file={file}
              onRemove={() => removeFile(index)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function FileItem({
  file,
  onRemove,
}: {
  file: UploadedFile;
  onRemove: () => void;
}) {
  return (
    <div className="glass rounded-lg p-4 flex items-center space-x-4">
      {/* Icon */}
      <div className="flex-shrink-0">
        {file.status === 'uploading' && (
          <div className="w-8 h-8 spinner" style={{ width: '24px', height: '24px' }} />
        )}
        {file.status === 'uploaded' && (
          <svg className="w-6 h-6 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        )}
        {file.status === 'error' && (
          <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.name}</p>
        <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>

        {/* Progress Bar */}
        {file.status === 'uploading' && (
          <div className="mt-2 w-full bg-gray-700 rounded-full h-1.5">
            <div
              className="bg-primary-600 h-1.5 rounded-full transition-all"
              style={{ width: `${file.progress}%` }}
            />
          </div>
        )}
      </div>

      {/* Remove Button */}
      <button
        onClick={onRemove}
        className="flex-shrink-0 p-1 hover:bg-white/10 rounded transition-colors"
        aria-label={`Remove ${file.name}`}
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </div>
  );
}
