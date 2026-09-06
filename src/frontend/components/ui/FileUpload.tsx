'use client';

import { useRef, useState } from 'react';
import { apiClient } from '@/lib/api';
import { formatFileSize } from '@/lib/utils';
import { DatasetValidationReport, UploadedFile } from '@/lib/types';
import { useAppStore } from '@/lib/store';

interface FileUploadProps { onUploadComplete?: () => void; }
const ACCEPTED_FILES = ['.nii', '.nii.gz', '.bval', '.bvals', '.bvec', '.bvecs'];

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [report, setReport] = useState<(DatasetValidationReport & { ready_for_pipeline: boolean }) | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addNotification, setCurrentJob } = useAppStore();

  const handleFiles = async (selectedFiles: File[]) => {
    const validFiles = selectedFiles.filter((file) => ACCEPTED_FILES.some((extension) => file.name.toLowerCase().endsWith(extension)));
    if (validFiles.length !== selectedFiles.length || validFiles.length === 0) {
      addNotification({ type: 'warning', title: 'Dataset files required', message: 'Select one DWI NIfTI file together with its .bval and .bvec files.', duration: 5000 });
      return;
    }
    const sessionId = crypto.randomUUID();
    setUploadId(sessionId);
    setReport(null);
    setFiles(validFiles.map((file) => ({ name: file.name, size: file.size, type: file.type || 'application/octet-stream', uploadedAt: new Date().toISOString(), status: 'uploading', progress: 0 })));
    try {
      for (const file of validFiles) {
        await apiClient.uploadFile(file, (progress) => setFiles((current) => current.map((item) => item.name === file.name ? { ...item, progress } : item)), sessionId);
        setFiles((current) => current.map((item) => item.name === file.name ? { ...item, status: 'uploaded', progress: 100 } : item));
      }
      const validation = await apiClient.validateUploadedDataset(sessionId);
      setReport(validation);
      if (validation.ready_for_pipeline) {
        addNotification({ type: 'success', title: 'Dataset ready for pipeline', message: 'The DWI volume and gradient tables passed validation. You can now start processing.', duration: 6000 });
        onUploadComplete?.();
      } else {
        addNotification({ type: 'warning', title: 'Upload needs attention', message: validation.errors.join(' ') || 'The dataset did not pass validation.', duration: 7000 });
      }
    } catch (error: any) {
      const message = error.message || 'Upload or validation failed';
      setFiles((current) => current.map((item) => item.status === 'uploading' ? { ...item, status: 'error', error: message } : item));
      addNotification({ type: 'error', title: 'Upload failed', message, duration: 7000 });
    }
  };

  const startPipeline = async () => {
    if (!uploadId || !report?.ready_for_pipeline) return;
    setIsSubmitting(true);
    try {
      const job = await apiClient.submitUploadedDataset(uploadId);
      setCurrentJob(job);
      addNotification({ type: 'success', title: 'Pipeline started', message: `Job ${job.id.slice(0, 8)} is running. Follow it in the execution panel above.`, duration: 6000 });
    } catch (error: any) {
      addNotification({ type: 'error', title: 'Pipeline submission failed', message: error.message || 'Could not start processing.', duration: 7000 });
    } finally { setIsSubmitting(false); }
  };

  return <div className="space-y-4">
    <div onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); handleFiles(Array.from(event.dataTransfer.files)); }} onClick={() => fileInputRef.current?.click()} className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${isDragging ? 'border-primary-500 bg-primary-500/10' : 'border-gray-600 hover:border-gray-500'}`} role="button" tabIndex={0} aria-label="Upload diffusion MRI dataset" onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') fileInputRef.current?.click(); }}>
      <input ref={fileInputRef} type="file" multiple accept=".nii,.nii.gz,.bval,.bvals,.bvec,.bvecs" onChange={(event) => handleFiles(event.target.files ? Array.from(event.target.files) : [])} className="hidden" />
      <p className="text-lg mb-2">{isDragging ? 'Drop the dataset files here' : 'Upload a diffusion MRI dataset'}</p>
      <p className="text-sm text-gray-400">Choose one 4D DWI NIfTI file plus the matching .bval and .bvec files.</p>
    </div>
    {files.length > 0 && <div className="space-y-2"><h4 className="text-sm font-semibold text-gray-300">Upload session</h4>{files.map((file) => <div key={file.name} className="glass rounded-lg p-3 flex items-center justify-between text-sm"><span>{file.name}</span><span className={file.status === 'error' ? 'text-red-400' : file.status === 'uploaded' ? 'text-green-400' : 'text-gray-400'}>{file.status === 'uploaded' ? 'Uploaded' : file.status === 'error' ? 'Failed' : `${file.progress}%`} · {formatFileSize(file.size)}</span></div>)}</div>}
    {report && <div className={`rounded-lg border p-4 ${report.ready_for_pipeline ? 'bg-green-500/10 border-green-500/40' : 'bg-amber-500/10 border-amber-500/40'}`}><p className="font-medium">{report.ready_for_pipeline ? 'Validated and ready for processing' : 'Dataset validation did not pass'}</p><p className="text-sm text-gray-300 mt-1">{report.ready_for_pipeline ? `${report.num_volumes} volumes; ${report.gradient_summary?.n_b0 ?? 0} b0 volumes; checksums recorded.` : report.errors.join(' ')}</p>{report.ready_for_pipeline && <button onClick={startPipeline} disabled={isSubmitting} className="mt-3 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded-lg text-sm font-medium">{isSubmitting ? 'Starting pipeline...' : 'Start pipeline'}</button>}</div>}
  </div>;
}
