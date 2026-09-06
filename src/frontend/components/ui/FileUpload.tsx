'use client';

import { useRef, useState } from 'react';
import { apiClient, UploadedDatasetCandidate, UploadedDatasetDiscovery, UploadedPartGroup } from '@/lib/api';
import { formatFileSize } from '@/lib/utils';
import { useAppStore } from '@/lib/store';

interface FileUploadProps { onUploadComplete?: () => void; }
const SUPPORTED = ['.nii', '.nii.gz', '.bval', '.bvals', '.bvec', '.bvecs'];

const relativeName = (file: File) => (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [discovery, setDiscovery] = useState<UploadedDatasetDiscovery | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0, bytes: 0 });
  const [submitting, setSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const { addNotification, setCurrentJob } = useAppStore();

  const inspectFiles = async (incoming: File[]) => {
    const files = incoming.filter((file) => SUPPORTED.some((extension) => file.name.toLowerCase().endsWith(extension)));
    if (!files.length) {
      addNotification({ type: 'warning', title: 'No supported dataset files', message: 'Choose a folder containing DWI NIfTI files and their bval/bvec sidecars.', duration: 6000 });
      return;
    }
    const ignored = incoming.length - files.length;
    const sessionId = crypto.randomUUID();
    setUploadId(sessionId);
    setDiscovery(null);
    setSelectedIds(new Set());
    setUploading(true);
    setProgress({ completed: 0, total: files.length, bytes: 0 });
    try {
      let uploadedBytes = 0;
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        await apiClient.uploadFile(file, undefined, sessionId, relativeName(file));
        uploadedBytes += file.size;
        setProgress({ completed: index + 1, total: files.length, bytes: uploadedBytes });
      }
      const report = await apiClient.discoverUploadedDatasets(sessionId);
      setDiscovery(report);
      const validIds = report.datasets.filter((item) => item.is_valid).map((item) => item.id);
      setSelectedIds(new Set(validIds));
      addNotification({
        type: report.summary.compatible ? 'success' : 'warning',
        title: 'Folder inspection complete',
        message: `${report.summary.compatible} compatible dataset(s), ${report.summary.incompatible} requiring attention.${ignored ? ` ${ignored} unrelated file(s) were skipped.` : ''}`,
        duration: 7000,
      });
      if (report.summary.compatible) onUploadComplete?.();
    } catch (error: any) {
      addNotification({ type: 'error', title: 'Folder upload failed', message: error.message || 'Could not inspect the selected files.', duration: 7000 });
    } finally { setUploading(false); }
  };

  const toggleCandidate = (candidate: UploadedDatasetCandidate) => {
    if (!candidate.is_valid) return;
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(candidate.id)) next.delete(candidate.id); else next.add(candidate.id);
      return next;
    });
  };

  const runDatasets = async (ids: string[]) => {
    if (!uploadId || !ids.length) return;
    setSubmitting(true);
    try {
      const jobs = [];
      for (const id of ids) jobs.push(await apiClient.submitUploadedDataset(uploadId, id));
      setCurrentJob(jobs[jobs.length - 1]);
      addNotification({ type: 'success', title: 'Pipeline jobs submitted', message: `${jobs.length} dataset${jobs.length === 1 ? '' : 's'} queued. The execution panel follows the latest job.`, duration: 7000 });
    } catch (error: any) {
      addNotification({ type: 'error', title: 'Pipeline submission failed', message: error.message || 'No further jobs were submitted.', duration: 7000 });
    } finally { setSubmitting(false); }
  };

  const mergeAndRun = async (group: UploadedPartGroup) => {
    if (!uploadId || !group.can_merge) return;
    setSubmitting(true);
    try {
      const job = await apiClient.submitUploadedDataset(uploadId, undefined, group.datasets.map((item) => item.id));
      setCurrentJob(job);
      addNotification({ type: 'success', title: 'Merged dataset queued', message: `${group.datasets.length} verified parts will be combined in order and processed as one dataset.`, duration: 7000 });
    } catch (error: any) {
      addNotification({ type: 'error', title: 'Merge submission failed', message: error.message || 'The selected parts were not queued.', duration: 7000 });
    } finally { setSubmitting(false); }
  };

  return <div className="space-y-5">
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <button type="button" onClick={() => fileInputRef.current?.click()} className="rounded-lg border border-primary-500/50 bg-primary-500/10 hover:bg-primary-500/20 p-5 text-left transition-colors">
        <span className="block text-sm font-semibold">Choose dataset files</span>
        <span className="block text-xs text-gray-400 mt-1">One DWI and matching bval/bvec files.</span>
      </button>
      <button type="button" onClick={() => folderInputRef.current?.click()} className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 hover:bg-cyan-500/20 p-5 text-left transition-colors">
        <span className="block text-sm font-semibold">Choose a dataset folder</span>
        <span className="block text-xs text-gray-400 mt-1">Scan all nested folders and organize compatible datasets.</span>
      </button>
    </div>
    <input ref={fileInputRef} type="file" multiple accept=".nii,.nii.gz,.bval,.bvals,.bvec,.bvecs" onChange={(event) => inspectFiles(event.target.files ? Array.from(event.target.files) : [])} className="hidden" />
    <input ref={folderInputRef} type="file" multiple {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} onChange={(event) => inspectFiles(event.target.files ? Array.from(event.target.files) : [])} className="hidden" />

    {uploading && <div className="rounded-lg border border-primary-500/30 bg-primary-950/30 p-4 text-sm"><div className="flex justify-between gap-3"><span>Uploading and preserving folder layout</span><span className="font-mono">{progress.completed}/{progress.total}</span></div><div className="mt-2 h-1.5 rounded-full bg-gray-700 overflow-hidden"><div className="h-full bg-primary-500" style={{ width: `${progress.total ? progress.completed / progress.total * 100 : 0}%` }} /></div><p className="mt-2 text-xs text-gray-400">{formatFileSize(progress.bytes)} transferred. Large files are sent one at a time to keep memory use bounded.</p></div>}

    {discovery && <section className="space-y-4" aria-live="polite">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-lg border border-gray-700 bg-black/20 p-4">
        <div><h4 className="font-semibold">Dataset compatibility review</h4><p className="text-xs text-gray-400 mt-1">{discovery.summary.total} DWI files found; {discovery.summary.compatible} ready; {discovery.summary.incompatible} incomplete or incompatible.</p></div>
        <button type="button" disabled={submitting || selectedIds.size === 0} onClick={() => runDatasets([...selectedIds])} className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded-lg text-sm font-medium whitespace-nowrap">{submitting ? 'Submitting...' : `Run selected (${selectedIds.size})`}</button>
      </div>

      <div className="space-y-3">{discovery.datasets.map((candidate) => <DatasetCard key={candidate.id} candidate={candidate} selected={selectedIds.has(candidate.id)} onToggle={() => toggleCandidate(candidate)} />)}</div>

      {discovery.part_groups.length > 0 && <section className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4"><h4 className="font-semibold text-amber-200">Numbered dataset parts</h4><p className="text-xs text-gray-400 mt-1">Parts are never joined automatically. Merge is available only after each part validates and spatial geometry matches.</p><div className="mt-3 space-y-3">{discovery.part_groups.map((group) => <div key={group.id} className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-md bg-black/20 p-3"><div><p className="text-sm font-medium">{group.label}</p><p className="text-xs text-gray-400">{group.datasets.map((item) => `Part ${item.part_index}: ${item.label}`).join(' · ')}</p></div><button type="button" disabled={!group.can_merge || submitting} onClick={() => mergeAndRun(group)} className="px-3 py-2 rounded-lg text-xs font-semibold bg-amber-600 hover:bg-amber-500 disabled:opacity-50">{group.can_merge ? 'Merge parts and run' : 'Parts need attention'}</button></div>)}</div></section>}
    </section>}
  </div>;
}

function DatasetCard({ candidate, selected, onToggle }: { candidate: UploadedDatasetCandidate; selected: boolean; onToggle: () => void }) {
  return <label className={`block rounded-lg border p-4 transition-colors ${candidate.is_valid ? selected ? 'border-primary-500 bg-primary-500/10' : 'border-gray-700 hover:border-gray-500' : 'border-red-900/70 bg-red-950/20 opacity-90'}`}>
    <div className="flex items-start gap-3"><input type="checkbox" checked={selected} disabled={!candidate.is_valid} onChange={onToggle} className="mt-1 accent-primary-500" /><div className="min-w-0 flex-1"><div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1"><p className="font-medium truncate">{candidate.label}</p><span className={`text-xs font-semibold ${candidate.is_valid ? 'text-green-400' : 'text-red-400'}`}>{candidate.is_valid ? 'READY' : 'NEEDS ATTENTION'}</span></div><p className="text-xs text-gray-400 mt-1 truncate">{candidate.relative_directory} · {candidate.num_volumes ?? 'Unknown'} volumes · {candidate.dimensions?.slice(0, 3).join(' × ') || 'Unknown geometry'}</p><p className="text-xs text-gray-500 mt-2 break-all">{candidate.files.join(' | ')}</p>{candidate.errors.length > 0 && <ul className="mt-2 text-xs text-red-300 list-disc list-inside">{candidate.errors.map((error) => <li key={error}>{error}</li>)}</ul>}{candidate.warnings.length > 0 && <ul className="mt-2 text-xs text-amber-300 list-disc list-inside">{candidate.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}</div></div>
  </label>;
}
