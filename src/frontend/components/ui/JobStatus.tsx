'use client';

import { useEffect, useState } from 'react';
import { Job } from '@/lib/types';
import { formatDate } from '@/lib/utils';
import { apiClient, pollJobUntilComplete } from '@/lib/api';
import { useAppStore } from '@/lib/store';

interface JobStatusProps {
  job: Job;
  compact?: boolean;
}

export default function JobStatus({ job, compact = false }: JobStatusProps) {
  const { updateJob } = useAppStore();
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    if (job.status === 'pending' || job.status === 'running') {
      if (!polling) {
        setPolling(true);
        pollJobUntilComplete(
          job.id,
          (updatedJob) => {
            updateJob(job.id, updatedJob);
          },
          2000
        )
          .then((completedJob) => {
            updateJob(job.id, completedJob);
          })
          .catch((error) => {
            console.error('Job polling error:', error);
            updateJob(job.id, {
              status: 'failed',
              error: error.message,
            });
          })
          .finally(() => {
            setPolling(false);
          });
      }
    }
  }, [job.id, job.status, polling, updateJob]);

  const statusColor = {
    pending: 'text-yellow-500',
    running: 'text-blue-500',
    completed: 'text-green-500',
    failed: 'text-red-500',
  }[job.status];

  const statusBg = {
    pending: 'bg-yellow-500/20 border-yellow-500/50',
    running: 'bg-blue-500/20 border-blue-500/50',
    completed: 'bg-green-500/20 border-green-500/50',
    failed: 'bg-red-500/20 border-red-500/50',
  }[job.status];

  if (compact) {
    return (
      <div className={`${statusBg} border rounded-lg p-3`}>
        <div className="flex items-center justify-between mb-2">
          <span className={`text-sm font-semibold ${statusColor}`}>
            {job.status.toUpperCase()}
          </span>
          <span className="text-xs text-gray-400">
            {formatDate(job.updated_at)}
          </span>
        </div>

        {job.status === 'running' && (
          <div className="space-y-2">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${job.progress}%` }}
              />
            </div>
            <p className="text-xs text-gray-400">{job.progress}% complete</p>
          </div>
        )}

        {job.status === 'failed' && job.error && (
          <p className="text-xs text-red-400 mt-2">{job.error}</p>
        )}
      </div>
    );
  }

  return (
    <div className={`${statusBg} border rounded-lg p-6`}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold mb-1">{job.task}</h3>
          <p className="text-sm text-gray-400">Job ID: {job.id}</p>
        </div>
        <StatusBadge status={job.status} />
      </div>

      {/* Progress */}
      {job.status === 'running' && (
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2">
            <span>Progress</span>
            <span>{job.progress}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-300 flex items-center justify-end pr-2"
              style={{ width: `${job.progress}%` }}
            >
              {job.progress > 10 && (
                <span className="text-xs font-semibold">{job.progress}%</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-400">Created</p>
          <p>{formatDate(job.created_at)}</p>
        </div>
        <div>
          <p className="text-gray-400">Updated</p>
          <p>{formatDate(job.updated_at)}</p>
        </div>
      </div>

      {/* Error */}
      {job.status === 'failed' && job.error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/50 rounded">
          <p className="text-sm font-semibold text-red-400 mb-1">Error</p>
          <p className="text-sm text-gray-300">{job.error}</p>
        </div>
      )}

      {/* Actions */}
      <div className="mt-4 flex gap-2">
        {(job.status === 'pending' || job.status === 'running') && (
          <button
            onClick={() => cancelJob(job.id)}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm transition-colors"
          >
            Cancel Job
          </button>
        )}
        <button
          onClick={() => refreshJob(job.id)}
          className="px-4 py-2 glass hover:bg-white/10 rounded-lg text-sm transition-colors"
        >
          Refresh
        </button>
      </div>
    </div>
  );

  async function cancelJob(jobId: string) {
    try {
      await apiClient.cancelJob(jobId);
      updateJob(jobId, { status: 'failed', error: 'Cancelled by user' });
    } catch (error: any) {
      console.error('Cancel failed:', error);
    }
  }

  async function refreshJob(jobId: string) {
    try {
      const updatedJob = await apiClient.getJobStatus(jobId);
      updateJob(jobId, updatedJob);
    } catch (error: any) {
      console.error('Refresh failed:', error);
    }
  }
}

function StatusBadge({ status }: { status: Job['status'] }) {
  const config = {
    pending: { label: 'Pending', icon: '⏳', color: 'bg-yellow-500' },
    running: { label: 'Running', icon: '▶️', color: 'bg-blue-500' },
    completed: { label: 'Completed', icon: '✓', color: 'bg-green-500' },
    failed: { label: 'Failed', icon: '✗', color: 'bg-red-500' },
  }[status];

  return (
    <div className={`${config.color}/20 border border-${config.color}/50 rounded-full px-3 py-1 flex items-center space-x-2`}>
      <span>{config.icon}</span>
      <span className="text-sm font-semibold">{config.label}</span>
    </div>
  );
}
