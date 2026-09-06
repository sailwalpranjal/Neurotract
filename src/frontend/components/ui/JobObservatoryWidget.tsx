'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import Link from 'next/link';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { ExecutionEvent } from '@/lib/types';

interface StageDefinition {
  id: string;
  name: string;
  shortName: string;
  description: string;
  minProgress: number;
}

const PIPELINE_STAGES: StageDefinition[] = [
  { id: 'validation', name: 'Dataset Validation', shortName: 'Validation', description: 'File integrity, shells, and dimensions', minProgress: 0.05 },
  { id: 'preprocessing', name: 'DWI Preprocessing', shortName: 'Preproc', description: 'Brain extraction and gradient corrections', minProgress: 0.20 },
  { id: 'dti', name: 'DTI Model Fitting', shortName: 'DTI', description: 'Weighted least-squares tensor estimation (FA/MD)', minProgress: 0.40 },
  { id: 'csd', name: 'CSD & FOD Modeling', shortName: 'CSD', description: 'Constrained spherical deconvolution for crossings', minProgress: 0.55 },
  { id: 'tractography', name: 'Streamline Tracking', shortName: 'Tracts', description: 'Probabilistic directional tracking and TRK export', minProgress: 0.68 },
  { id: 'brain_mesh', name: 'Surface Reconstruction', shortName: 'Surface', description: 'Marching cubes cortical brain mesh', minProgress: 0.88 },
  { id: 'connectome', name: 'Connectome Mapping', shortName: 'Connectome', description: 'Endpoint parcellation assignment and adjacency', minProgress: 0.92 },
  { id: 'graph_metrics', name: 'Graph Theory Metrics', shortName: 'Network', description: 'Global efficiency, modularity, and degree', minProgress: 0.96 },
  { id: 'reporting', name: 'Report Generation', shortName: 'Report', description: 'Standalone HTML provenance audit report', minProgress: 0.98 },
  { id: 'completed', name: 'Analysis Complete', shortName: 'Ready', description: 'Artifacts verified and ready for exploration', minProgress: 1.0 },
];

export default function JobObservatoryWidget() {
  const {
    activeSubject,
    currentJob,
    setCurrentJob,
    executionEvents,
    addExecutionEvent,
    clearExecutionEvents,
    addNotification,
  } = useAppStore();

  const [activeJobId, setActiveJobId] = useState<string>('');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentStage, setCurrentStage] = useState<string>('idle');
  const [progress, setProgress] = useState<number>(0);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [logFilter, setLogFilter] = useState<'all' | 'milestones' | 'telemetry' | 'errors'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [copiedLogs, setCopiedLogs] = useState<boolean>(false);
  const [isLogExpanded, setIsLogExpanded] = useState<boolean>(true);

  const unsubscribeRef = useRef<(() => void) | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const handleTriggerPipeline = async () => {
    setIsExecuting(true);
    try {
      const subject = activeSubject || 'SUB1';
      addNotification({
        type: 'info',
        title: 'Submitting Analysis Job',
        message: `Initiating real-time diffusion MRI pipeline for ${subject}...`,
        duration: 3000,
      });
      const job = await apiClient.submitJob({
        subject_id: subject,
        mode: 'quick',
      });
      setActiveJobId(job.id);
      setCurrentJob(job);
      startSubscription(job.id);
      addNotification({
        type: 'success',
        title: 'Job Submitted',
        message: `Job ${job.id.slice(0, 8)} started. Streaming live SSE events...`,
        duration: 4000,
      });
    } catch (err: any) {
      addNotification({
        type: 'error',
        title: 'Submission Failed',
        message: err.message || 'Could not submit pipeline job',
        duration: 6000,
      });
    } finally {
      setIsExecuting(false);
    }
  };

  // Auto-subscribe if currentJob changes
  useEffect(() => {
    if (currentJob?.id && currentJob.id !== activeJobId) {
      setActiveJobId(currentJob.id);
      startSubscription(currentJob.id);
    }
  }, [currentJob?.id]);

  const startSubscription = (jobId: string) => {
    if (!jobId) return;
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
    }

    clearExecutionEvents();
    setIsSubscribed(true);

    const unsub = apiClient.subscribeJobEvents(
      jobId,
      (evt: ExecutionEvent) => {
        addExecutionEvent(evt);
        if (evt.stage) setCurrentStage(evt.stage);
        if (evt.progress !== undefined) setProgress(evt.progress);

        if (evt.event_type === 'job_completed') {
          setIsSubscribed(false);
        } else if (evt.event_type === 'job_failed') {
          setIsSubscribed(false);
        }
      },
      () => {
        setIsSubscribed(false);
      }
    );

    unsubscribeRef.current = unsub;
  };

  const stopSubscription = () => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
    setIsSubscribed(false);
  };

  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
    };
  }, []);

  // Bounded auto-scroll that NEVER touches the parent window
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [executionEvents.length, autoScroll]);

  const handleLogContainerScroll = () => {
    if (!logContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 24;
    if (isAtBottom !== autoScroll) {
      setAutoScroll(isAtBottom);
    }
  };

  // Extract stage telemetry summaries from events
  const stageTelemetry = useMemo(() => {
    const map: Record<string, Record<string, any>> = {};
    for (const evt of executionEvents) {
      if (evt.stage && evt.telemetry && Object.keys(evt.telemetry).length > 0) {
        map[evt.stage] = { ...(map[evt.stage] || {}), ...evt.telemetry };
      }
    }
    return map;
  }, [executionEvents]);

  // Stage status determination
  const getStageStatus = (stage: StageDefinition, index: number) => {
    if (progress >= 1.0 || currentStage === 'completed') return 'completed';
    const activeStageIndex = PIPELINE_STAGES.findIndex((s) => s.id === currentStage);

    if (activeStageIndex === -1) {
      if (progress >= stage.minProgress) return 'completed';
      return 'pending';
    }

    if (index < activeStageIndex) return 'completed';
    if (index === activeStageIndex) return 'running';
    return 'pending';
  };

  // Filtered log events
  const filteredEvents = useMemo(() => {
    return executionEvents.filter((evt) => {
      if (logFilter === 'milestones' && !evt.event_type.startsWith('stage_') && evt.event_type !== 'job_completed') {
        return false;
      }
      if (logFilter === 'telemetry' && (!evt.telemetry || Object.keys(evt.telemetry).length === 0)) {
        return false;
      }
      if (logFilter === 'errors' && evt.event_type !== 'job_failed' && !evt.error) {
        return false;
      }
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const text = `${evt.event_type} ${evt.stage || ''} ${evt.message || ''} ${JSON.stringify(evt.telemetry || {})}`.toLowerCase();
        if (!text.includes(q)) return false;
      }
      return true;
    });
  }, [executionEvents, logFilter, searchQuery]);

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(JSON.stringify(executionEvents, null, 2));
    setCopiedLogs(true);
    setTimeout(() => setCopiedLogs(false), 2000);
  };

  const isCompleted = progress >= 1.0 || currentStage === 'completed';

  return (
    <div className="bg-slate-900/95 backdrop-blur-md rounded-xl p-5 border border-slate-800 space-y-5 text-slate-100 shadow-xl">
      {/* 1. Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <span className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </span>
          <div>
            <h3 className="text-base font-semibold text-slate-100 tracking-wide flex items-center gap-2">
              Real-Time Pipeline Execution Engine
              {isSubscribed && (
                <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-800/60">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  LIVE SSE
                </span>
              )}
            </h3>
            <p className="text-xs text-slate-400">
              10-stage scientific processing with live telemetry and reproducible provenance
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleTriggerPipeline}
            disabled={isExecuting || isSubscribed}
            className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-sm transition-all cursor-pointer disabled:cursor-not-allowed"
          >
            {isExecuting ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Launching...</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Execute Pipeline ({activeSubject || 'SUB1'})</span>
              </>
            )}
          </button>

          <input
            type="text"
            placeholder="Job ID (e.g. 737f7f...)"
            value={activeJobId}
            onChange={(e) => setActiveJobId(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono w-32 sm:w-44 focus:border-slate-700 outline-none"
          />
          {isSubscribed ? (
            <button
              onClick={stopSubscription}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition-colors"
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={() => startSubscription(activeJobId)}
              disabled={!activeJobId}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 rounded-lg text-xs font-semibold transition-colors"
            >
              Connect SSE
            </button>
          )}
        </div>
      </div>

      {/* 2. Visual 10-Stage Pipeline DAG Stepper */}
      <div className="space-y-3 bg-slate-950 p-4 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 font-mono flex items-center gap-2">
            Active Stage: <strong className="text-emerald-400 uppercase">{currentStage}</strong>
          </span>
          <span className="text-emerald-400 font-bold font-mono text-sm">
            {(progress * 100).toFixed(0)}%
          </span>
        </div>

        {/* Global Progress Bar */}
        <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-emerald-600 via-teal-500 to-cyan-400 transition-all duration-300"
            style={{ width: `${Math.min(100, Math.max(0, progress * 100))}%` }}
          />
        </div>

        {/* 10-Stage Stepper Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2 pt-2">
          {PIPELINE_STAGES.map((stage, idx) => {
            const status = getStageStatus(stage, idx);
            return (
              <div
                key={stage.id}
                className={`p-2 rounded-lg border text-center transition-all ${
                  status === 'completed'
                    ? 'bg-emerald-950/20 border-emerald-800/50 text-emerald-300'
                    : status === 'running'
                    ? 'bg-slate-800 border-emerald-500 text-slate-100 shadow-md ring-1 ring-emerald-500/30'
                    : 'bg-slate-900/50 border-slate-800 text-slate-500'
                }`}
              >
                <div className="flex items-center justify-center gap-1 text-[11px] font-semibold font-mono">
                  {status === 'completed' && (
                    <svg className="w-3 h-3 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {status === 'running' && (
                    <svg className="w-3 h-3 text-emerald-400 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                  )}
                  <span className="truncate">{stage.shortName}</span>
                </div>
                <p className="text-[9px] text-slate-400 truncate mt-0.5">
                  {status === 'completed' ? 'Done' : status === 'running' ? 'Active' : 'Pending'}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. Stage Key Scientific Telemetry Cards */}
      {Object.keys(stageTelemetry).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
          {stageTelemetry.validation && (
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Dataset Ingestion</span>
              <div className="text-slate-200 font-semibold mt-0.5">
                {stageTelemetry.validation.num_volumes} vols
              </div>
              <div className="text-[11px] text-emerald-400">
                {stageTelemetry.validation.voxel_size?.map((v: number) => v.toFixed(1)).join('×')} mm iso
              </div>
            </div>
          )}
          {stageTelemetry.dti && (
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">DTI Microstructure</span>
              <div className="text-slate-200 font-semibold mt-0.5">
                FA: {stageTelemetry.dti.mean_fa_brain}
              </div>
              <div className="text-[11px] text-emerald-400">
                MD: {stageTelemetry.dti.mean_md_brain}
              </div>
            </div>
          )}
          {stageTelemetry.tractography && (
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Tractography</span>
              <div className="text-slate-200 font-semibold mt-0.5">
                {stageTelemetry.tractography.total_seeds?.toLocaleString()} seeds
              </div>
              <div className="text-[11px] text-cyan-400">
                Step: {stageTelemetry.tractography.step_size_mm} mm
              </div>
            </div>
          )}
          {stageTelemetry.connectome && (
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Connectome Graph</span>
              <div className="text-slate-200 font-semibold mt-0.5">
                Structural Edges
              </div>
              <div className="text-[11px] text-amber-400">
                Desikan-Killiany 89
              </div>
            </div>
          )}
        </div>
      )}

      {/* 4. Embedded Completion Banner (Non-disruptive) */}
      {isCompleted && (
        <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-800/60 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            <div>
              <h4 className="text-sm font-semibold text-emerald-300">
                Pipeline Finished Successfully
              </h4>
              <p className="text-xs text-slate-300">
                Outputs, surface meshes, streamlines, and graph metrics are fully verified and loaded.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/viewer"
              className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold transition-colors flex items-center gap-1"
            >
              3D & MPR Slices
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            <Link
              href="/analysis"
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold transition-colors"
            >
              Analyze Connectome
            </Link>
          </div>
        </div>
      )}

      {/* 5. Collapsible Technical Event Console */}
      <div className="space-y-2 border-t border-slate-800 pt-3">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <button
            onClick={() => setIsLogExpanded(!isLogExpanded)}
            className="flex items-center gap-1.5 font-semibold text-slate-300 hover:text-white transition-colors"
          >
            <svg
              className={`w-4 h-4 transform transition-transform ${isLogExpanded ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
            </svg>
            <span>Technical Event Stream ({executionEvents.length} events)</span>
          </button>

          {isLogExpanded && (
            <div className="flex items-center gap-2">
              {/* Filter Tabs */}
              <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[11px]">
                <button
                  onClick={() => setLogFilter('all')}
                  className={`px-2 py-0.5 rounded ${logFilter === 'all' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                >
                  All
                </button>
                <button
                  onClick={() => setLogFilter('milestones')}
                  className={`px-2 py-0.5 rounded ${logFilter === 'milestones' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                >
                  Stages
                </button>
                <button
                  onClick={() => setLogFilter('telemetry')}
                  className={`px-2 py-0.5 rounded ${logFilter === 'telemetry' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                >
                  Telemetry
                </button>
              </div>

              {/* Auto-scroll toggle */}
              <label className="flex items-center gap-1.5 text-[11px] text-slate-400 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="rounded border-slate-700 accent-emerald-500"
                />
                <span>Auto-scroll</span>
              </label>

              {/* Copy logs */}
              {executionEvents.length > 0 && (
                <button
                  onClick={handleCopyLogs}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 rounded transition-colors"
                >
                  {copiedLogs ? 'Copied!' : 'Copy JSON'}
                </button>
              )}

              {/* Clear */}
              {executionEvents.length > 0 && (
                <button
                  onClick={clearExecutionEvents}
                  className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>

        {isLogExpanded && (
          <div
            ref={logContainerRef}
            onScroll={handleLogContainerScroll}
            className="h-48 bg-slate-950 rounded-xl p-3 border border-slate-800 font-mono text-xs overflow-y-auto space-y-1.5 text-slate-300"
          >
            {filteredEvents.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-600 text-xs font-sans">
                {executionEvents.length === 0
                  ? 'Waiting for execution stream... Trigger pipeline to start real-time telemetry.'
                  : 'No events matching current filter.'}
              </div>
            ) : (
              filteredEvents.map((evt, idx) => (
                <div key={idx} className="flex items-start gap-2 text-[11px] leading-tight">
                  <span className="text-slate-600 whitespace-nowrap">
                    {new Date(evt.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`font-semibold ${
                    evt.event_type.startsWith('stage_')
                      ? 'text-emerald-400'
                      : evt.event_type === 'job_completed'
                      ? 'text-cyan-400 font-bold'
                      : evt.event_type === 'job_failed'
                      ? 'text-red-400 font-bold'
                      : 'text-slate-400'
                  }`}>
                    [{evt.event_type}]
                  </span>
                  <span className="text-slate-200">{evt.message || ''}</span>
                  {evt.telemetry && Object.keys(evt.telemetry).length > 0 && (
                    <span className="text-cyan-400/80 text-[10px] font-mono">
                      {JSON.stringify(evt.telemetry)}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

