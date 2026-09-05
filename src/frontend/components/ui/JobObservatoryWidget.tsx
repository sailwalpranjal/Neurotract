'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { ExecutionEvent } from '@/lib/types';

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
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

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
          addNotification({
            type: 'success',
            title: 'Job Completed',
            message: `Pipeline execution for ${jobId} finished successfully.`,
            duration: 5000,
          });
          setIsSubscribed(false);
        } else if (evt.event_type === 'job_failed') {
          addNotification({
            type: 'error',
            title: 'Job Failed',
            message: evt.error || 'Pipeline execution encountered an error.',
            duration: 8000,
          });
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

  // Auto-scroll event log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [executionEvents.length]);

  return (
    <div className="glass rounded-xl p-6 border border-neutral-800 space-y-5 text-gray-100">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-neutral-800">
        <div className="flex items-center gap-3">
          <span className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </span>
          <div>
            <h3 className="text-lg font-bold text-white tracking-wide">
              Live Pipeline Execution Observatory
            </h3>
            <p className="text-xs text-neutral-400">
              Real-time Server-Sent Events (SSE) streaming with telemetry and stage timeline
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleTriggerPipeline}
            disabled={isExecuting || isSubscribed}
            className="px-3.5 py-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-950/40 transition-all cursor-pointer disabled:cursor-not-allowed"
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
                <span>Run Live Pipeline ({activeSubject || 'SUB1'})</span>
              </>
            )}
          </button>

          <input
            type="text"
            placeholder="Job ID (e.g. job-abc)"
            value={activeJobId}
            onChange={(e) => setActiveJobId(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono w-32 sm:w-40"
          />
          {isSubscribed ? (
            <button
              onClick={stopSubscription}
              className="px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-xs font-semibold transition-colors"
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={() => startSubscription(activeJobId)}
              disabled={!activeJobId}
              className="px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-200 rounded-lg text-xs font-semibold transition-colors"
            >
              Connect SSE
            </button>
          )}
        </div>
      </div>

      {/* Progress & Stage Status */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-neutral-300 font-mono">
            Current Stage: <strong className="text-primary-400 uppercase">{currentStage}</strong>
          </span>
          <span className="text-emerald-400 font-bold font-mono">
            {progress.toFixed(0)}%
          </span>
        </div>
        <div className="w-full h-2 bg-neutral-900 rounded-full overflow-hidden border border-neutral-800">
          <div
            className="h-full bg-gradient-to-r from-primary-500 via-cyan-400 to-emerald-400 transition-all duration-300"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      </div>

      {/* Event Stream Log */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-neutral-400 font-semibold">
          <span>Execution Event Stream ({executionEvents.length} events)</span>
          {executionEvents.length > 0 && (
            <button
              onClick={clearExecutionEvents}
              className="text-[10px] text-neutral-500 hover:text-neutral-300 lowercase"
            >
              clear log
            </button>
          )}
        </div>

        <div className="h-44 bg-neutral-950 rounded-xl p-3 border border-neutral-800 font-mono text-xs overflow-y-auto space-y-1.5 text-neutral-300">
          {executionEvents.length === 0 ? (
            <div className="h-full flex items-center justify-center text-neutral-600 text-xs font-sans">
              Waiting for execution stream... Connect to a running job to observe real-time telemetry.
            </div>
          ) : (
            executionEvents.map((evt, idx) => (
              <div key={idx} className="flex items-start gap-2 text-[11px] leading-tight">
                <span className="text-neutral-600 whitespace-nowrap">
                  {new Date(evt.timestamp).toLocaleTimeString()}
                </span>
                <span className="text-primary-400 font-bold">[{evt.event_type}]</span>
                <span className="text-neutral-300">{evt.message || ''}</span>
                {evt.telemetry && (
                  <span className="text-cyan-400/90 text-[10px]">
                    {JSON.stringify(evt.telemetry)}
                  </span>
                )}
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  );
}
