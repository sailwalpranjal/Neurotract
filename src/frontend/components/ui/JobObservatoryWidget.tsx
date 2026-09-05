'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '@/lib/store';
import { apiClient } from '@/lib/api';
import { ExecutionEvent } from '@/lib/types';

export default function JobObservatoryWidget() {
  const {
    currentJob,
    executionEvents,
    addExecutionEvent,
    clearExecutionEvents,
    addNotification,
  } = useAppStore();

  const [activeJobId, setActiveJobId] = useState<string>('');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [currentStage, setCurrentStage] = useState<string>('idle');
  const [progress, setProgress] = useState<number>(0);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

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

        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Job ID (e.g. job-abc)"
            value={activeJobId}
            onChange={(e) => setActiveJobId(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono w-36 sm:w-44"
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
              className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-800 disabled:text-neutral-500 text-white rounded-lg text-xs font-semibold transition-colors"
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
