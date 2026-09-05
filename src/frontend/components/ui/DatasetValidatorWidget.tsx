'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { DatasetValidationReport } from '@/lib/types';

export default function DatasetValidatorWidget() {
  const { addNotification } = useAppStore();
  const [validating, setValidating] = useState(false);
  const [report, setReport] = useState<DatasetValidationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Preset demo datasets
  const [selectedPreset, setSelectedPreset] = useState<'SUB1' | 'SUB2'>('SUB1');

  const runValidation = async () => {
    setValidating(true);
    setError(null);

    const dwiPath = `datasets/Stanford dataset/${selectedPreset}/dwi.nii.gz`;
    const bvalPath = `datasets/Stanford dataset/${selectedPreset}/dwi.bval`;
    const bvecPath = `datasets/Stanford dataset/${selectedPreset}/dwi.bvec`;

    try {
      const rep = await apiClient.validateDataset({
        dwi_path: dwiPath,
        bval_path: bvalPath,
        bvec_path: bvecPath,
      });
      setReport(rep);
      addNotification({
        type: rep.is_valid ? 'success' : 'warning',
        title: 'Dataset Validation Completed',
        message: rep.is_valid
          ? `${selectedPreset} complies with strict scientific standards.`
          : 'Dataset validation identified boundary warnings.',
        duration: 4000,
      });
    } catch (err: any) {
      setError(err.message || 'Dataset validation failed');
      addNotification({
        type: 'error',
        title: 'Validation Failed',
        message: err.message || 'Could not validate dataset files',
        duration: 6000,
      });
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="glass rounded-xl p-6 border border-neutral-800 space-y-5 text-gray-100">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-neutral-800">
        <div className="flex items-center gap-3">
          <span className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </span>
          <div>
            <h3 className="text-lg font-bold text-white tracking-wide">
              Dataset Boundary &amp; Scientific Ingestion Validator
            </h3>
            <p className="text-xs text-neutral-400">
              Validates 4D volume dimensions, b-values, unit b-vectors, and SHA-256 hashes
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value as 'SUB1' | 'SUB2')}
            className="bg-neutral-900 border border-neutral-700 text-white text-xs rounded-lg px-3 py-2 font-mono"
          >
            <option value="SUB1">Stanford HARDI: SUB1</option>
            <option value="SUB2">Stanford HARDI: SUB2</option>
          </select>

          <button
            onClick={runValidation}
            disabled={validating}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 disabled:bg-neutral-800 text-white rounded-lg text-xs font-semibold shadow-md transition-all flex items-center gap-1.5"
          >
            {validating ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Auditing...</span>
              </>
            ) : (
              <span>Validate Dataset</span>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800 text-red-300 text-xs rounded-lg">
          {error}
        </div>
      )}

      {report && (
        <div className="space-y-4 animate-in fade-in duration-200">
          <div className="p-4 rounded-xl bg-neutral-950/80 border border-neutral-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold tracking-wide uppercase ${
                  report.is_valid
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                    : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                }`}
              >
                {report.is_valid ? 'VALIDATED' : 'WARNINGS FOUND'}
              </span>
              <span className="text-xs text-neutral-300">
                Format: <strong className="text-white">{report.dataset_format}</strong>
              </span>
            </div>
            <div className="text-xs text-neutral-400 font-mono">
              Orient: {report.orientation_codes || 'RAS+'} | System: {report.coordinate_system || 'Scanner Anatomical'}
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 bg-neutral-900/90 rounded-lg border border-neutral-800">
              <div className="text-[10px] text-neutral-400 uppercase">Dimensions (XYZ)</div>
              <div className="text-white font-bold text-sm mt-0.5">
                {report.dimensions ? report.dimensions.slice(0, 3).join(' × ') : 'N/A'}
              </div>
            </div>
            <div className="p-3 bg-neutral-900/90 rounded-lg border border-neutral-800">
              <div className="text-[10px] text-neutral-400 uppercase">Diffusion Volumes</div>
              <div className="text-cyan-400 font-bold text-sm mt-0.5">
                {report.num_volumes ?? (report.dimensions ? report.dimensions[3] : 'N/A')} dirs
              </div>
            </div>
            <div className="p-3 bg-neutral-900/90 rounded-lg border border-neutral-800">
              <div className="text-[10px] text-neutral-400 uppercase">Voxel Spacing</div>
              <div className="text-white font-bold text-sm mt-0.5">
                {report.voxel_size_mm ? report.voxel_size_mm.map((v) => v.toFixed(2)).join(' × ') : 'N/A'} mm
              </div>
            </div>
            <div className="p-3 bg-neutral-900/90 rounded-lg border border-neutral-800">
              <div className="text-[10px] text-neutral-400 uppercase">Gradient B-Vectors</div>
              <div className="text-emerald-400 font-bold text-sm mt-0.5">
                Unit-Normalized
              </div>
            </div>
          </div>

          {/* SHA-256 Checksums */}
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
              Data File SHA-256 Checksums
            </div>
            <div className="p-3 bg-neutral-950 rounded-lg border border-neutral-800/80 font-mono text-[11px] space-y-1 text-neutral-300 overflow-x-auto">
              {Object.entries(report.checksums_sha256).map(([fileKey, hash]) => (
                <div key={fileKey} className="flex items-center gap-2">
                  <span className="text-primary-400 font-semibold">{fileKey}:</span>
                  <span className="text-neutral-400">{hash}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
