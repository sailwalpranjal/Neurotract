'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';
import { GraphMetrics, ParcellationLabel } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface AdvancedMetricsProps {
  metrics: GraphMetrics;
  labels: ParcellationLabel[];
  connectome: number[][] | null;
}

export default function AdvancedMetrics({ metrics, labels, connectome }: AdvancedMetricsProps) {
  const nodeCount = metrics.nodal.degree.length;

  // Degree distribution histogram data
  const degreeDistribution = useMemo(() => {
    const degrees = metrics.nodal.degree;
    const maxDeg = Math.max(...degrees);
    const binCount = Math.min(20, maxDeg);
    const binSize = maxDeg / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => i * binSize);
    return { values: degrees, bins };
  }, [metrics.nodal.degree]);

  // Inter-hemispheric comparison
  const hemisphereComparison = useMemo(() => {
    const left: number[] = [];
    const right: number[] = [];
    const leftLabels: string[] = [];
    const rightLabels: string[] = [];

    labels.forEach((l, i) => {
      if (l.hemisphere === 'left') {
        left.push(metrics.nodal.degree[i]);
        leftLabels.push(l.abbreviation);
      } else if (l.hemisphere === 'right') {
        right.push(metrics.nodal.degree[i]);
        rightLabels.push(l.abbreviation);
      }
    });

    const leftMean = left.length > 0 ? left.reduce((a, b) => a + b, 0) / left.length : 0;
    const rightMean = right.length > 0 ? right.reduce((a, b) => a + b, 0) / right.length : 0;

    // Also compute strength comparison
    const leftStrength: number[] = [];
    const rightStrength: number[] = [];
    labels.forEach((l, i) => {
      const s = metrics.nodal.node_strength?.[i] || 0;
      if (l.hemisphere === 'left') leftStrength.push(s);
      else if (l.hemisphere === 'right') rightStrength.push(s);
    });

    const leftStrMean = leftStrength.length > 0 ? leftStrength.reduce((a, b) => a + b, 0) / leftStrength.length : 0;
    const rightStrMean = rightStrength.length > 0 ? rightStrength.reduce((a, b) => a + b, 0) / rightStrength.length : 0;

    return {
      left, right, leftLabels, rightLabels,
      leftMean, rightMean,
      leftStrMean, rightStrMean,
      leftStrength, rightStrength,
    };
  }, [metrics, labels]);

  // Z-scores against normative ranges
  const zScores = useMemo(() => {
    const norms: Record<string, { mean: number; sd: number }> = {
      clustering_coefficient: { mean: 0.40, sd: 0.10 },
      characteristic_path_length: { mean: 3.0, sd: 0.5 },
      global_efficiency: { mean: 0.50, sd: 0.10 },
      modularity: { mean: 0.35, sd: 0.08 },
      small_worldness: { mean: 2.0, sd: 0.5 },
      assortativity: { mean: 0.05, sd: 0.15 },
    };

    const g = metrics.global;
    return Object.entries(norms).map(([key, norm]) => {
      const val = g[key as keyof typeof g] as number | undefined;
      if (val === undefined) return null;
      const z = (val - norm.mean) / norm.sd;
      return {
        metric: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        value: val,
        zScore: z,
        mean: norm.mean,
        sd: norm.sd,
        significance: Math.abs(z) > 2 ? 'significant' : Math.abs(z) > 1 ? 'marginal' : 'normal',
      };
    }).filter(Boolean) as { metric: string; value: number; zScore: number; mean: number; sd: number; significance: string }[];
  }, [metrics.global]);

  // Lobe-wise aggregate metrics
  const lobeMetrics = useMemo(() => {
    const lobes: Record<string, { degrees: number[]; strengths: number[]; betweenness: number[]; count: number }> = {};

    labels.forEach((l, i) => {
      const lobe = l.lobe || 'unknown';
      if (!lobes[lobe]) lobes[lobe] = { degrees: [], strengths: [], betweenness: [], count: 0 };
      lobes[lobe].degrees.push(metrics.nodal.degree[i]);
      lobes[lobe].strengths.push(metrics.nodal.node_strength?.[i] || 0);
      lobes[lobe].betweenness.push(metrics.nodal.betweenness_centrality[i]);
      lobes[lobe].count++;
    });

    return Object.entries(lobes)
      .filter(([name]) => name !== 'unknown')
      .map(([name, data]) => ({
        name,
        count: data.count,
        avgDegree: data.degrees.reduce((a, b) => a + b, 0) / data.count,
        avgStrength: data.strengths.reduce((a, b) => a + b, 0) / data.count,
        avgBetweenness: data.betweenness.reduce((a, b) => a + b, 0) / data.count,
        maxDegree: Math.max(...data.degrees),
      }))
      .sort((a, b) => b.avgDegree - a.avgDegree);
  }, [metrics, labels]);

  // Hub vulnerability index (regions with highest betweenness - removing them would disconnect the network)
  const vulnerabilityIndex = useMemo(() => {
    return metrics.nodal.betweenness_centrality
      .map((bc, i) => ({
        index: i,
        label: labels[i]?.anatomical_name || `Region ${i}`,
        abbreviation: labels[i]?.abbreviation || `R${i}`,
        betweenness: bc,
        degree: metrics.nodal.degree[i],
        strength: metrics.nodal.node_strength?.[i] || 0,
        vulnerability: bc * metrics.nodal.degree[i], // combined score
      }))
      .sort((a, b) => b.vulnerability - a.vulnerability)
      .slice(0, 15);
  }, [metrics, labels]);

  const chartLayout = {
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#fff', size: 10 },
    autosize: true,
  };

  return (
    <div className="space-y-8">
      {/* Z-Score Analysis */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-2">Normative Z-Score Analysis</h2>
        <p className="text-gray-400 text-sm mb-4">
          Global metrics compared to normative values from healthy structural connectome literature.
          Z-scores beyond +/-2 indicate significant deviation.
        </p>
        <div className="plotly-container">
          <Plot
            data={[{
              x: zScores.map(z => z.metric),
              y: zScores.map(z => z.zScore),
              type: 'bar',
              marker: {
                color: zScores.map(z =>
                  z.significance === 'significant' ? '#ef4444' :
                  z.significance === 'marginal' ? '#f59e0b' : '#10b981'
                ),
              },
              text: zScores.map(z => `Z = ${z.zScore.toFixed(2)}`),
              textposition: 'outside',
            }]}
            layout={{
              ...chartLayout,
              margin: { l: 50, r: 20, t: 20, b: 100 },
              yaxis: { title: 'Z-Score', zeroline: true, zerolinecolor: '#666' },
              xaxis: { tickangle: -30 },
              shapes: [
                { type: 'line', y0: 2, y1: 2, x0: -0.5, x1: zScores.length - 0.5, line: { color: '#ef4444', dash: 'dash', width: 1 } },
                { type: 'line', y0: -2, y1: -2, x0: -0.5, x1: zScores.length - 0.5, line: { color: '#ef4444', dash: 'dash', width: 1 } },
              ],
            }}
            useResizeHandler
            style={{ width: '100%', height: '350px' }}
            config={{ responsive: true }}
          />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
          {zScores.map((z) => (
            <div key={z.metric} className={`p-3 rounded-lg border text-sm ${
              z.significance === 'significant' ? 'border-red-500/30 bg-red-500/5' :
              z.significance === 'marginal' ? 'border-yellow-500/30 bg-yellow-500/5' :
              'border-green-500/30 bg-green-500/5'
            }`}>
              <div className="text-gray-400 text-xs">{z.metric}</div>
              <div className="font-semibold">{z.value.toFixed(4)}</div>
              <div className="text-xs text-gray-500">
                Norm: {z.mean.toFixed(2)} +/- {z.sd.toFixed(2)} | Z = {z.zScore.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Degree Distribution */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-2">Degree Distribution</h2>
        <p className="text-gray-400 text-sm mb-4">
          Distribution of nodal degrees. Brain networks typically follow a truncated power-law or exponential distribution.
        </p>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="plotly-container">
            <Plot
              data={[{
                x: degreeDistribution.values,
                type: 'histogram',
                marker: { color: '#0ea5e9' },
                nbinsx: 20,
                name: 'Count',
              }]}
              layout={{
                ...chartLayout,
                margin: { l: 50, r: 20, t: 30, b: 40 },
                title: 'Degree Histogram',
                xaxis: { title: 'Degree' },
                yaxis: { title: 'Count' },
              }}
              useResizeHandler
              style={{ width: '100%', height: '300px' }}
              config={{ responsive: true }}
            />
          </div>
          <div className="plotly-container">
            <Plot
              data={[{
                x: metrics.nodal.degree,
                y: metrics.nodal.betweenness_centrality,
                mode: 'markers',
                type: 'scatter',
                marker: {
                  size: (metrics.nodal.node_strength || metrics.nodal.degree).map(s => Math.max(4, Math.min(20, s / 5))),
                  color: metrics.nodal.closeness_centrality,
                  colorscale: 'Viridis',
                  showscale: true,
                  colorbar: { title: 'Closeness', len: 0.5 },
                },
                text: labels.map(l => l?.abbreviation || ''),
                hovertemplate: '%{text}<br>Degree: %{x}<br>Betweenness: %{y:.4f}<extra></extra>',
              }]}
              layout={{
                ...chartLayout,
                margin: { l: 50, r: 60, t: 30, b: 40 },
                title: 'Degree vs Betweenness',
                xaxis: { title: 'Degree' },
                yaxis: { title: 'Betweenness Centrality' },
              }}
              useResizeHandler
              style={{ width: '100%', height: '300px' }}
              config={{ responsive: true }}
            />
          </div>
        </div>
      </div>

      {/* Inter-hemispheric Comparison */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-2">Inter-Hemispheric Comparison</h2>
        <p className="text-gray-400 text-sm mb-4">
          Comparison of left vs right hemisphere connectivity. Asymmetry may indicate lateralization patterns or pathology.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatBox label="Left Mean Degree" value={hemisphereComparison.leftMean.toFixed(2)} />
          <StatBox label="Right Mean Degree" value={hemisphereComparison.rightMean.toFixed(2)} />
          <StatBox label="Left Mean Strength" value={hemisphereComparison.leftStrMean.toFixed(2)} />
          <StatBox label="Right Mean Strength" value={hemisphereComparison.rightStrMean.toFixed(2)} />
        </div>
        <div className="plotly-container">
          <Plot
            data={[
              {
                y: hemisphereComparison.left,
                name: 'Left Hemisphere',
                type: 'box',
                marker: { color: '#0ea5e9' },
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.5,
              },
              {
                y: hemisphereComparison.right,
                name: 'Right Hemisphere',
                type: 'box',
                marker: { color: '#e94560' },
                boxpoints: 'all',
                jitter: 0.3,
                pointpos: -1.5,
              },
            ]}
            layout={{
              ...chartLayout,
              margin: { l: 50, r: 20, t: 30, b: 40 },
              title: 'Degree Distribution by Hemisphere',
              yaxis: { title: 'Degree' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '350px' }}
            config={{ responsive: true }}
          />
        </div>
      </div>

      {/* Lobe-wise Analysis */}
      {lobeMetrics.length > 0 && (
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-2">Lobe-wise Connectivity</h2>
          <p className="text-gray-400 text-sm mb-4">
            Aggregate connectivity metrics by cortical lobe.
          </p>
          <div className="plotly-container">
            <Plot
              data={[
                {
                  x: lobeMetrics.map(l => l.name.charAt(0).toUpperCase() + l.name.slice(1)),
                  y: lobeMetrics.map(l => l.avgDegree),
                  name: 'Avg Degree',
                  type: 'bar',
                  marker: { color: '#0ea5e9' },
                },
                {
                  x: lobeMetrics.map(l => l.name.charAt(0).toUpperCase() + l.name.slice(1)),
                  y: lobeMetrics.map(l => l.avgStrength),
                  name: 'Avg Strength',
                  type: 'bar',
                  marker: { color: '#8b5cf6' },
                },
              ]}
              layout={{
                ...chartLayout,
                margin: { l: 50, r: 20, t: 30, b: 60 },
                title: 'Average Metrics by Lobe',
                barmode: 'group',
                yaxis: { title: 'Value' },
              }}
              useResizeHandler
              style={{ width: '100%', height: '350px' }}
              config={{ responsive: true }}
            />
          </div>
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10 text-left text-gray-400">
                  <th className="pb-2 pr-4">Lobe</th>
                  <th className="pb-2 pr-4">Regions</th>
                  <th className="pb-2 pr-4">Avg Degree</th>
                  <th className="pb-2 pr-4">Max Degree</th>
                  <th className="pb-2 pr-4">Avg Strength</th>
                  <th className="pb-2">Avg Betweenness</th>
                </tr>
              </thead>
              <tbody>
                {lobeMetrics.map((l) => (
                  <tr key={l.name} className="border-b border-white/5">
                    <td className="py-2 pr-4 font-medium">{l.name.charAt(0).toUpperCase() + l.name.slice(1)}</td>
                    <td className="py-2 pr-4 text-gray-400">{l.count}</td>
                    <td className="py-2 pr-4">{l.avgDegree.toFixed(2)}</td>
                    <td className="py-2 pr-4">{l.maxDegree}</td>
                    <td className="py-2 pr-4">{l.avgStrength.toFixed(2)}</td>
                    <td className="py-2">{l.avgBetweenness.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Vulnerability Index */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-2">Network Vulnerability Index</h2>
        <p className="text-gray-400 text-sm mb-4">
          Regions ranked by vulnerability score (betweenness x degree). High-vulnerability hubs are critical
          for network integrity — their disruption would most impact information flow.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-gray-400">
                <th className="pb-2 pr-3">Rank</th>
                <th className="pb-2 pr-3">Region</th>
                <th className="pb-2 pr-3">Abbrev</th>
                <th className="pb-2 pr-3">Degree</th>
                <th className="pb-2 pr-3">Betweenness</th>
                <th className="pb-2 pr-3">Strength</th>
                <th className="pb-2">Vulnerability</th>
              </tr>
            </thead>
            <tbody>
              {vulnerabilityIndex.map((v, i) => (
                <tr key={v.index} className="border-b border-white/5">
                  <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
                  <td className="py-2 pr-3 font-medium max-w-[200px] truncate" title={v.label}>{v.label}</td>
                  <td className="py-2 pr-3 text-primary-400">{v.abbreviation}</td>
                  <td className="py-2 pr-3">{v.degree}</td>
                  <td className="py-2 pr-3">{v.betweenness.toFixed(4)}</td>
                  <td className="py-2 pr-3">{v.strength.toFixed(2)}</td>
                  <td className="py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-500 rounded-full"
                          style={{ width: `${(v.vulnerability / vulnerabilityIndex[0].vulnerability) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400">{v.vulnerability.toFixed(2)}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Rich Club Coefficients */}
      {(metrics.rich_club?.coefficients?.length ?? 0) > 0 && (metrics.rich_club?.k_values?.length ?? 0) > 0 && (
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-2">Rich Club Analysis</h2>
          <p className="text-gray-400 text-sm mb-4">
            Rich club coefficient as a function of degree threshold k. Values above the
            normalized threshold indicate hub regions preferentially interconnect.
          </p>
          <div className="plotly-container">
            <Plot
              data={[{
                x: metrics.rich_club!.k_values,
                y: metrics.rich_club!.coefficients,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: '#f59e0b', size: 5 },
                line: { color: '#f59e0b', width: 2 },
                name: 'Rich Club Coefficient',
              }]}
              layout={{
                ...chartLayout,
                margin: { l: 50, r: 20, t: 30, b: 40 },
                title: 'Rich Club Curve',
                xaxis: { title: 'Degree Threshold (k)' },
                yaxis: { title: 'Rich Club Coefficient' },
              }}
              useResizeHandler
              style={{ width: '100%', height: '350px' }}
              config={{ responsive: true }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white/5 rounded-lg p-3 text-center">
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-lg font-semibold mt-1">{value}</div>
    </div>
  );
}
