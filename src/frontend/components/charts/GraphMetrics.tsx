'use client';

import dynamic from 'next/dynamic';
import { GraphMetrics, ParcellationLabel } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface GraphMetricsProps {
  metrics: GraphMetrics;
  labels?: ParcellationLabel[];
}

export default function GraphMetricsChart({ metrics, labels }: GraphMetricsProps) {
  if (!metrics.nodal) return null;

  const nodeCount = metrics.nodal.degree.length;
  const nodeLabels = labels && labels.length === nodeCount
    ? labels.map((l) => l.abbreviation)
    : Array.from({ length: nodeCount }, (_, i) => `R${i + 1}`);

  const chartLayout = (title: string) => ({
    title,
    xaxis: { title: 'Brain Region', tickangle: -45, tickfont: { size: 8 } },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#fff', size: 10 },
    autosize: true,
    margin: { l: 50, r: 20, t: 40, b: 80 },
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      {/* Degree Distribution */}
      <div className="plotly-container">
        <Plot
          data={[{
            x: nodeLabels,
            y: metrics.nodal.degree,
            type: 'bar',
            marker: { color: '#0ea5e9' },
            name: 'Degree',
          }]}
          layout={{ ...chartLayout('Nodal Degree'), yaxis: { title: 'Degree' } }}
          useResizeHandler
          style={{ width: '100%', height: '300px' }}
          config={{ responsive: true }}
        />
      </div>

      {/* Betweenness Centrality */}
      <div className="plotly-container">
        <Plot
          data={[{
            x: nodeLabels,
            y: metrics.nodal.betweenness_centrality,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#e94560', size: 4 },
            line: { color: '#e94560', width: 1.5 },
            name: 'Betweenness',
          }]}
          layout={{ ...chartLayout('Betweenness Centrality'), yaxis: { title: 'Centrality' } }}
          useResizeHandler
          style={{ width: '100%', height: '300px' }}
          config={{ responsive: true }}
        />
      </div>

      {/* Closeness Centrality */}
      <div className="plotly-container">
        <Plot
          data={[{
            x: nodeLabels,
            y: metrics.nodal.closeness_centrality,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#10b981', size: 4 },
            line: { color: '#10b981', width: 1.5 },
            name: 'Closeness',
          }]}
          layout={{ ...chartLayout('Closeness Centrality'), yaxis: { title: 'Centrality' } }}
          useResizeHandler
          style={{ width: '100%', height: '300px' }}
          config={{ responsive: true }}
        />
      </div>

      {/* Local Efficiency */}
      <div className="plotly-container">
        <Plot
          data={[{
            x: nodeLabels,
            y: metrics.nodal.local_efficiency,
            type: 'bar',
            marker: { color: '#f59e0b' },
            name: 'Local Efficiency',
          }]}
          layout={{ ...chartLayout('Local Efficiency'), yaxis: { title: 'Efficiency' } }}
          useResizeHandler
          style={{ width: '100%', height: '300px' }}
          config={{ responsive: true }}
        />
      </div>

      {/* Node Strength */}
      {metrics.nodal.node_strength && metrics.nodal.node_strength.length > 0 && (
        <div className="plotly-container">
          <Plot
            data={[{
              x: nodeLabels,
              y: metrics.nodal.node_strength,
              type: 'bar',
              marker: { color: '#8b5cf6' },
              name: 'Strength',
            }]}
            layout={{ ...chartLayout('Node Strength'), yaxis: { title: 'Strength' } }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true }}
          />
        </div>
      )}

      {/* Eigenvector Centrality */}
      {metrics.nodal.eigenvector_centrality && metrics.nodal.eigenvector_centrality.length > 0 && (
        <div className="plotly-container">
          <Plot
            data={[{
              x: nodeLabels,
              y: metrics.nodal.eigenvector_centrality,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: '#ec4899', size: 4 },
              line: { color: '#ec4899', width: 1.5 },
              name: 'Eigenvector',
            }]}
            layout={{ ...chartLayout('Eigenvector Centrality'), yaxis: { title: 'Centrality' } }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ responsive: true }}
          />
        </div>
      )}
    </div>
  );
}
