'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { GraphMetrics, ParcellationLabel } from '@/lib/types';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface EducationalPanelProps {
  metrics: GraphMetrics;
  labels: ParcellationLabel[];
}

const CONCEPTS = [
  {
    id: 'graph_theory',
    title: 'Graph Theory Basics',
    content: `The brain can be modeled as a network (graph) where regions are **nodes** and white matter tracts connecting them are **edges**.
    This mathematical framework lets us quantify brain organization using established graph metrics.

    - **Nodes**: Brain regions (parcels) from the Desikan-Killiany atlas (89 regions)
    - **Edges**: White matter fiber bundles detected through tractography
    - **Weight**: Number of streamlines connecting two regions (connection strength)`,
  },
  {
    id: 'small_world',
    title: 'Small-World Networks',
    content: `A small-world network has two key properties:

    1. **High clustering**: Neighboring nodes tend to be connected (like friend groups)
    2. **Short path lengths**: Any two nodes can be reached in few steps (like "six degrees of separation")

    Brain networks are small-world, meaning they balance **local specialization** (high clustering within functional modules) with **global integration** (short paths for long-range communication). The small-world index (sigma) compares your network to random networks — values above 1.0 confirm small-world topology.`,
  },
  {
    id: 'modularity',
    title: 'Community Structure & Modularity',
    content: `Brain networks organize into **communities** (modules) — groups of regions that are more densely connected to each other than to other groups.

    These communities often correspond to known functional systems:
    - **Visual network**: Occipital lobe regions
    - **Motor network**: Precentral, postcentral regions
    - **Default mode**: Medial prefrontal, posterior cingulate, precuneus
    - **Frontoparietal**: Lateral prefrontal, parietal regions

    **Modularity (Q)** measures how strongly the network divides into communities. Typical values: 0.2-0.5. Higher values mean more distinct modules.`,
  },
  {
    id: 'hubs',
    title: 'Hub Regions & Centrality',
    content: `Some brain regions are disproportionately important for network function — these are **hubs**.

    We identify hubs using centrality metrics:
    - **Degree**: Number of connections (highly connected regions)
    - **Betweenness centrality**: How many shortest paths pass through a region (information bottlenecks)
    - **Closeness centrality**: How close a region is to all others (efficient communicators)
    - **Eigenvector centrality**: Being connected to other well-connected regions

    Hub regions include the precuneus, superior frontal gyrus, and insula — they form the structural "backbone" of the brain network.`,
  },
  {
    id: 'efficiency',
    title: 'Network Efficiency',
    content: `**Global efficiency** measures how well information can travel across the entire network. It's the average of inverse shortest path lengths. Higher values mean more efficient communication.

    **Local efficiency** measures how well each node's neighbors can communicate with each other. This reflects the network's resilience — if a hub fails, can its neighbors still communicate?

    Together, these metrics describe the brain's capacity for both **integrated** (global) and **segregated** (local) processing — two fundamental principles of brain organization.`,
  },
];

export default function EducationalPanel({ metrics, labels }: EducationalPanelProps) {
  const [expandedConcept, setExpandedConcept] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'learn' | 'correlations' | 'compare'>('learn');

  const nodeCount = metrics.nodal.degree.length;
  const nodeLabels = labels.length === nodeCount
    ? labels.map(l => l.abbreviation)
    : Array.from({ length: nodeCount }, (_, i) => `R${i + 1}`);

  // Compute metric correlations
  const degreeStrengthCorr = computeCorrelation(metrics.nodal.degree, metrics.nodal.node_strength || []);
  const degreeBetweennessCorr = computeCorrelation(metrics.nodal.degree, metrics.nodal.betweenness_centrality);

  // Random network comparison values
  const nEdges = metrics.nodal.degree.reduce((a, b) => a + b, 0) / 2;
  const randomClustering = nEdges > 0 ? (2 * nEdges) / (nodeCount * (nodeCount - 1)) : 0;
  const randomPathLength = nodeCount > 1 && randomClustering > 0 ? Math.log(nodeCount) / Math.log(nodeCount * randomClustering) : 0;

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex gap-1 bg-white/5 rounded-lg p-1">
        {(['learn', 'correlations', 'compare'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            {tab === 'learn' ? 'Learn Concepts' : tab === 'correlations' ? 'Metric Correlations' : 'Network Comparison'}
          </button>
        ))}
      </div>

      {/* Learn Tab */}
      {activeTab === 'learn' && (
        <div className="space-y-3">
          {CONCEPTS.map((concept) => (
            <div key={concept.id} className="glass rounded-lg border border-white/10 overflow-hidden">
              <button
                onClick={() => setExpandedConcept(expandedConcept === concept.id ? null : concept.id)}
                className="w-full text-left px-5 py-4 flex items-center justify-between hover:bg-white/5 transition-colors"
              >
                <span className="font-semibold">{concept.title}</span>
                <svg
                  className={`w-5 h-5 transition-transform ${expandedConcept === concept.id ? 'rotate-180' : ''}`}
                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {expandedConcept === concept.id && (
                <div className="px-5 pb-4 text-sm text-gray-300 leading-relaxed whitespace-pre-line border-t border-white/10 pt-3">
                  {concept.content}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Correlations Tab */}
      {activeTab === 'correlations' && (
        <div className="space-y-6">
          <div className="glass rounded-xl p-5">
            <h3 className="font-semibold mb-2">What are metric correlations?</h3>
            <p className="text-sm text-gray-300">
              By plotting different metrics against each other, we can see how brain properties relate.
              For example, do highly connected regions also tend to be information bottlenecks?
              A strong positive correlation (r close to 1) means the metrics increase together.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Degree vs Strength scatter */}
            {metrics.nodal.node_strength && metrics.nodal.node_strength.length > 0 && (
              <div>
                <div className="plotly-container">
                  <Plot
                    data={[{
                      x: metrics.nodal.degree,
                      y: metrics.nodal.node_strength,
                      mode: 'markers',
                      type: 'scatter',
                      marker: { color: '#0ea5e9', size: 6, opacity: 0.7 },
                      text: nodeLabels,
                      hovertemplate: '%{text}<br>Degree: %{x}<br>Strength: %{y:.2f}<extra></extra>',
                    }]}
                    layout={{
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      font: { color: '#fff', size: 10 },
                      autosize: true,
                      margin: { l: 50, r: 20, t: 30, b: 40 },
                      title: `Degree vs Strength (r = ${degreeStrengthCorr.toFixed(3)})`,
                      xaxis: { title: 'Degree' },
                      yaxis: { title: 'Node Strength' },
                    }}
                    useResizeHandler
                    style={{ width: '100%', height: '300px' }}
                    config={{ responsive: true }}
                  />
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  {degreeStrengthCorr > 0.7
                    ? 'Strong positive correlation: regions with more connections also tend to have stronger total connection weights.'
                    : degreeStrengthCorr > 0.3
                    ? 'Moderate correlation: connection count and total weight are somewhat related.'
                    : 'Weak correlation: having many connections doesn\'t necessarily mean strong total weight.'}
                </p>
              </div>
            )}

            {/* Degree vs Betweenness */}
            <div>
              <div className="plotly-container">
                <Plot
                  data={[{
                    x: metrics.nodal.degree,
                    y: metrics.nodal.betweenness_centrality,
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: '#e94560', size: 6, opacity: 0.7 },
                    text: nodeLabels,
                    hovertemplate: '%{text}<br>Degree: %{x}<br>Betweenness: %{y:.4f}<extra></extra>',
                  }]}
                  layout={{
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#fff', size: 10 },
                    autosize: true,
                    margin: { l: 50, r: 20, t: 30, b: 40 },
                    title: `Degree vs Betweenness (r = ${degreeBetweennessCorr.toFixed(3)})`,
                    xaxis: { title: 'Degree' },
                    yaxis: { title: 'Betweenness Centrality' },
                  }}
                  useResizeHandler
                  style={{ width: '100%', height: '300px' }}
                  config={{ responsive: true }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {degreeBetweennessCorr > 0.7
                  ? 'Strong correlation: highly connected regions are also critical information relays.'
                  : degreeBetweennessCorr > 0.3
                  ? 'Moderate correlation: some highly connected regions serve as important bridges.'
                  : 'Weak correlation: high connectivity and being an information bottleneck are somewhat independent.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Compare Tab */}
      {activeTab === 'compare' && (
        <div className="space-y-6">
          <div className="glass rounded-xl p-5">
            <h3 className="font-semibold mb-2">Comparison with Random Networks</h3>
            <p className="text-sm text-gray-300">
              To understand whether observed network properties are meaningful, we compare them
              to equivalent random networks (Erdos-Renyi model with the same number of nodes and edges).
              Brain networks typically show much higher clustering than random while maintaining similar path lengths.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <ComparisonCard
              metric="Clustering Coefficient"
              actual={metrics.global.clustering_coefficient}
              random={randomClustering}
              explanation="Brain networks cluster ~10x more than random — reflecting local specialization."
            />
            <ComparisonCard
              metric="Path Length"
              actual={metrics.global.characteristic_path_length}
              random={randomPathLength}
              explanation="Path lengths are similar to random — preserving efficient long-range communication."
            />
            <ComparisonCard
              metric="Small-World Index"
              actual={metrics.global.small_worldness}
              random={1.0}
              explanation="Values >1 confirm small-world: high clustering without sacrificing path efficiency."
            />
          </div>

          <div className="plotly-container">
            <Plot
              data={[
                {
                  x: ['Clustering', 'Global Efficiency', 'Modularity'],
                  y: [metrics.global.clustering_coefficient, metrics.global.global_efficiency, metrics.global.modularity],
                  type: 'bar',
                  name: 'Brain Network',
                  marker: { color: '#0ea5e9' },
                },
                {
                  x: ['Clustering', 'Global Efficiency', 'Modularity'],
                  y: [randomClustering, randomClustering > 0 ? 1 / (Math.log(nodeCount) / Math.log(nodeCount * randomClustering)) : 0, 0],
                  type: 'bar',
                  name: 'Random Network',
                  marker: { color: '#666' },
                },
              ]}
              layout={{
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#fff', size: 10 },
                autosize: true,
                margin: { l: 50, r: 20, t: 30, b: 40 },
                title: 'Brain vs Random Network',
                barmode: 'group',
                yaxis: { title: 'Value' },
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

function ComparisonCard({ metric, actual, random, explanation }: {
  metric: string;
  actual: number;
  random: number;
  explanation: string;
}) {
  const ratio = random > 0 ? actual / random : 0;
  return (
    <div className="bg-white/5 rounded-lg p-4 border border-white/10">
      <div className="text-sm text-gray-400 mb-1">{metric}</div>
      <div className="flex items-baseline gap-3 mb-2">
        <div>
          <span className="text-lg font-semibold text-primary-400">{actual.toFixed(3)}</span>
          <span className="text-xs text-gray-500 ml-1">actual</span>
        </div>
        <div>
          <span className="text-sm text-gray-400">{random.toFixed(3)}</span>
          <span className="text-xs text-gray-500 ml-1">random</span>
        </div>
      </div>
      {ratio > 0 && (
        <div className="text-xs text-gray-500 mb-2">
          Ratio: {ratio.toFixed(2)}x
        </div>
      )}
      <p className="text-xs text-gray-400">{explanation}</p>
    </div>
  );
}

function computeCorrelation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  const meanX = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const meanY = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
  let sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    sumXY += dx * dy;
    sumX2 += dx * dx;
    sumY2 += dy * dy;
  }
  const denom = Math.sqrt(sumX2 * sumY2);
  return denom > 0 ? sumXY / denom : 0;
}
