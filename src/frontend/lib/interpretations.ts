// Metric interpretations for different user types
// Normative ranges based on neuroscience literature for structural brain networks

import { UserType, GraphMetrics, MetricInterpretation } from './types';

// Normative ranges for graph metrics (from structural connectivity literature)
export const METRIC_NORMALS: Record<string, {
  range: [number, number];
  unit: string;
  highMeaning: string;
  lowMeaning: string;
}> = {
  clustering_coefficient: {
    range: [0.2, 0.6],
    unit: '',
    highMeaning: 'Strong local clustering - brain regions form tightly connected groups',
    lowMeaning: 'Reduced local organization - fewer localized processing clusters',
  },
  characteristic_path_length: {
    range: [2.0, 4.0],
    unit: 'steps',
    highMeaning: 'Longer communication paths - potential disconnection between regions',
    lowMeaning: 'Short communication paths - efficient long-range connectivity',
  },
  global_efficiency: {
    range: [0.3, 0.7],
    unit: '',
    highMeaning: 'High information transfer efficiency across the entire network',
    lowMeaning: 'Reduced overall network communication efficiency',
  },
  modularity: {
    range: [0.2, 0.5],
    unit: '',
    highMeaning: 'Strong modular organization - distinct functional communities',
    lowMeaning: 'Less distinct community structure',
  },
  assortativity: {
    range: [-0.2, 0.3],
    unit: '',
    highMeaning: 'Hub regions preferentially connect to other hubs',
    lowMeaning: 'Hub regions tend to connect to peripheral regions',
  },
  small_worldness: {
    range: [1.0, 3.0],
    unit: 'sigma',
    highMeaning: 'Strong small-world organization (efficient + clustered)',
    lowMeaning: 'Reduced small-world properties',
  },
  density: {
    range: [0.05, 0.2],
    unit: '',
    highMeaning: 'Dense connectivity - many connections between regions',
    lowMeaning: 'Sparse connectivity - fewer inter-regional connections',
  },
  transitivity: {
    range: [0.2, 0.6],
    unit: '',
    highMeaning: 'High tendency to form interconnected triangles',
    lowMeaning: 'Low triangle formation in the network',
  },
};

export function getMetricStatus(
  metricName: string,
  value: number
): 'normal' | 'elevated' | 'reduced' {
  const norms = METRIC_NORMALS[metricName];
  if (!norms) return 'normal';

  if (value > norms.range[1]) return 'elevated';
  if (value < norms.range[0]) return 'reduced';
  return 'normal';
}

export function interpretMetric(
  metricName: string,
  value: number,
  userType: UserType
): MetricInterpretation {
  const norms = METRIC_NORMALS[metricName];
  const status = getMetricStatus(metricName, value);

  const displayName = metricName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  let doctor = '';
  let student = '';
  let general = '';

  if (!norms) {
    doctor = `${displayName}: ${value.toFixed(4)}`;
    student = `${displayName} measures a specific property of the brain network.`;
    general = `This is a measurement of how your brain regions are connected.`;
    return { doctor, student, general, status };
  }

  const rangeStr = `[${norms.range[0]}, ${norms.range[1]}]`;

  switch (status) {
    case 'elevated':
      doctor = `${displayName} = ${value.toFixed(4)} (elevated, normative range: ${rangeStr}). ${norms.highMeaning}. Clinical correlation recommended.`;
      student = `${displayName} is ${value.toFixed(4)}, which is above the typical range of ${rangeStr}. This indicates: ${norms.highMeaning.toLowerCase()}.`;
      general = `This measurement (${value.toFixed(3)}) is higher than usual. ${norms.highMeaning}.`;
      break;
    case 'reduced':
      doctor = `${displayName} = ${value.toFixed(4)} (reduced, normative range: ${rangeStr}). ${norms.lowMeaning}. Further investigation may be warranted.`;
      student = `${displayName} is ${value.toFixed(4)}, which is below the typical range of ${rangeStr}. This suggests: ${norms.lowMeaning.toLowerCase()}.`;
      general = `This measurement (${value.toFixed(3)}) is lower than usual. ${norms.lowMeaning}.`;
      break;
    default:
      doctor = `${displayName} = ${value.toFixed(4)} (within normative range: ${rangeStr}). No abnormality detected.`;
      student = `${displayName} is ${value.toFixed(4)}, which falls within the typical range of ${rangeStr}. This is considered normal for structural brain networks.`;
      general = `This measurement (${value.toFixed(3)}) is within the normal range, which is a positive finding.`;
  }

  return {
    doctor,
    student,
    general,
    normalRange: norms.range,
    status,
  };
}

export function generateSummary(
  metrics: GraphMetrics,
  userType: UserType
): string {
  const g = metrics.global;
  const nCommunities = metrics.communities?.louvain_partition
    ? new Set(metrics.communities.louvain_partition).size
    : 0;
  const nRegions = metrics.nodal.degree.length;

  switch (userType) {
    case 'doctor':
      return `Structural connectome analysis of ${nRegions} cortical and subcortical regions. ` +
        `Global efficiency: ${g.global_efficiency.toFixed(3)} (${getMetricStatus('global_efficiency', g.global_efficiency)}). ` +
        `Clustering: ${g.clustering_coefficient.toFixed(3)} (${getMetricStatus('clustering_coefficient', g.clustering_coefficient)}). ` +
        `Path length: ${g.characteristic_path_length.toFixed(2)} steps (${getMetricStatus('characteristic_path_length', g.characteristic_path_length)}). ` +
        `Modularity: ${g.modularity.toFixed(3)} with ${nCommunities} detected communities. ` +
        `Small-world index: ${g.small_worldness.toFixed(2)}. ` +
        (g.density !== undefined ? `Network density: ${(g.density * 100).toFixed(1)}%. ` : '') +
        `Recommend comparison with age-matched normative data.`;

    case 'student':
      return `This brain network consists of ${nRegions} regions connected by white matter tracts. ` +
        `The clustering coefficient (${g.clustering_coefficient.toFixed(3)}) measures how much neighboring regions ` +
        `tend to be interconnected, forming local processing clusters. ` +
        `The characteristic path length (${g.characteristic_path_length.toFixed(2)} steps) indicates the average ` +
        `number of connections needed to travel between any two regions. ` +
        `The network shows ${nCommunities} distinct communities (modularity: ${g.modularity.toFixed(3)}), ` +
        `which likely correspond to functional systems like visual, motor, and default mode networks. ` +
        `The small-world index (${g.small_worldness.toFixed(2)}) indicates ` +
        (g.small_worldness > 1 ? 'the network has small-world properties, balancing local specialization with global integration.' :
        'the network structure.');

    case 'general':
    default:
      return `Your brain scan analyzed ${nRegions} different regions of the brain and how they connect to each other through white matter pathways. ` +
        `The brain regions are organized into ${nCommunities} groups that work closely together. ` +
        `Information can travel between any two brain regions in about ${g.characteristic_path_length.toFixed(1)} steps on average, ` +
        `which ${getMetricStatus('characteristic_path_length', g.characteristic_path_length) === 'normal' ? 'is within the normal range' : 'may need further review'}. ` +
        `Overall, the network is working ` +
        (g.global_efficiency > 0.3 ? 'efficiently' : 'at a reduced capacity') +
        ` to transfer information across different parts of the brain.`;
  }
}

export function getMetricDescription(metricName: string, userType: UserType): string {
  const descriptions: Record<string, Record<UserType, string>> = {
    clustering_coefficient: {
      doctor: 'Fraction of triangles around a node (Watts & Strogatz). Reflects local segregation.',
      student: 'Measures how much a region\'s neighbors are also connected to each other, indicating local processing clusters.',
      general: 'Shows how closely grouped your brain connections are in small neighborhoods.',
    },
    characteristic_path_length: {
      doctor: 'Mean shortest path length across all node pairs. Reflects global integration capacity.',
      student: 'Average number of steps needed to travel between any two brain regions through the network.',
      general: 'How many connections apart any two brain regions are on average.',
    },
    global_efficiency: {
      doctor: 'Inverse of mean shortest path. More robust to disconnected components than path length.',
      student: 'Measures how efficiently information can be exchanged across the entire brain network.',
      general: 'How well your brain can send information between distant regions.',
    },
    modularity: {
      doctor: 'Newman modularity (Q) from community detection. Reflects modular decomposability.',
      student: 'Measures how strongly the network divides into distinct communities or modules.',
      general: 'How well your brain regions are organized into specialized groups.',
    },
    assortativity: {
      doctor: 'Pearson correlation of degrees at connected nodes. Positive = assortative mixing.',
      student: 'Measures whether highly connected regions (hubs) tend to connect to other hubs.',
      general: 'Whether the most connected brain regions link to each other.',
    },
    small_worldness: {
      doctor: 'Sigma = (C/C_rand) / (L/L_rand). Values >1 indicate small-world topology.',
      student: 'Ratio comparing the network to random networks. Values above 1 indicate the brain balances local specialization with global communication.',
      general: 'Measures whether your brain has the efficient organizational pattern found in healthy brains.',
    },
    density: {
      doctor: 'Ratio of actual to possible connections. Reflects overall connectivity level.',
      student: 'Proportion of all possible connections that actually exist in the network.',
      general: 'What fraction of all possible brain connections are actually present.',
    },
    transitivity: {
      doctor: 'Global clustering measure. Ratio of triangles to connected triples.',
      student: 'A global measure of clustering that considers the whole network rather than averaging individual nodes.',
      general: 'How much brain regions form interconnected groups across the whole brain.',
    },
  };

  return descriptions[metricName]?.[userType] || '';
}
