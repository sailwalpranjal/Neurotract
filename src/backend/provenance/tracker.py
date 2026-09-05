"""
Scientific Provenance & Traceability System for NeuroTract 2.0

Enforces full end-to-end provenance for every derived metric, scalar map,
tractogram, and connectome. Connects results to datasets, algorithms,
parameters, software versions, and random seeds.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
import json
import logging

logger = logging.getLogger(__name__)

# Canonical metric registry with mathematical definitions and literature citations
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "global_efficiency": {
        "name": "Global Efficiency",
        "symbol": "E_glob",
        "formula": "E_glob = 1 / (N * (N - 1)) * sum_{i != j} (1 / d_{ij})",
        "description": "Average inverse shortest path length between all pairs of nodes in the structural network.",
        "units": "dimensionless [0, 1]",
        "reference": "Latora & Marchiori (2001). Efficient behavior of small-world networks. Phys. Rev. Lett. 87(19): 198701.",
        "interpretation": "Measures the overall capacity for parallel information transfer across the entire structural network."
    },
    "clustering_coefficient": {
        "name": "Clustering Coefficient",
        "symbol": "C",
        "formula": "C = 1/N * sum_i (2 * t_i / (k_i * (k_i - 1)))",
        "description": "Mean fraction of a node's neighbors that are connected to each other.",
        "units": "dimensionless [0, 1]",
        "reference": "Watts & Strogatz (1998). Collective dynamics of 'small-world' networks. Nature 393: 440-442.",
        "interpretation": "Measures the degree of local interconnectivity and modular specialization."
    },
    "characteristic_path_length": {
        "name": "Characteristic Path Length",
        "symbol": "L",
        "formula": "L = 1 / (N * (N - 1)) * sum_{i != j} d_{ij}",
        "description": "Average shortest path length across all reachable pairs of regions in the network.",
        "units": "steps / hops",
        "reference": "Watts & Strogatz (1998). Nature 393: 440-442.",
        "interpretation": "Represents the average structural distance required to traverse between brain regions."
    },
    "modularity": {
        "name": "Modularity (Louvain)",
        "symbol": "Q",
        "formula": "Q = 1/(2m) * sum_{ij} [A_{ij} - (k_i * k_j) / (2m)] * delta(c_i, c_j)",
        "description": "Degree to which the structural connectome subdivides into non-overlapping communities.",
        "units": "dimensionless [-0.5, 1.0]",
        "reference": "Blondel et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech. P10008.",
        "interpretation": "Quantifies the segregation of the brain into distinct topological modules."
    },
    "small_world_sigma": {
        "name": "Small-Worldness (Sigma)",
        "symbol": "sigma",
        "formula": "sigma = (C / C_rand) / (L / L_rand)",
        "description": "Ratio of normalized clustering to normalized path length compared to degree-matched random graphs.",
        "units": "dimensionless (sigma > 1 indicates small-world topology)",
        "reference": "Humphries & Gurney (2008). Network 'small-world-ness'. PLOS ONE 3(4): e0002051.",
        "interpretation": "Reflects the balance between specialized local clustering and efficient global integration."
    },
    "density": {
        "name": "Network Density",
        "symbol": "rho",
        "formula": "rho = 2 * E / (N * (N - 1))",
        "description": "Ratio of existing structural connections to maximum possible connections.",
        "units": "dimensionless [0, 1]",
        "reference": "Sporns (2011). Networks of the Brain. MIT Press.",
        "interpretation": "Represents the overall connection density of the parcellated network."
    },
    "transitivity": {
        "name": "Transitivity",
        "symbol": "T",
        "formula": "T = 3 * (number of triangles) / (number of connected triplets)",
        "description": "Classical global clustering metric based on the fraction of closed triads.",
        "units": "dimensionless [0, 1]",
        "reference": "Newman (2003). The structure and function of complex networks. SIAM Review 45(2): 167-256.",
        "interpretation": "Reflects network-wide triangle prevalence without node-degree bias."
    },
    "assortativity": {
        "name": "Degree Assortativity",
        "symbol": "r",
        "formula": "Pearson correlation coefficient of degree between pairs of connected nodes.",
        "description": "Propensity of high-degree hub regions to connect preferentially to other hubs.",
        "units": "dimensionless [-1.0, 1.0]",
        "reference": "Newman (2002). Assortative mixing in networks. Phys. Rev. Lett. 89: 208701.",
        "interpretation": "Positive assortativity indicates interconnected hub cores; negative indicates peripheral hub connections."
    },
    "fa": {
        "name": "Fractional Anisotropy",
        "symbol": "FA",
        "formula": "FA = sqrt(3/2) * sqrt(sum((lambda_i - MD)^2)) / sqrt(sum(lambda_i^2))",
        "description": "Degree of directional preference of water diffusion in each voxel.",
        "units": "dimensionless [0, 1]",
        "reference": "Basser & Pierpaoli (1996). Microstructural and physiological features of tissues elucidated by dMRI. JMR B 111(3): 209-219.",
        "interpretation": "Higher values indicate aligned cellular barriers such as myelinated axon bundles."
    },
    "md": {
        "name": "Mean Diffusivity",
        "symbol": "MD",
        "formula": "MD = (lambda_1 + lambda_2 + lambda_3) / 3",
        "description": "Orientationally-averaged magnitude of water diffusion in each voxel.",
        "units": "mm^2 / s",
        "reference": "Basser, Mattiello, LeBihan (1994). MR diffusion tensor spectroscopy and imaging. Biophys. J. 66(1): 259-267.",
        "interpretation": "Inversely relates to cellular density and membrane barriers."
    }
}


def get_software_versions() -> Dict[str, str]:
    """Retrieve installed versions of all scientific libraries"""
    versions = {"neurotract": "2.0.0"}
    for pkg in ["dipy", "nibabel", "networkx", "scipy", "numpy", "sklearn", "fastapi"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"
    return versions


@dataclass
class MetricProvenance:
    """Provenance record for a single computed metric"""
    metric_id: str
    name: str
    value: Any
    units: str
    formula: str
    description: str
    reference_citation: str
    input_properties: Dict[str, Any]
    execution_id: str
    timestamp: str
    software_versions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineExecutionProvenance:
    """Full execution provenance record for a pipeline run"""
    execution_id: str
    dataset_name: str
    dataset_checksums: Dict[str, str]
    created_at: str
    completed_at: Optional[str]
    status: str
    rng_seed: int
    software_versions: Dict[str, str]
    preprocessing_params: Dict[str, Any]
    tractography_params: Dict[str, Any]
    connectome_params: Dict[str, Any]
    stage_timings_seconds: Dict[str, float] = field(default_factory=dict)
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    metric_provenance_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceTracker:
    """
    Manages generation, storage, and querying of scientific provenance records
    """

    def __init__(self, storage_dir: Union[str, Path] = "output/provenance"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_executions: Dict[str, PipelineExecutionProvenance] = {}

    def create_execution(
        self,
        dataset_name: str,
        dataset_checksums: Dict[str, str],
        rng_seed: int,
        preprocessing_params: Optional[Dict[str, Any]] = None,
        tractography_params: Optional[Dict[str, Any]] = None,
        connectome_params: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> PipelineExecutionProvenance:
        """Create a new execution provenance tracker"""
        exec_id = execution_id or str(uuid.uuid4())
        record = PipelineExecutionProvenance(
            execution_id=exec_id,
            dataset_name=dataset_name,
            dataset_checksums=dataset_checksums,
            created_at=datetime.utcnow().isoformat() + "Z",
            completed_at=None,
            status="running",
            rng_seed=rng_seed,
            software_versions=get_software_versions(),
            preprocessing_params=preprocessing_params or {},
            tractography_params=tractography_params or {},
            connectome_params=connectome_params or {},
        )
        self.active_executions[exec_id] = record
        self.save_record(record)
        return record

    def record_stage_timing(self, execution_id: str, stage: str, duration_sec: float):
        """Record stage execution timing"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].stage_timings_seconds[stage] = round(duration_sec, 3)

    def record_telemetry(self, execution_id: str, telemetry: Dict[str, Any]):
        """Record telemetry summary"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].telemetry_summary.update(telemetry)

    def record_output_artifact(self, execution_id: str, artifact_type: str, path: str):
        """Record generated output artifact path"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id].output_artifacts[artifact_type] = str(path)

    def create_metric_provenance(
        self,
        execution_id: str,
        metric_key: str,
        value: Any,
        input_properties: Optional[Dict[str, Any]] = None
    ) -> MetricProvenance:
        """Create inspectable provenance for a specific derived metric"""
        meta = METRIC_REGISTRY.get(metric_key, {
            "name": metric_key.replace("_", " ").title(),
            "formula": "Computed from structural network adjacency matrix",
            "description": f"Derived connectivity metric: {metric_key}",
            "units": "arbitrary",
            "reference": "Sporns (2011). Networks of the Brain.",
        })

        prov = MetricProvenance(
            metric_id=str(uuid.uuid4()),
            name=meta["name"],
            value=value,
            units=meta["units"],
            formula=meta["formula"],
            description=meta["description"],
            reference_citation=meta.get("reference", ""),
            input_properties=input_properties or {},
            execution_id=execution_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            software_versions=get_software_versions()
        )

        if execution_id in self.active_executions:
            self.active_executions[execution_id].metric_provenance_records[metric_key] = prov.to_dict()

        return prov

    def complete_execution(self, execution_id: str, status: str = "completed"):
        """Mark an execution as completed and persist to disk"""
        if execution_id in self.active_executions:
            rec = self.active_executions[execution_id]
            rec.completed_at = datetime.utcnow().isoformat() + "Z"
            rec.status = status
            self.save_record(rec)

    def save_record(self, record: PipelineExecutionProvenance):
        """Save execution record to JSON file"""
        out_file = self.storage_dir / f"{record.execution_id}.json"
        with open(out_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def get_record(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve execution provenance by ID"""
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].to_dict()
        file_path = self.storage_dir / f"{execution_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    def get_metric_provenance(self, execution_id: str, metric_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metric provenance for a specific metric in an execution"""
        rec = self.get_record(execution_id)
        if rec and "metric_provenance_records" in rec:
            return rec["metric_provenance_records"].get(metric_key)
        # Fallback: construct standard definition if execution ID is demo
        if metric_key in METRIC_REGISTRY:
            meta = METRIC_REGISTRY[metric_key]
            return {
                "name": meta["name"],
                "units": meta["units"],
                "formula": meta["formula"],
                "description": meta["description"],
                "reference_citation": meta.get("reference", ""),
                "execution_id": execution_id,
                "software_versions": get_software_versions()
            }
        return None


# Global singleton instance
global_provenance_tracker = ProvenanceTracker()
