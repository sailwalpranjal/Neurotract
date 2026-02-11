"""
Parcellation Label Mapping

Maps generic parcellation indices (parcel_0, parcel_1, ...) to
anatomical region names based on the Desikan-Killiany atlas.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Desikan-Killiany atlas mapping for aparc-reduced (89 parcels)
# Based on FreeSurfer's aparc parcellation scheme
# Index 0 = background/unknown, 1-34 = left cortical, 35-68 = right cortical, 69-88 = subcortical
DESIKAN_KILLIANY_89: Dict[int, Dict[str, str]] = {
    0: {"name": "Unknown", "abbreviation": "UNK", "hemisphere": "none", "lobe": "none",
        "description": "Background or unclassified tissue"},
    # Left hemisphere cortical regions (1-34)
    1: {"name": "Left Banks STS", "abbreviation": "L-BSTS", "hemisphere": "left", "lobe": "temporal",
        "description": "Banks of the superior temporal sulcus, involved in social perception"},
    2: {"name": "Left Caudal Anterior Cingulate", "abbreviation": "L-CAC", "hemisphere": "left", "lobe": "frontal",
        "description": "Caudal anterior cingulate cortex, involved in error detection and conflict monitoring"},
    3: {"name": "Left Caudal Middle Frontal", "abbreviation": "L-CMF", "hemisphere": "left", "lobe": "frontal",
        "description": "Caudal middle frontal gyrus, involved in attention and eye movements"},
    4: {"name": "Left Cuneus", "abbreviation": "L-CUN", "hemisphere": "left", "lobe": "occipital",
        "description": "Cuneus, involved in basic visual processing"},
    5: {"name": "Left Entorhinal", "abbreviation": "L-ENT", "hemisphere": "left", "lobe": "temporal",
        "description": "Entorhinal cortex, critical for memory and spatial navigation"},
    6: {"name": "Left Fusiform", "abbreviation": "L-FUS", "hemisphere": "left", "lobe": "temporal",
        "description": "Fusiform gyrus, involved in face and object recognition"},
    7: {"name": "Left Inferior Parietal", "abbreviation": "L-IP", "hemisphere": "left", "lobe": "parietal",
        "description": "Inferior parietal lobule, involved in language, attention, and spatial processing"},
    8: {"name": "Left Inferior Temporal", "abbreviation": "L-IT", "hemisphere": "left", "lobe": "temporal",
        "description": "Inferior temporal gyrus, involved in visual object recognition"},
    9: {"name": "Left Isthmus Cingulate", "abbreviation": "L-IC", "hemisphere": "left", "lobe": "limbic",
        "description": "Isthmus of the cingulate gyrus, involved in memory retrieval"},
    10: {"name": "Left Lateral Occipital", "abbreviation": "L-LO", "hemisphere": "left", "lobe": "occipital",
         "description": "Lateral occipital cortex, involved in object recognition"},
    11: {"name": "Left Lateral Orbitofrontal", "abbreviation": "L-LOF", "hemisphere": "left", "lobe": "frontal",
         "description": "Lateral orbitofrontal cortex, involved in decision-making and reward"},
    12: {"name": "Left Lingual", "abbreviation": "L-LIN", "hemisphere": "left", "lobe": "occipital",
         "description": "Lingual gyrus, involved in visual processing and word recognition"},
    13: {"name": "Left Medial Orbitofrontal", "abbreviation": "L-MOF", "hemisphere": "left", "lobe": "frontal",
         "description": "Medial orbitofrontal cortex, involved in value-based decision making"},
    14: {"name": "Left Middle Temporal", "abbreviation": "L-MT", "hemisphere": "left", "lobe": "temporal",
         "description": "Middle temporal gyrus, involved in language comprehension and semantic memory"},
    15: {"name": "Left Parahippocampal", "abbreviation": "L-PH", "hemisphere": "left", "lobe": "temporal",
         "description": "Parahippocampal gyrus, involved in memory encoding and spatial processing"},
    16: {"name": "Left Paracentral", "abbreviation": "L-PC", "hemisphere": "left", "lobe": "frontal",
         "description": "Paracentral lobule, involved in motor and sensory functions of lower limbs"},
    17: {"name": "Left Pars Opercularis", "abbreviation": "L-POP", "hemisphere": "left", "lobe": "frontal",
         "description": "Pars opercularis (Broca's area), critical for speech production"},
    18: {"name": "Left Pars Orbitalis", "abbreviation": "L-POR", "hemisphere": "left", "lobe": "frontal",
         "description": "Pars orbitalis, involved in language and semantic processing"},
    19: {"name": "Left Pars Triangularis", "abbreviation": "L-PTR", "hemisphere": "left", "lobe": "frontal",
         "description": "Pars triangularis (Broca's area), involved in language processing"},
    20: {"name": "Left Pericalcarine", "abbreviation": "L-PCAL", "hemisphere": "left", "lobe": "occipital",
         "description": "Pericalcarine cortex (primary visual cortex V1)"},
    21: {"name": "Left Postcentral", "abbreviation": "L-POST", "hemisphere": "left", "lobe": "parietal",
         "description": "Postcentral gyrus (primary somatosensory cortex)"},
    22: {"name": "Left Posterior Cingulate", "abbreviation": "L-PCC", "hemisphere": "left", "lobe": "limbic",
         "description": "Posterior cingulate cortex, involved in self-referential processing"},
    23: {"name": "Left Precentral", "abbreviation": "L-PRE", "hemisphere": "left", "lobe": "frontal",
         "description": "Precentral gyrus (primary motor cortex)"},
    24: {"name": "Left Precuneus", "abbreviation": "L-PCUN", "hemisphere": "left", "lobe": "parietal",
         "description": "Precuneus, involved in self-awareness and episodic memory"},
    25: {"name": "Left Rostral Anterior Cingulate", "abbreviation": "L-RAC", "hemisphere": "left", "lobe": "frontal",
         "description": "Rostral anterior cingulate, involved in emotion regulation"},
    26: {"name": "Left Rostral Middle Frontal", "abbreviation": "L-RMF", "hemisphere": "left", "lobe": "frontal",
         "description": "Rostral middle frontal gyrus, involved in working memory"},
    27: {"name": "Left Superior Frontal", "abbreviation": "L-SF", "hemisphere": "left", "lobe": "frontal",
         "description": "Superior frontal gyrus, involved in executive function and self-awareness"},
    28: {"name": "Left Superior Parietal", "abbreviation": "L-SP", "hemisphere": "left", "lobe": "parietal",
         "description": "Superior parietal lobule, involved in visuospatial processing"},
    29: {"name": "Left Superior Temporal", "abbreviation": "L-ST", "hemisphere": "left", "lobe": "temporal",
         "description": "Superior temporal gyrus, involved in auditory processing and language"},
    30: {"name": "Left Supramarginal", "abbreviation": "L-SM", "hemisphere": "left", "lobe": "parietal",
         "description": "Supramarginal gyrus, involved in phonological processing and language"},
    31: {"name": "Left Frontal Pole", "abbreviation": "L-FP", "hemisphere": "left", "lobe": "frontal",
         "description": "Frontal pole, involved in complex planning and abstract reasoning"},
    32: {"name": "Left Temporal Pole", "abbreviation": "L-TP", "hemisphere": "left", "lobe": "temporal",
         "description": "Temporal pole, involved in semantic memory and social cognition"},
    33: {"name": "Left Transverse Temporal", "abbreviation": "L-TT", "hemisphere": "left", "lobe": "temporal",
         "description": "Transverse temporal gyrus (Heschl's gyrus, primary auditory cortex)"},
    34: {"name": "Left Insula", "abbreviation": "L-INS", "hemisphere": "left", "lobe": "insular",
         "description": "Insula, involved in interoception, emotion, and homeostasis"},
    # Right hemisphere cortical regions (35-68)
    35: {"name": "Right Banks STS", "abbreviation": "R-BSTS", "hemisphere": "right", "lobe": "temporal",
         "description": "Banks of the superior temporal sulcus, involved in social perception"},
    36: {"name": "Right Caudal Anterior Cingulate", "abbreviation": "R-CAC", "hemisphere": "right", "lobe": "frontal",
         "description": "Caudal anterior cingulate cortex, involved in error detection"},
    37: {"name": "Right Caudal Middle Frontal", "abbreviation": "R-CMF", "hemisphere": "right", "lobe": "frontal",
         "description": "Caudal middle frontal gyrus, involved in attention"},
    38: {"name": "Right Cuneus", "abbreviation": "R-CUN", "hemisphere": "right", "lobe": "occipital",
         "description": "Cuneus, involved in basic visual processing"},
    39: {"name": "Right Entorhinal", "abbreviation": "R-ENT", "hemisphere": "right", "lobe": "temporal",
         "description": "Entorhinal cortex, critical for memory and spatial navigation"},
    40: {"name": "Right Fusiform", "abbreviation": "R-FUS", "hemisphere": "right", "lobe": "temporal",
         "description": "Fusiform gyrus, involved in face and object recognition"},
    41: {"name": "Right Inferior Parietal", "abbreviation": "R-IP", "hemisphere": "right", "lobe": "parietal",
         "description": "Inferior parietal lobule, involved in attention and spatial processing"},
    42: {"name": "Right Inferior Temporal", "abbreviation": "R-IT", "hemisphere": "right", "lobe": "temporal",
         "description": "Inferior temporal gyrus, involved in visual object recognition"},
    43: {"name": "Right Isthmus Cingulate", "abbreviation": "R-IC", "hemisphere": "right", "lobe": "limbic",
         "description": "Isthmus of the cingulate gyrus, involved in memory retrieval"},
    44: {"name": "Right Lateral Occipital", "abbreviation": "R-LO", "hemisphere": "right", "lobe": "occipital",
         "description": "Lateral occipital cortex, involved in object recognition"},
    45: {"name": "Right Lateral Orbitofrontal", "abbreviation": "R-LOF", "hemisphere": "right", "lobe": "frontal",
         "description": "Lateral orbitofrontal cortex, involved in decision-making"},
    46: {"name": "Right Lingual", "abbreviation": "R-LIN", "hemisphere": "right", "lobe": "occipital",
         "description": "Lingual gyrus, involved in visual processing"},
    47: {"name": "Right Medial Orbitofrontal", "abbreviation": "R-MOF", "hemisphere": "right", "lobe": "frontal",
         "description": "Medial orbitofrontal cortex, involved in value-based decisions"},
    48: {"name": "Right Middle Temporal", "abbreviation": "R-MT", "hemisphere": "right", "lobe": "temporal",
         "description": "Middle temporal gyrus, involved in semantic processing"},
    49: {"name": "Right Parahippocampal", "abbreviation": "R-PH", "hemisphere": "right", "lobe": "temporal",
         "description": "Parahippocampal gyrus, involved in memory and scene processing"},
    50: {"name": "Right Paracentral", "abbreviation": "R-PC", "hemisphere": "right", "lobe": "frontal",
         "description": "Paracentral lobule, involved in motor and sensory functions"},
    51: {"name": "Right Pars Opercularis", "abbreviation": "R-POP", "hemisphere": "right", "lobe": "frontal",
         "description": "Pars opercularis, involved in motor and language processing"},
    52: {"name": "Right Pars Orbitalis", "abbreviation": "R-POR", "hemisphere": "right", "lobe": "frontal",
         "description": "Pars orbitalis, involved in language processing"},
    53: {"name": "Right Pars Triangularis", "abbreviation": "R-PTR", "hemisphere": "right", "lobe": "frontal",
         "description": "Pars triangularis, involved in language processing"},
    54: {"name": "Right Pericalcarine", "abbreviation": "R-PCAL", "hemisphere": "right", "lobe": "occipital",
         "description": "Pericalcarine cortex (primary visual cortex V1)"},
    55: {"name": "Right Postcentral", "abbreviation": "R-POST", "hemisphere": "right", "lobe": "parietal",
         "description": "Postcentral gyrus (primary somatosensory cortex)"},
    56: {"name": "Right Posterior Cingulate", "abbreviation": "R-PCC", "hemisphere": "right", "lobe": "limbic",
         "description": "Posterior cingulate cortex, involved in self-referential processing"},
    57: {"name": "Right Precentral", "abbreviation": "R-PRE", "hemisphere": "right", "lobe": "frontal",
         "description": "Precentral gyrus (primary motor cortex)"},
    58: {"name": "Right Precuneus", "abbreviation": "R-PCUN", "hemisphere": "right", "lobe": "parietal",
         "description": "Precuneus, involved in self-awareness and episodic memory"},
    59: {"name": "Right Rostral Anterior Cingulate", "abbreviation": "R-RAC", "hemisphere": "right", "lobe": "frontal",
         "description": "Rostral anterior cingulate, involved in emotion regulation"},
    60: {"name": "Right Rostral Middle Frontal", "abbreviation": "R-RMF", "hemisphere": "right", "lobe": "frontal",
         "description": "Rostral middle frontal gyrus, involved in working memory"},
    61: {"name": "Right Superior Frontal", "abbreviation": "R-SF", "hemisphere": "right", "lobe": "frontal",
         "description": "Superior frontal gyrus, involved in executive function"},
    62: {"name": "Right Superior Parietal", "abbreviation": "R-SP", "hemisphere": "right", "lobe": "parietal",
         "description": "Superior parietal lobule, involved in visuospatial processing"},
    63: {"name": "Right Superior Temporal", "abbreviation": "R-ST", "hemisphere": "right", "lobe": "temporal",
         "description": "Superior temporal gyrus, involved in auditory processing"},
    64: {"name": "Right Supramarginal", "abbreviation": "R-SM", "hemisphere": "right", "lobe": "parietal",
         "description": "Supramarginal gyrus, involved in phonological processing"},
    65: {"name": "Right Frontal Pole", "abbreviation": "R-FP", "hemisphere": "right", "lobe": "frontal",
         "description": "Frontal pole, involved in complex planning"},
    66: {"name": "Right Temporal Pole", "abbreviation": "R-TP", "hemisphere": "right", "lobe": "temporal",
         "description": "Temporal pole, involved in semantic memory"},
    67: {"name": "Right Transverse Temporal", "abbreviation": "R-TT", "hemisphere": "right", "lobe": "temporal",
         "description": "Transverse temporal gyrus (primary auditory cortex)"},
    68: {"name": "Right Insula", "abbreviation": "R-INS", "hemisphere": "right", "lobe": "insular",
         "description": "Insula, involved in interoception and emotion"},
    # Subcortical regions (69-88)
    69: {"name": "Left Thalamus", "abbreviation": "L-THA", "hemisphere": "left", "lobe": "subcortical",
         "description": "Thalamus, relay center for sensory and motor signals to the cortex"},
    70: {"name": "Left Caudate", "abbreviation": "L-CAU", "hemisphere": "left", "lobe": "subcortical",
         "description": "Caudate nucleus, involved in learning, memory, and goal-directed action"},
    71: {"name": "Left Putamen", "abbreviation": "L-PUT", "hemisphere": "left", "lobe": "subcortical",
         "description": "Putamen, involved in motor planning and execution"},
    72: {"name": "Left Pallidum", "abbreviation": "L-PAL", "hemisphere": "left", "lobe": "subcortical",
         "description": "Globus pallidus, involved in regulation of voluntary movement"},
    73: {"name": "Left Hippocampus", "abbreviation": "L-HIP", "hemisphere": "left", "lobe": "subcortical",
         "description": "Hippocampus, critical for memory formation and spatial navigation"},
    74: {"name": "Left Amygdala", "abbreviation": "L-AMY", "hemisphere": "left", "lobe": "subcortical",
         "description": "Amygdala, involved in emotion processing, especially fear and threat"},
    75: {"name": "Left Accumbens", "abbreviation": "L-ACC", "hemisphere": "left", "lobe": "subcortical",
         "description": "Nucleus accumbens, involved in reward processing and motivation"},
    76: {"name": "Left Ventral DC", "abbreviation": "L-VDC", "hemisphere": "left", "lobe": "subcortical",
         "description": "Ventral diencephalon, includes hypothalamus and subthalamic areas"},
    77: {"name": "Right Thalamus", "abbreviation": "R-THA", "hemisphere": "right", "lobe": "subcortical",
         "description": "Thalamus, relay center for sensory and motor signals"},
    78: {"name": "Right Caudate", "abbreviation": "R-CAU", "hemisphere": "right", "lobe": "subcortical",
         "description": "Caudate nucleus, involved in learning and goal-directed action"},
    79: {"name": "Right Putamen", "abbreviation": "R-PUT", "hemisphere": "right", "lobe": "subcortical",
         "description": "Putamen, involved in motor planning and execution"},
    80: {"name": "Right Pallidum", "abbreviation": "R-PAL", "hemisphere": "right", "lobe": "subcortical",
         "description": "Globus pallidus, involved in movement regulation"},
    81: {"name": "Right Hippocampus", "abbreviation": "R-HIP", "hemisphere": "right", "lobe": "subcortical",
         "description": "Hippocampus, critical for memory formation"},
    82: {"name": "Right Amygdala", "abbreviation": "R-AMY", "hemisphere": "right", "lobe": "subcortical",
         "description": "Amygdala, involved in emotion processing"},
    83: {"name": "Right Accumbens", "abbreviation": "R-ACC", "hemisphere": "right", "lobe": "subcortical",
         "description": "Nucleus accumbens, involved in reward processing"},
    84: {"name": "Right Ventral DC", "abbreviation": "R-VDC", "hemisphere": "right", "lobe": "subcortical",
         "description": "Ventral diencephalon, includes hypothalamus"},
    85: {"name": "Left Cerebellum Cortex", "abbreviation": "L-CBLM", "hemisphere": "left", "lobe": "cerebellum",
         "description": "Cerebellar cortex, involved in motor coordination and cognitive function"},
    86: {"name": "Right Cerebellum Cortex", "abbreviation": "R-CBLM", "hemisphere": "right", "lobe": "cerebellum",
         "description": "Cerebellar cortex, involved in motor coordination"},
    87: {"name": "Brain Stem", "abbreviation": "BSTM", "hemisphere": "midline", "lobe": "brainstem",
         "description": "Brain stem, controls vital functions like breathing, heart rate, and consciousness"},
    88: {"name": "Corpus Callosum", "abbreviation": "CC", "hemisphere": "midline", "lobe": "white_matter",
         "description": "Corpus callosum, major white matter tract connecting left and right hemispheres"},
}


def get_parcellation_labels(
    labels_file: str = None,
    n_parcels: int = 89,
) -> List[Dict[str, Any]]:
    """
    Map generic parcellation indices to anatomical names.

    Args:
        labels_file: Path to connectome_labels.txt (optional)
        n_parcels: Number of parcels expected

    Returns:
        List of label dictionaries with anatomical names
    """
    # Read generic labels if file provided
    generic_labels = []
    if labels_file and Path(labels_file).exists():
        with open(labels_file) as f:
            generic_labels = [line.strip() for line in f if line.strip()]

    labels = []
    for i in range(n_parcels):
        generic_name = generic_labels[i] if i < len(generic_labels) else f"parcel_{i}"

        if i in DESIKAN_KILLIANY_89:
            info = DESIKAN_KILLIANY_89[i]
            labels.append({
                "index": i,
                "generic_name": generic_name,
                "anatomical_name": info["name"],
                "abbreviation": info["abbreviation"],
                "hemisphere": info["hemisphere"],
                "lobe": info["lobe"],
                "description": info["description"],
            })
        else:
            labels.append({
                "index": i,
                "generic_name": generic_name,
                "anatomical_name": generic_name,
                "abbreviation": f"P{i}",
                "hemisphere": "unknown",
                "lobe": "unknown",
                "description": f"Region {i}",
            })

    return labels


# Approximate centroid positions for major brain lobes (in MNI/RAS space, mm)
# Used for 3D label placement in the viewer
LOBE_CENTROIDS = {
    "Left Frontal": [-30, 40, 30],
    "Right Frontal": [30, 40, 30],
    "Left Parietal": [-30, -40, 50],
    "Right Parietal": [30, -40, 50],
    "Left Temporal": [-50, -10, -15],
    "Right Temporal": [50, -10, -15],
    "Left Occipital": [-15, -80, 10],
    "Right Occipital": [15, -80, 10],
    "Cerebellum": [0, -60, -30],
    "Brain Stem": [0, -25, -30],
}
