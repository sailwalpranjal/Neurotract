# NeuroTract: Brain White Matter Tractography and Connectivity Analysis Platform

## Project Overview

NeuroTract is a software system for analyzing the structural connectivity of the human brain using diffusion magnetic resonance imaging (dMRI). The platform processes raw MRI scanner data through a 7-stage pipeline that produces three-dimensional reconstructions of white matter fiber pathways and computes graph-theoretic measures of brain network organization. The system outputs both quantitative connectivity metrics and interactive visualizations that allow clinicians to explore brain structure in detail.

The project consists of a Python-based processing backend that handles computationally intensive neuroimaging operations, a REST API server that exposes results through HTTP endpoints, and a web frontend built with Next.js and Three.js that renders brain data in real-time 3D graphics.

## Why This Project Exists

### The Clinical Problem

Neurological disorders including Alzheimer's disease, multiple sclerosis, traumatic brain injury, stroke, and schizophrenia cause measurable disruptions to the brain's white matter connectivity. Traditional MRI scans show structural damage, but diffusion MRI reveals functional connectivity problems that may appear before visible tissue damage. However, processing diffusion MRI data requires specialized software, significant computational resources, and domain expertise that limits its use in clinical settings.

### The Research Gap

Existing tractography tools (MRtrix3, DSI Studio, TrackVis, FSL) are designed primarily as command-line utilities for researchers. They require manual execution of each processing step, lack integrated visualization, and do not provide interpretations adapted to different user expertise levels. Researchers spend significant time writing custom scripts to connect these tools, and the results are typically static images rather than interactive 3D models.

### This Project's Solution

NeuroTract was built to provide an end-to-end platform that:

1. Accepts raw MRI scanner files as input without requiring preprocessing
2. Automatically executes all processing steps with validated parameter defaults
3. Produces both quantitative metrics and interactive visualizations
4. Adapts its interface based on whether the user is a clinician, student, or general audience
5. Provides complete transparency into processing decisions through detailed logging
6. Enables exploration of results through web browser without installing desktop software

## How the System Works: File Processing Pipeline

### Input: What Files Does the System Need?

The minimum required input consists of three files produced by an MRI scanner:

**1. Diffusion-Weighted Image (DWI)**
A 4-dimensional NIfTI file (.nii or .nii.gz) containing multiple 3D brain volumes acquired with different diffusion gradient directions. Each voxel (3D pixel) contains signal intensity values that change based on water molecule movement along white matter fibers. Typical file size: 200-800 MB depending on spatial resolution and number of gradient directions.

**2. B-values File**
A text file (.bval) listing the diffusion weighting strength for each volume in the DWI file. Higher b-values increase sensitivity to white matter fiber direction but reduce signal-to-noise ratio. Typical values range from 0 (no diffusion weighting) to 1000-4000 s/mm².

**3. B-vectors File**
A text file (.bvec) specifying the 3D gradient direction for each DWI volume. These directions are distributed across a sphere to sample diffusion in all spatial orientations.

Optional inputs include a T1-weighted anatomical scan for improved brain extraction and a pre-computed parcellation atlas that divides the brain into anatomically labeled regions.

### Stage 1: Data Ingestion and Validation

The system first loads the NIfTI file using the Nibabel library, extracting both the image data and the affine transformation matrix that maps voxel coordinates to real-world millimeter positions. The gradient table (b-values and b-vectors) undergoes validation to ensure the number of entries matches the number of DWI volumes and that b-vectors are unit-normalized.

If the input follows the BIDS (Brain Imaging Data Structure) standard, the system automatically discovers associated metadata files and organizes outputs according to BIDS conventions. For non-BIDS datasets, the user provides file paths directly.

The system creates a structured output directory for the subject with subdirectories for each processing stage, ensuring that intermediate results are preserved for quality control and reprocessing.

### Stage 2: Preprocessing

Raw scanner data contains artifacts that must be corrected before tractography:

**Motion and Eddy Current Correction**
Subject movement during the long MRI scan (30-60 minutes) causes misalignment between volumes. Eddy currents induced by the rapid switching of magnetic field gradients distort the images. The system uses FSL's eddy tool if available (faster, GPU-accelerated) or falls back to DIPY's registration-based correction. Each DWI volume is aligned to a reference b=0 volume through affine transformation.

**Brain Extraction**
The system segments brain tissue from surrounding skull, scalp, and neck tissue using DIPY's median_otsu algorithm. This creates a binary brain mask that restricts subsequent processing to relevant tissue. The algorithm computes the Otsu threshold on the median-filtered image to handle intensity inhomogeneity.

**Bias Field Correction**
Spatial variations in the MRI radio frequency coil create smooth intensity gradients across the image unrelated to tissue properties. The system estimates and removes this bias field using either N4BiasFieldCorrection (from ANTs, if available) or a polynomial fitting approach. This normalization improves quantitative accuracy of diffusion metrics.

The outputs include corrected DWI volumes, the binary brain mask, and quality control reports documenting the number of volumes rejected due to excessive motion.

### Stage 3: Microstructure Modeling

**Diffusion Tensor Imaging (DTI)**
The system fits a diffusion tensor (a 3x3 symmetric matrix describing water movement) to the DWI signal at each brain voxel. This uses weighted least squares with the Levenberg-Marquardt algorithm to solve the Stejskal-Tanner equation. From the tensor eigenvectors and eigenvalues, the system computes:

- Fractional Anisotropy (FA): 0-1 scale measuring how directional diffusion is (higher in well-organized white matter)
- Mean Diffusivity (MD): Average diffusion rate across all directions
- Radial Diffusivity (RD): Diffusion perpendicular to the main fiber direction
- Axial Diffusivity (AD): Diffusion along the main fiber direction

These scalar maps are saved as separate NIfTI files and used to guide tractography termination.

**Constrained Spherical Deconvolution (CSD)**
DTI assumes a single fiber orientation per voxel, which fails at crossing fibers (present in ~90% of white matter voxels). CSD models the DWI signal as a convolution of a fiber orientation distribution (FOD) with a response function representing the signal from a single perfectly aligned fiber bundle.

The system estimates the response function using the Dhollander algorithm, which automatically identifies voxels containing only white matter, gray matter, or cerebrospinal fluid based on signal characteristics. It then solves the deconvolution problem in the spherical harmonic basis using constrained optimization to ensure the FOD is non-negative and normalized.

The output FOD file contains spherical harmonic coefficients (typically order 6-8) at each white matter voxel. Visualization of the FOD shows multiple lobes pointing along different fiber orientations.

### Stage 4: Probabilistic Tractography

Tractography reconstructs the 3D trajectories of white matter bundles by numerically integrating streamlines that follow the fiber orientations specified by the FOD.

**Seeding**
The system places seed points (starting positions for tracking) uniformly throughout white matter voxels. The user-configurable seeds-per-voxel parameter (default: 2) determines tracking density. Typical whole-brain tractography uses 50,000 to 200,000 seeds.

**Integration**
Starting from each seed, the system propagates a streamline by:
1. Sampling the FOD at the current position to find fiber orientation peaks
2. Randomly selecting one peak weighted by its amplitude (probabilistic tracking)
3. Computing the next position using 4th-order Runge-Kutta (RK4) integration with adaptive step size
4. Repeating until termination criteria are met

Termination occurs when:
- The streamline exits the brain mask
- The fractional anisotropy drops below the threshold (default: 0.1), indicating the tract has entered gray matter or CSF
- The curvature exceeds the maximum angle between steps (default: 30 degrees), preventing anatomically implausible turns
- The streamline reaches a maximum length (default: 250mm)

**Clustering**
The system groups similar streamlines using the QuickBundles algorithm, which clusters based on geometric similarity measured by the Minimum Direct Flip (MDF) distance. This reduces redundancy and identifies distinct fiber bundles.

Output is a .trk file (TrackVis format) containing streamline coordinates, the number of points per streamline, and optional per-point or per-streamline properties (FA values, curvature, etc.). Typical whole-brain tractography produces 100,000-300,000 streamlines totaling 10-50 million points.

### Stage 5: Surface Generation

**Brain Mesh Construction**
The system creates a triangular mesh representing the brain surface using the marching cubes algorithm on the Gaussian-smoothed brain mask (sigma=1.0). This produces a polygonal approximation of the 3D brain surface with vertices, faces, and surface normals.

The mesh vertices are transformed to RAS (Right-Anterior-Superior) coordinate space using the NIfTI affine matrix, ensuring alignment with the streamline coordinates for correct visualization.

**Parcellation Mapping**
The system loads the parcellation atlas (Desikan-Killiany 89-region scheme by default), which assigns each brain voxel to an anatomically labeled region (e.g., left superior frontal gyrus, right hippocampus). The mapping includes:
- Generic region names from the atlas
- Full anatomical names
- Standard abbreviations
- Hemisphere classification (left/right/midline)
- Lobe assignment (frontal/parietal/temporal/occipital/limbic/subcortical)
- Plain-language descriptions

This mapping enables the system to translate numerical results into anatomical interpretations.

### Stage 6: Connectome Construction

The structural connectome is a graph representing brain regions as nodes and white matter connections as weighted edges.

**Endpoint Mapping**
For each streamline, the system determines which parcellation regions contain its start and end points. A streamline that begins in region A and ends in region B contributes to the edge weight between those regions.

**Edge Weighting Strategies**

*Count-based*: Edge weight = number of streamlines connecting two regions. Simple but biased toward large, nearby regions.

*FA-weighted*: Edge weight = sum of mean FA values along connecting streamlines. Incorporates microstructure information about connection integrity.

*Length-normalized*: Edge weight = streamline count divided by mean streamline length. Reduces distance bias.

The system uses count-based weighting by default and saves the NxN connectivity matrix (N = number of parcels) as a NumPy array file (.npy).

### Stage 7: Graph-Theoretic Analysis

The connectome is analyzed as a weighted undirected graph using NetworkX and custom implementations of graph metrics.

**Global Metrics** (characterize the entire brain network):

*Clustering Coefficient*: Average probability that a node's neighbors are also connected to each other. Reflects local network efficiency and modularity. Healthy brains typically show values of 0.35-0.45.

*Characteristic Path Length*: Average shortest path between all pairs of nodes. Measures global information transfer efficiency. Values around 2.5-3.5 indicate efficient long-range communication.

*Global Efficiency*: Average inverse shortest path length. Alternative to characteristic path length that handles disconnected components. Higher values (0.4-0.6) indicate better global integration.

*Modularity*: Strength of division into communities (groups of densely interconnected regions). Values of 0.25-0.45 suggest specialized functional subsystems exist.

*Small-Worldness*: Ratio comparing the brain network to equivalent random networks. Values above 1.0 indicate the network balances high local clustering with short global path lengths, the signature of efficient biological networks.

*Assortativity*: Tendency for high-degree nodes (hubs) to connect to other high-degree nodes. Can be positive (hubs connect to hubs) or negative (hubs connect to low-degree nodes).

**Nodal Metrics** (computed separately for each brain region):

*Degree*: Number of connections to other regions. Hub regions have high degree (15-30+ connections).

*Betweenness Centrality*: Fraction of shortest paths passing through a region. Measures how critical a region is for information routing. Values above 0.05 identify bottleneck nodes.

*Closeness Centrality*: Inverse of average shortest path to all other nodes. Regions with high closeness can communicate efficiently with the entire network.

*Eigenvector Centrality*: Measures influence based on connections to other influential nodes. Identifies nodes embedded in highly connected neighborhoods.

*Local Efficiency*: Average efficiency of a node's neighborhood, reflecting fault tolerance.

*Node Strength*: Sum of edge weights for connections to a node. Captures total connectivity strength beyond connection count.

**Community Detection**:
The system applies Louvain and Leiden algorithms to identify communities (modules). These optimize modularity by iteratively reassigning nodes to groups that maximize internal connectivity and minimize external connectivity. The resulting partition typically identifies 4-8 communities corresponding to known functional systems (default mode, frontoparietal, motor, visual, etc.).

**Rich Club Analysis**:
Computes the rich club coefficient as a function of degree threshold k. If regions with degree > k are more densely interconnected than expected by chance, this indicates a "rich club" of hub regions that preferentially wire to each other, forming a high-capacity backbone for information integration.

All metrics are saved to a JSON file containing global metrics, arrays of nodal metrics (one value per region), community assignments, and rich club curves.

## Web Interface: How Results Are Visualized

### Home Page

The home page serves as the entry point and dashboard:

**Server Connection Monitoring**: Continuously checks the backend health endpoint (/health) and displays connection status with green (connected) or red (disconnected) indication. If the backend is unreachable, the interface prompts the user to start the server and provides the command.

**Subject Discovery**: Automatically scans the output/ directory for processed results and displays cards for each available subject. Each card shows:
- Subject ID
- Data availability badges (streamlines, DTI maps, FOD, connectome, metrics)
- Quick stats (number of streamlines, brain regions analyzed, total connections)
- "Load & View" button that sets the active subject and navigates to the viewer

**Pipeline Visualization**: An interactive diagram shows the 7 processing stages with arrows indicating data flow. This educates users about what operations occur between raw MRI files and final results.

**User Type Selection**: Three options (Clinician, Researcher, Student) determine how analysis results are presented. This selection is stored in browser localStorage and persists across sessions.

**File Upload**: A drag-and-drop zone accepts neuroimaging files (.nii, .nii.gz, .trk, .tck, .dcm). Files are uploaded to the backend via POST /upload with progress tracking. After successful upload, the system currently displays a message prompting manual pipeline execution (automatic execution is planned).

### 3D Viewer Page

The viewer loads two data types in parallel: streamline coordinates and brain surface geometry. Both are fetched from the backend REST API and rendered in a Three.js WebGL scene.

**Brain Surface Models**:

The system offers three rendering modes:

*Hologram*: A high-quality 3D brain mesh created from anatomical atlas data. This model includes detailed cortical surface geometry with anatomically accurate sulci and gyri. The mesh is pre-textured with semi-transparent material settings suitable for overlay with streamline data.

*Point Cloud*: A volumetric representation of the brain as approximately 1 million colored points. Each point samples a position within brain tissue, creating a cloud-like appearance that reveals internal structure when rotated.

*MRI Mesh*: Generated during pipeline execution from the subject's own brain mask using marching cubes. This model matches the subject's specific anatomy but has lower polygon count than the pre-made models. File size varies (typically 2-5MB) depending on mask resolution.

The system initially scales GLB models to an arbitrary size of 150 units, but when streamline data is available, it re-scales and repositions the model to align with the streamline coordinate space. This ensures the brain surface correctly encloses the white matter tracts. The alignment computes a bounding box from streamline coordinates and applies a uniform scale (0.85x to create slight inset) plus translation to match the model's bounding box to the streamline bounds.

**Streamline Rendering**:

The backend subsamples the full streamline set (typically 100,000+ streamlines) to 3,000 streamlines for web rendering to maintain interactive frame rates. Each streamline is converted to a Three.js Line object with its coordinates passed as a BufferGeometry position attribute.

Color mapping modes:
- *Orientation*: RGB encodes the streamline's mean orientation vector (x=red, y=green, z=blue). Left-right fibers appear red, anterior-posterior green, inferior-superior blue. This reveals anatomical structure.
- *Length*: Color maps from short (blue) to long (red) streamlines according to the selected colormap.
- *FA*: Colors based on fractional anisotropy values if available in the streamline file.
- *Custom*: Uses per-streamline colors if present in the source data.

Streamlines are rendered with configurable opacity (default 0.7) to handle dense regions, and line width (default 1.0 pixel) can be adjusted for visibility.

**Camera and Controls**:

An orbital camera surrounds the brain at default position [250, 100, 250] looking toward the origin. The user controls the camera via:
- Left mouse button: Rotate around the brain
- Right mouse button: Pan laterally
- Scroll wheel: Zoom in/out
- View preset buttons: Snap to anatomical orientations (anterior, posterior, left, right, superior, inferior, 3/4 view)

Auto-rotation mode continuously rotates the scene at a configurable speed, useful for demonstrations or overviews.

**Anatomical Labels**:

When enabled, the system renders text labels at approximate centroid positions for major brain lobes (Frontal, Parietal, Temporal, Occipital, Cingulate). These use HTML overlays positioned in 3D space via the @react-three/drei Html component, allowing the labels to always face the camera.

**Material Controls Sidebar**:

The right sidebar provides sliders to adjust visual properties:
- Brain surface opacity (0-1): Controls transparency from invisible to fully opaque
- Brain surface color: Color picker allowing any RGB color selection
- Wireframe mode: Renders only polygon edges instead of filled faces
- Emissive intensity (0-2): Makes the surface self-luminous, useful for seeing internal structure
- Metalness (0-1): Adjusts PBR material metallic reflectance
- Roughness (0-1): Controls surface micro-facet scattering
- Streamline visibility: Shows/hides white matter tracts independently of surface
- Streamline opacity, color mapping, line width controls

These controls enable users to emphasize different anatomical features depending on their analysis needs.

**Keyboard Shortcuts**:

12 keyboard hotkeys enable rapid scene manipulation:
- R: Reset camera to default position
- B: Toggle brain surface visibility
- T: Toggle streamline (tract) visibility
- L: Toggle anatomical labels
- S: Toggle slice viewer overlays
- A: Toggle auto-rotation
- W: Toggle wireframe mode
- 1/2/3: Switch between hologram/point cloud/MRI mesh models
- G: Toggle performance statistics (FPS, memory usage)

**WebGL Context Management**:

The large GLB models can exhaust GPU memory on lower-end hardware. The system handles webglcontextlost events by displaying an error message with a retry button. When the context is restored (webglcontextrestored), rendering resumes automatically. The Canvas is configured with device pixel ratio limited to [1, 1.5] and anti-aliasing enabled to balance quality and performance.

**Scene Information Display**:

An overlay shows real-time statistics:
- Streamline count (displayed / total in file)
- Total points rendered
- Mean streamline length in millimeters
- Mesh vertex count if using MRI mesh model

This helps users understand the data density and confirm correct loading.

### Analysis Page

The analysis page adapts its entire feature set based on the selected user type, presenting the same underlying data with different levels of detail and interpretation.

**Common Elements Across All User Types**:

All users see:
1. A text summary generated based on global metrics, explaining the overall network organization in language appropriate to their expertise level
2. Metric cards displaying key connectivity measures with status indicators (normal, elevated, reduced)
3. Community structure visualization showing how brain regions cluster into functional modules

**Clinician Mode (Advanced Analysis)**:

Designed for medical professionals making clinical decisions based on connectivity data:

*Normative Z-Score Analysis*:
Compares the subject's global metrics against normative ranges derived from published healthy control populations. Each metric receives a Z-score: (observed - population_mean) / population_SD. Values beyond ±2 standard deviations are flagged as potentially abnormal. A bar chart displays Z-scores with reference lines at ±2SD.

Example normative ranges (source: published structural connectome studies):
- Clustering coefficient: 0.40 ± 0.10
- Path length: 3.0 ± 0.5
- Global efficiency: 0.50 ± 0.10
- Modularity: 0.35 ± 0.08

*Degree Distribution Analysis*:
A histogram shows how connection counts are distributed across brain regions. Healthy brains typically follow a truncated power-law distribution where most regions have moderate degree and few regions are highly connected hubs. Deviations may indicate disrupted network topology.

A degree-betweenness scatter plot identifies which high-degree regions also serve as critical routing hubs. Marker size encodes node strength and color encodes closeness centrality, creating a multi-dimensional view.

*Inter-Hemispheric Comparison*:
Side-by-side box plots compare connectivity metrics between left and right hemisphere regions. The system computes mean degree and node strength separately for each hemisphere. Significant asymmetry may indicate lateralized pathology or naturally lateralized functions like language.

*Lobe-wise Aggregate Metrics*:
A table breaks down connectivity by cortical lobe:
- Frontal: executive function, decision-making
- Parietal: sensory integration, spatial attention
- Temporal: memory, language processing
- Occipital: visual processing

For each lobe, the table shows: region count, average degree, maximum degree, average node strength, average betweenness. This helps localize connectivity changes to specific functional systems.

*Network Vulnerability Index*:
Ranks brain regions by vulnerability score = betweenness × degree. High-vulnerability hubs are single points of failure whose disruption would maximally fragment the network. The top 15 regions are displayed with progress bars showing relative vulnerability.

This is clinically relevant for predicting functional consequences of lesions in specific locations.

*Rich Club Coefficient Curve*:
Plots rich club coefficient φ(k) as a function of degree threshold k. If the curve exceeds normalized thresholds (typically φ(k) > 1.2), it indicates that hub regions preferentially interconnect beyond chance expectation. Loss of rich club organization is associated with cognitive impairment in several disorders.

*Hub Regions Table*:
Lists the top 15 most connected regions with full anatomical names, abbreviations, degree, betweenness, node strength, and eigenvector centrality. This identifies the structural core of the brain network.

*Complete Nodal Metric Charts*:
Six Plotly.js charts showing all nodal metrics across all brain regions with anatomical abbreviations on the x-axis. Users can zoom, pan, and hover to see exact values. The charts include:
1. Degree bar chart
2. Betweenness centrality scatter
3. Closeness centrality scatter
4. Local efficiency line plot
5. Node strength bar chart
6. Eigenvector centrality scatter

*Full Connectome Matrix*:
A heatmap showing the complete N×N connectivity matrix with region labels on both axes. Controls include:
- Log scale toggle: Applies log(1+weight) transformation to reveal weak connections
- Threshold slider: Filters edges below a strength threshold to focus on major pathways
- Hover labels with region names and edge weights

**Student Mode (Educational Analysis)**:

Designed for students learning about brain networks and graph theory:

*Learning Center*:
A three-tab interface provides self-contained educational resources:

**Learn Concepts Tab**: Five expandable accordion panels with detailed explanations:
1. Graph Theory Basics: Explains nodes, edges, weights, and why the brain can be modeled as a network
2. Small-World Networks: Describes the balance between local clustering and global efficiency that characterizes biological networks
3. Community Structure & Modularity: Explains how regions organize into functional modules and what modularity quantifies
4. Hub Regions & Centrality: Defines different centrality metrics and why certain regions are disproportionately important
5. Network Efficiency: Distinguishes local and local efficiency and their biological significance

**Metric Correlations Tab**: Interactive scatter plots showing relationships between metrics:
- Degree vs. Node Strength: Tests whether highly connected regions also have strong connections (typically r = 0.7-0.9)
- Degree vs. Betweenness: Tests whether hubs are also routing bottlenecks (typically r = 0.4-0.6)

Each plot computes and displays Pearson correlation coefficient with interpretation:
- r > 0.7: Strong positive correlation
- 0.3 < r < 0.7: Moderate correlation
- r < 0.3: Weak correlation

**Network Comparison Tab**: Compares the brain network to an equivalent Erdos-Renyi random network with the same number of nodes and edges:
- Shows clustering coefficient, efficiency, and modularity side-by-side in a grouped bar chart
- Displays ratio values (brain/random) - brain networks typically show 10x higher clustering while maintaining similar path length
- Explains why small-world organization arises from this combination

*Educational Interpretations*:
Every metric card includes extended tooltips explaining what the metric measures, how it's computed, and what deviations might indicate.

*Hub Regions with Context*:
The top 10 hub regions are shown (instead of 15 in clinician mode) with additional text explaining why hubs are important for network function and what happens when they're damaged.

**General User Mode (Simplified Brain Health)**:

Designed for patients or general audiences without neuroscience training:

*Brain Network Health Overview*:
A large circular progress indicator displays an overall health percentage (0-100%) computed as: (number of normal indicators / total indicators) × 100%. The circle is color-coded:
- Green (80-100%): Healthy connectivity patterns
- Yellow (50-79%): Some indicators outside normal ranges
- Red (<50%): Multiple connectivity concerns

*Five Health Indicators*:
Each indicator uses traffic-light color coding (green/yellow/red) and includes:

1. *Network Organization*: Based on clustering coefficient. Green if clustering ≥ 0.2. Plain-language explanation: "Your brain regions are well-organized into local groups that work together efficiently" vs. "The organization of local brain region groups may benefit from further review."

2. *Communication Speed*: Based on characteristic path length. Green if path length is 2.5-4.0. Explanation: "Information can travel between brain regions efficiently" vs. "Communication paths are longer than typical, which may affect processing speed."

3. *Overall Efficiency*: Based on global efficiency. Green if efficiency ≥ 0.3. Explanation: "Your brain network shows good overall information transfer efficiency."

4. *Brain Modules*: Based on community count. Green if 3-10 communities detected. Explanation: "Your brain is organized into [N] distinct modules, which is typical for healthy brains. These likely correspond to different functional systems (vision, movement, thinking, etc.)."

5. *Network Balance*: Based on small-worldness. Green if σ > 1.0. Explanation: "Your brain has 'small-world' organization — it efficiently balances local specialized processing with global communication. This is the hallmark of a well-organized brain."

*What This Means Section*:
A plain-language explanation panel describes:
- What white matter pathways are
- How the analysis was performed (analyzed [N] brain regions)
- The city analogy: neighborhoods (clusters) connected by highways (long-range connections)
- What the traffic lights mean
- Medical disclaimer that this is not a diagnosis and should be interpreted by a healthcare professional

*Simplified Metric Cards*:
Shows only 4 key metrics (clustering, path length, efficiency, modularity) instead of 8.

*Restricted Exports*:
Offers only metrics JSON export. Connectome and streamline downloads are hidden with a prompt to switch to Researcher mode for advanced export options.

## Technical Architecture and Design Decisions

### Backend Technology Choices

**Python 3.11**:
Selected for mature scientific computing ecosystem (NumPy, SciPy) and widespread adoption in neuroimaging (Nibabel, DIPY). Type hints improve code maintainability.

**FastAPI**:
Chosen over Flask or Django for automatic Pydantic data validation, async support, and performance comparable to Node.js.

**DIPY**:
The most comprehensive open-source Python diffusion MRI library. Provides validated implementations of DTI, CSD, tractography, and streamline I/O. Active development and extensive documentation made it preferable to custom implementations.

**NetworkX vs. igraph**:
NetworkX used for most graph operations due to pure-Python implementation and extensive metric coverage. python-igraph with leidenalg added specifically for Leiden community detection, which is more accurate than Louvain but requires C++ compiled code.

**Memory-Mapped I/O**:
Large NIfTI volumes (500MB+) loaded via numpy.memmap instead of loading entirely into RAM. This trades I/O performance for memory footprint, enabling processing on systems.

**Numba JIT Compilation**:
The RK4 integration loop (executed millions of times during tractography) annotated with @numba.jit for 10-50x speedup versus pure Python. First execution compiles the function, subsequent calls run at C speed.

### Frontend Technology Choices

**Next.js 14**:
Provides server-side rendering for initial page load, file-based routing, automatic code splitting, and optimized production builds. The App Router with React Server Components reduces JavaScript bundle size.

**Three.js + React Three Fiber**:
Three.js is the de facto standard for WebGL rendering in browsers. React Three Fiber wraps Three.js with React components, enabling declarative scene construction. The @react-three/drei library provides pre-built helpers for camera controls, model loading, and HTML overlays.

**Zustand State Management**:
Lighter-weight than Redux with simpler API. State persists to localStorage automatically for user preferences (user type, viewer settings). Middleware pattern for devtools integration during development.

**Plotly.js for Charts**:
Provides interactive scientific visualizations (scatter, box plots, histograms, heatmaps) with zoom, pan, and hover tooltips. Designed for research data visualization unlike Chart.js.

**Tailwind CSS**:
Utility-first CSS framework enables rapid styling without context switching to CSS files. The JIT compiler generates only the classes actually used, resulting in small production CSS bundles.

**TypeScript**:
Strict typing prevents entire classes of runtime errors and improves IDE autocomplete. Interface definitions in lib/types.ts ensure frontend-backend contract is enforced at compile time.

### Design Patterns

**Separation of Concerns**:
Backend handles all computation, frontend handles only presentation. The REST API provides a clean boundary allowing either component to be replaced independently.

**Progressive Enhancement**:
Core functionality works with JavaScript disabled (server-rendered HTML). Interactive features (3D viewer, charts) enhance the experience when WebGL and JavaScript are available.

**Graceful Degradation**:
Missing optional dependencies (FSL) trigger automatic fallback to alternative implementations (DIPY). Large GLB models that fail to load due to memory constraints fall back to smaller MRI mesh model.

**Adapter Pattern**:
The analysis page renders different components (AdvancedMetrics, EducationalPanel, BrainHealthSummary) based on user type while consuming the same underlying GraphMetrics data structure.

**Repository Pattern**:
The API client (lib/api.ts) abstracts HTTP requests behind typed methods. If the backend changes (e.g., GraphQL instead of REST), only this file changes.

## Project Strengths and Limitations

### Strengths

**1. Accessibility**: Runs on consumer hardware without requiring expensive compute infrastructure. The web interface eliminates software installation barriers for end users.

**2. Automation**: End-to-end pipeline with validated defaults allows non-experts to process data without understanding every parameter.

**3. Transparency**: Every processing decision is logged. Users can review exactly what operations were performed and with what parameters.

**4. Extensibility**: Modular architecture allows adding new tractography algorithms, graph metrics, or visualization modes without rewriting core systems.

**5. Educational Value**: Multi-level interface teaches graph theory concepts while enabling analysis, making it suitable for classroom use.

**6. Open Source**: Complete source code allows validation of algorithms, customization for specific research needs, and contribution of improvements.

**7. Modern Web Stack**: Uses current web standards (WebGL 2, ES2020) ensuring browser compatibility for the next several years.

**8. Validated Algorithms**: Uses algorithms from DIPY and NetworkX rather than custom implementations, ensuring correctness against literature.

### Limitations

**1. Processing Speed**: Python is slower than optimized C++ implementations (MRtrix3). Full pipeline takes 20-45 minutes versus 10-20 minutes for comparable specialized tools. This is the price paid for code readability and maintainability.

**2. Memory Scaling**: The 16GB minimum handles typical quality scans (2mm isotropic, 64 directions). High-resolution clinical scans (1mm, 128+ directions) may require 32GB or memory-mapped mode sacrifices speed.

**3. Limited Tractography Algorithms**: Currently supports only probabilistic CSD-based tracking. Deterministic tracking, global tracking, and particle filtering methods are not yet implemented.

**4. GPU Acceleration**: Tractography runs on CPU only. GPU-accelerated implementations (CUDA) could achieve 10-100x speedup but would require NVIDIA hardware and add deployment complexity.

**5. Browser Performance**: Rendering 3,000 streamlines in WebGL is the limit for maintaining 60 FPS on integrated graphics. Desktop applications using OpenGL could handle the full 100,000+ streamline dataset.

**6. Single-Subject Analysis**: Group analysis (comparing multiple subjects, statistical testing) is not implemented. Each subject is processed independently.

**7. Longitudinal Tracking**: No support for processing multiple timepoints from the same subject or computing change metrics over time.

**8. Limited File Format Support**: Inputs must be NIfTI. DICOM import requires manual conversion using external tools (dcm2niix). Siemens/Philips proprietary formats are not supported.

**9. Manual Upload Workflow**: Current implementation requires users to upload files then manually run the pipeline. Automatic processing upon upload is not yet implemented.

**10. No User Authentication**: The web interface has no login system. Anyone with network access can view all results. This precludes deployment with real patient data without additional security.

**11. Limited Cross-Platform Testing**: Developed and tested primarily on Windows and Linux. macOS compatibility is likely but not comprehensively verified.

**12. Single Backend Instance**: The FastAPI server is single-threaded and blocks during processing. Multiple simultaneous users would experience degraded performance. A production deployment would need task queue (Celery) and multiple worker processes.

## Potential Applications

**Clinical Neurology**: Pre-surgical planning for brain tumor or epilepsy surgery to identify critical white matter bundles that must be preserved.

**Neurological Research**: Quantifying connectivity changes in Alzheimer's, multiple sclerosis, traumatic brain injury, or psychiatric disorders across patient cohorts.

**Aging Studies**: Tracking white matter decline with normal aging to distinguish healthy aging from prodromal dementia.

**Brain Development**: Mapping maturation of white matter networks from childhood through adolescence.

**Education**: Teaching neuroanatomy and graph theory concepts to medical students, neuroscience students, and radiology residents.

**Methodology Research**: Developing and validating new tractography algorithms, connectivity metrics, or parcellation schemes.

## Comparison to Existing Tools

**MRtrix3** (Command-line tractography suite):
MRtrix3 provides faster processing (C++ implementation) and more algorithm options (global tracking, SIFT filtering). NeuroTract offers integrated visualization, adaptive interface, and web accessibility. Use MRtrix3 for large-scale batch processing; use NeuroTract for interactive exploration and education.

**DSI Studio** (Desktop application):
DSI Studio has a comprehensive GUI and supports additional diffusion models (GQI, QSDR). NeuroTract's web interface requires no installation and provides multi-level adaptive analysis. DSI Studio is better for advanced diffusion modeling; NeuroTract is better for standard workflows and education.

**Connectome Workbench** (HCP visualization tool):
Workbench excels at surface-based analysis and cifti file formats. NeuroTract focuses on volumetric tractography and provides graph analysis not available in Workbench.

**TrackVis/DiffusionToolkit** (Legacy tools):
TrackVis pioneered tractography visualization but is no longer maintained. NeuroTract provides modern web-based rendering with better performance and active development.

**BrainSuite** (Commercial platform):
BrainSuite offers validated preprocessing pipelines and requires paid licenses. NeuroTract is open source and free but may have fewer validation studies.

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04 |
| CPU | Intel i5 or AMD Ryzen 5, 4 cores | Intel i7 or AMD Ryzen 7, 8 cores |
| RAM | 16 GB | 32 GB |
| Storage | 500 GB free | 1 TB SSD |
| GPU | Integrated graphics with WebGL 2.0 | Dedicated GPU (NVIDIA/AMD) with 2GB+ VRAM |
| Python | 3.10 or later | 3.11 |
| Node.js | 18 or later | 20 LTS |
| Browser | Chrome 100+, Firefox 100+ | Chrome 120+ |

### Backend Setup (Python)

**Option 1: Conda (Recommended for avoiding dependency conflicts)**

```bash
git clone https://github.com/sailwalpranjal/Neurotract.git
cd Neurotract

conda env create -f environment.yml
conda activate Neurotract

python -m src.backend.cli --version
```

**Option 2: pip with virtual environment**

```bash
git clone https://github.com/sailwalpranjal/Neurotract.git
cd Neurotract

python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m src.backend.cli --version
```

### Frontend Setup (Node.js)

```bash
cd src/frontend
npm install
```

### Verifying Installation

```bash
# Check backend
python -m src.backend.cli --version
# Should print: NeuroTract 0.1.0

# Check frontend build
cd src/frontend
npm run build
# Should complete with 0 errors (warnings about exhaustive-deps are expected)
```

## Quick Start: Processing Your First Dataset

### Using Included Test Data

The repository includes sample data from the Stanford HARDI dataset. Process it end-to-end:

```bash
# Ensure .venv is activated
# All commands run from project root

# Step 1: Preprocessing (5-10 minutes)
python -m src.backend.cli preprocess \
  --input "datasets/Stanford dataset/SUB1_b1000_1.nii.gz" \
  --bvals "datasets/Stanford dataset/SUB1_b1000_1.bvals" \
  --bvecs "datasets/Stanford dataset/SUB1_b1000_1.bvecs" \
  --output "output/SUB1/preprocessed" \
  --no-motion-correction

# Step 2: DTI computation (2-3 minutes)
python -m src.backend.cli dti \
  --input "output/SUB1/preprocessed/preprocessed_dwi.nii.gz" \
  --bvals "output/SUB1/preprocessed/preprocessed_dwi.bval" \
  --bvecs "output/SUB1/preprocessed/preprocessed_dwi.bvec" \
  --mask "output/SUB1/preprocessed/preprocessed_brain_mask.nii.gz" \
  --output "output/SUB1/dti"

# Step 3: CSD/FOD estimation (10-15 minutes)
python -m src.backend.cli csd \
  --input "output/SUB1/preprocessed/preprocessed_dwi.nii.gz" \
  --bvals "output/SUB1/preprocessed/preprocessed_dwi.bval" \
  --bvecs "output/SUB1/preprocessed/preprocessed_dwi.bvec" \
  --mask "output/SUB1/preprocessed/preprocessed_brain_mask.nii.gz" \
  --output "output/SUB1/fod.nii.gz"

# Step 4: Tractography (2-4 minutes)
python -m src.backend.cli tractography \
  --fod "output/SUB1/fod.nii.gz" \
  --mask "output/SUB1/preprocessed/preprocessed_brain_mask.nii.gz" \
  --fa-map "output/SUB1/dti/dti_fa.nii.gz" \
  --seeds-per-voxel 2 \
  --step-size 0.5 \
  --max-angle 30 \
  --fa-threshold 0.1 \
  --output "output/SUB1/streamlines.trk"

# Step 5: Connectome construction (5-8 minutes)
python -m src.backend.cli connectome \
  --streamlines "output/SUB1/streamlines.trk" \
  --parcellation "datasets/Stanford dataset/SUB1_aparc-reduced.nii.gz" \
  --weighting count \
  --output "output/SUB1/connectome.npy"

# Step 6: Graph metrics (1-2 minutes)
python -m src.backend.cli metrics \
  --connectome "output/SUB1/connectome.npy" \
  --output "output/SUB1/metrics.json"

# Step 7: View results
python -c "import json; print(json.dumps(json.load(open('output/SUB1/metrics.json')), indent=2))"
```

### Launching the Web Interface

Open two terminal windows:

**Terminal 1: Backend server**
```bash
# From project root with .venv activated
python -m uvicorn src.backend.api.server:app --host 0.0.0.0 --port 8000

# Output shows:
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

**Terminal 2: Frontend dev server**
```bash
cd src/frontend
npm run dev

# Output shows:
# ready - started server on 0.0.0.0:3000
```

Open http://localhost:3000 in your browser. The home page will:
1. Detect the backend at localhost:8000 (green status indicator)
2. Display SUB1 in the available subjects list
3. Show data availability badges (streamlines, DTI, FOD, connectome, metrics all green)

Click "Load & View" to open the 3D viewer showing SUB1's white matter tracts.

## API Reference

The REST API exposes all pipeline results through HTTP endpoints. Full interactive documentation is available at http://localhost:8000/docs when the server is running.

### Health Check
```
GET /health
```
Returns: `{"status": "ok", "version": "0.1.0"}`

### List Available Results
```
GET /results/available
```
Returns array of processed subjects with metadata:
```json
[
  {
    "subject_id": "SUB1",
    "has_streamlines": true,
    "has_metrics": true,
    "has_connectome": true,
    "has_dti": true,
    "has_fod": true,
    "streamline_stats": {
      "bundle_statistics": {
        "n_streamlines": 157289,
        "mean_length": 82.4,
        "min_length": 10.2,
        "max_length": 198.7
      }
    },
    "connectome_info": {
      "n_parcels": 89,
      "n_edges": 3917,
      "density": 0.512
    }
  }
]
```

### Get Streamlines
```
GET /results/{subject_id}/streamlines?subsample=3000
```
Returns subsampled streamlines for visualization:
```json
{
  "streamlines": [
    {
      "points": [x1, y1, z1, x2, y2, z2, ...],
      "numPoints": 34,
      "color": [0.8, 0.3, 0.1]
    }
  ],
  "bounds": {
    "min": [-82.3, -116.4, -72.1],
    "max": [85.7, 119.2, 98.5]
  },
  "metadata": {
    "count": 3000,
    "totalInFile": 157289,
    "totalPoints": 98450,
    "meanLength": 82.4
  }
}
```

### Get Graph Metrics
```
GET /results/{subject_id}/metrics
```
Returns complete graph metrics JSON matching the structure saved by the CLI metrics command.

### Get Connectome Matrix
```
GET /results/{subject_id}/connectome
```
Returns NxN connectivity matrix as 2D array.

### Get Brain Mesh
```
GET /results/{subject_id}/brain-mesh
```
Returns marching cubes mesh:
```json
{
  "vertices": [x1,y1,z1, x2,y2,z2, ...],
  "faces": [i1,j1,k1, i2,j2,k2, ...],
  "normals": [nx1,ny1,nz1, ...],
  "metadata": {
    "n_vertices": 145678,
    "n_faces": 291352,
    "bounds": {...},
    "source": "brain_mask.nii.gz"
  }
}
```

### Get Parcellation Labels
```
GET /results/{subject_id}/parcellation-labels
```
Returns anatomical region mappings:
```json
{
  "labels": [
    {
      "index": 0,
      "generic_name": "Unknown",
      "anatomical_name": "Background",
      "abbreviation": "BG",
      "hemisphere": "midline",
      "lobe": "none",
      "description": "Non-brain background"
    },
    {
      "index": 1,
      "generic_name": "ctx-lh-superiorfrontal",
      "anatomical_name": "Left Superior Frontal Gyrus",
      "abbreviation": "L-SFG",
      "hemisphere": "left",
      "lobe": "frontal",
      "description": "Executive function and working memory"
    }
  ],
  "atlas": "Desikan-Killiany"
}
```

### Download Files
```
GET /results/{subject_id}/download/{filename}
```
Supported filenames: `metrics.json`, `connectome.csv`, `streamlines.trk`
Returns file with appropriate content-type headers for browser download.

## Troubleshooting

### Backend Errors

**ImportError: No module named 'dipy'**
The virtual environment is not activated. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac).

**Out of memory during tractography**
Reduce seeds per voxel: `--seeds-per-voxel 1` instead of 2. Or use the low-memory mode: `python -m src.backend.cli run --mode lowmem`

**ValueError: b-values and b-vectors length mismatch**
The .bval and .bvec files must have the same number of entries as volumes in the .nii.gz file. Check file integrity and ensure they correspond to the same acquisition.

**Backend 404 errors on /results/SUB1/brain-mesh**
The brain mesh endpoint requires that the full pipeline has been run including surface generation. If you ran individual steps manually, the mesh file may not exist in output/SUB1/. Check for the presence of brain_mesh.json.

### Frontend Errors

**Connection Error: Cannot reach backend**
The FastAPI server is not running. Start it with: `python -m uvicorn src.backend.api.server:app --host 0.0.0.0 --port 8000`
If using a different port, update `NEXT_PUBLIC_API_URL` in src/frontend/.env.local.

**WebGL context lost error in viewer**
The browser GPU memory is exhausted, often due to the large point cloud model (33MB). Click Retry to restore the context, or switch to a lighter model (hologram or MRI mesh) using keyboard shortcut 1 or 3.

**Hydration mismatch warning in console**
This is expected for components that use localStorage (UserTypeSelector, Header badges). The server renders a default state while the client renders the persisted value. The mounted state pattern prevents visual flicker. This does not affect functionality.

**Build fails with TypeScript errors**
Run `cd src/frontend && npm run build` to see specific errors. Most common issues:
- Undefined property access: Add optional chaining (`?.`) or null checks
- Type mismatches: Ensure API response types match lib/types.ts interfaces

**Plotly charts not rendering**
The react-plotly.js package requires a custom type declaration. Ensure types/react-plotly.d.ts exists. If charts appear blank, check browser console for JavaScript errors.

**Streamlines appear outside brain model**
This indicates coordinate space mismatch. Ensure you're using the latest version where BrainModel.tsx receives streamlineBounds prop and aligns accordingly. If the issue persists, verify that the streamline .trk file and NIfTI files share the same coordinate system.

### Performance Issues

**Slow page load times**
The GLB models (especially point cloud at 33MB) require significant download time. Consider hosting them on a CDN or preloading in the background.

**Low frame rate in viewer**
Reduce the number of visible streamlines using the level-of-detail setting in Controls. Lower the device pixel ratio by editing the dpr prop in BrainViewer.tsx Canvas component.

**Slow graph metric computation**
Large connectomes (100+ regions) with high connection density can take several minutes for betweenness centrality computation (O(n³) complexity). This is inherent to the algorithm. Consider using approximate betweenness methods for very large networks.

## Development and Contributing

### Code Style

**Python**:
- black for formatting (line length 100)
- flake8 for linting
- mypy for type checking
- pytest for unit tests

**TypeScript**:
- ESLint with Next.js config
- Prettier for formatting
- React hooks exhaustive-deps warnings are expected and documented

### Running Tests

```bash
# Python tests
pytest tests/ -v --cov=src --cov-report=html

# Specific module
pytest tests/test_tractography.py -v

# Frontend type check
cd src/frontend
npx tsc --noEmit

# Frontend build test
npm run build
```

### Adding New Features

**New Tractography Algorithm**:
1. Implement in `src/backend/tractography/`
2. Add CLI argument to `cli.py`
3. Update API endpoint in `server.py`
4. Add tests in `tests/test_tractography.py`

**New Graph Metric**:
1. Implement in `src/backend/connectome/metrics.py`
2. Add to GraphMetrics interface in `src/frontend/lib/types.ts`
3. Update interpretation in `src/frontend/lib/interpretations.ts`
4. Add visualization in analysis page components

**New Visualization**:
1. Create component in `src/frontend/components/viewer/`
2. Add controls in `Controls.tsx`
3. Integrate into `BrainViewer.tsx` scene
4. Document keyboard shortcut if applicable

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-algorithm`
3. Make changes with descriptive commits
4. Ensure all tests pass and build succeeds
5. Update documentation if adding user-facing features
6. Submit PR with clear description of changes and rationale

## Project Status and Future Roadmap

**Current Version**: 0.1.0 (Initial Release)

**Completed Features**:
- Full 7-stage processing pipeline
- Three brain rendering modes
- Multi-user adaptive analysis interface
- REST API with 31 endpoints
- Comprehensive graph metrics
- Interactive 3D visualization
- Anatomical parcellation mapping

**Planned Features (Version 0.2)**:
- Automatic pipeline execution upon file upload
- Background job processing with real-time progress updates
- User authentication and result privacy
- Group-level statistical analysis
- Longitudinal change tracking (multiple timepoints per subject)
- DICOM import support (integration with dcm2niix)
- GPU-accelerated tractography (CUDA implementation)
- Additional tractography algorithms (deterministic, global tracking)
- Export to BIDS derivatives format

**Long-term Vision (Version 1.0)**:
- Multi-tenant cloud deployment with user accounts
- Automated quality control with ML-based artifact detection
- Pathology detection models for Alzheimer's, MS, TBI
- Integration with electronic health records (FHIR)
- Mobile-responsive viewer for tablet devices
- Collaborative annotation and reporting tools
- Public data repository with thousands of processed connectomes

## License

MIT License

Copyright (c) 2026 NeuroTract Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation

If you use NeuroTract in academic research, please cite:

```bibtex
@software{neurotract2026,
  title = {NeuroTract: Brain White Matter Tractography and Connectivity Analysis Platform},
  author = {NeuroTract Contributors},
  year = {2026},
  url = {https://github.com/sailwalpranjal/Neurotract},
  version = {0.1.0}
}
```

## Acknowledgments

This project builds upon foundational work by the neuroimaging community:

- DIPY development team for diffusion MRI algorithms
- Nibabel maintainers for neuroimaging file I/O
- NetworkX and python-igraph teams for graph analysis tools
- FastAPI framework creators
- Three.js and React Three Fiber communities
- Desikan, Killiany, and FreeSurfer teams for the parcellation atlas
