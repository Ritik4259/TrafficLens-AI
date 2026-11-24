# ADR 0004: Multimodal Scene Understanding via Vision-Language Model Integration

## Status

Accepted

## Context

Phase 1 through Phase 4 of TrafficLens-AI established a high-performing computer vision, tracking, and spatiotemporal analytics pipeline (YOLO11n detector, ByteTrack tracker, directional tripwire counters, tumbling window flow calculation, polygonal ROI density estimation, and stationary vehicle detection). The system exposes verified API contracts and delivers synchronized multi-stream video inspection via the frontend workspace.

However, classical computer vision models and rule-based heuristics cannot provide qualitative, semantic, and contextual scene reasoning. For example:
- The tracker records a stationary vehicle event (track #12 stationary for 14.5 seconds at coordinate `[x, y]`), but cannot determine *why* it stopped (e.g., breakdown on the shoulder, waiting at an unconfigured traffic light, delivery unloading, queue spillback, or debris on the roadway).
- The pipeline measures vehicle flow and spatial occupancy, but cannot describe roadway geometry, environmental conditions (rain, wet reflections, twilight glare, fog), or visual anomalies (lane blockages, construction cones, emergency service presence).
- Operational end-users require natural-language scene descriptions and evidence-grounded answers to contextual queries regarding specific sequence intervals.

At the same time, integrating Vision-Language Models (VLMs) into an engineering-grade traffic platform introduces severe failure modes:
1. **Hallucination of Numerical Facts**: VLMs notoriously fabricate vehicle counts, speeds, trajectories, and timestamps if asked to perform raw perceptual accounting.
2. **Resource Exhaustion on Constrained Hardware**: Our local deployment environment is strictly resource-bounded (Intel CPU with 8 GB system RAM with ~1 GB free under development load, and an NVIDIA GeForce RTX 2050 Laptop GPU with 4 GB VRAM; active PyTorch runtime is CPU-only). Running local 2B+ parameter models in FP16 or without quantization will lead to out-of-memory (OOM) crashes or system unresponsiveness.
3. **Vendor & Runtime Coupling**: Hardcoding specific inference libraries (`transformers`, `ollama`, or vendor APIs) into FastAPI route handlers or CV pipelines destroys maintainability, testability, and portability.
4. **Epistemic Conflation**: Blurring the boundary between deterministic CV measurements and probabilistic VLM interpretations undermines trust in automated analytics.

## Decision

We adopt a decoupled, evidence-grounded **Multimodal Scene Understanding Architecture** governed by the following core decisions:

### 1. Strict Epistemic Separation of Responsibilities

The system enforces a rigid functional boundary between the Computer Vision Pipeline and the Vision-Language Model:
- **CV Pipeline (Source of Numerical Truth)**: Exclusively owns vehicle detection, object localization, multi-object tracking, identity persistence, directional counting, flow calculations (VPM), spatial occupancy ratios, and stationary duration timing. The VLM is strictly forbidden from overriding or recalculating these metrics.
- **VLM (Source of Qualitative & Contextual Interpretation)**: Enriches the deterministic telemetry by interpreting environmental factors (weather, road surface wetness, lighting), describing static scene geometry (lane layout, curbs, dividers), explaining probable root causes for stopped-vehicle events, and answering user queries grounded in visual evidence.
- **Categorization of Claims**: Every claim produced by the VLM must be classified into one of four distinct epistemic categories in the output schema:
  - `CV-derived fact`: Direct citations of tracker IDs, timestamps, and measured durations.
  - `VLM visual observation`: Directly verifiable visual features in keyframes (e.g., hazard lights flashing, vehicle on shoulder, asphalt reflections).
  - `Contextual inference`: Plausible hypotheses derived from visual evidence and telemetry (e.g., "vehicle likely waiting for queue dissipation").
  - `Unknown / unverified`: Explicit disclosures of information that cannot be determined from the available camera view (e.g., signal phase out of frame, driver status).

### 2. Deterministic Multimodal Evidence Pipeline

Rather than streaming entire raw video sequences into the VLM (which is computationally prohibitive and context-inefficient), we introduce an upstream **Keyframe Extractor** and **Evidence Packager**:
- **Deterministic Keyframe Selection**:
  - *Event-Triggered Frames*: Captured at the exact `start_timestamp_s` and `end_timestamp_s` of stationary vehicle events, and at transitions into `high` congestion states.
  - *Periodic Anchor Frames*: Sampled at uniform intervals (e.g., every 10 seconds) to anchor baseline scene appearance.
  - *Duplicate & Redundancy Suppression*: Keyframes closer than $\Delta t < 1.0\text{ s}$ or exhibiting negligible optical displacement are pruned.
  - *Strict Frame Budget*: A hard cap of at most $K = 5$ keyframes per analysis request ensures predictable context length, fast inference, and bounded token expenditure.
- **Structured Evidence Package**:
  The VLM receives a compact, strongly-typed JSON package alongside the extracted keyframe images. The package summarizes sequence metadata, camera parameters, active event descriptions, relevant line flow windows, and spatial bounding boxes of interest.

### 3. Model-Agnostic Provider Abstraction (`VLMProvider`)

We define an abstract Python Protocol (`VLMProvider`) residing in `backend/app/services/vlm/provider.py`:
- The provider contract exposes a single asynchronous method:
  `async def generate_scene_understanding(evidence_package: EvidencePackage) -> SceneInterpretationResponseSchema`
- Concrete provider implementations are decoupled from application business logic:
  - `OpenAICompatibleRemoteProvider`: Interacts with OpenAI-compatible endpoints (vLLM, Ollama, Hugging Face TGI, or cloud multimodal endpoints like Gemini / Claude / GPT-4o-mini).
  - `HuggingFaceLocalProvider`: Direct in-process execution using `transformers` and `bitsandbytes` (for local GPU/CPU execution when configured).
  - `MockVLMProvider`: Deterministic synthetic provider for unit and regression testing without external network or GPU requirements.
- The active provider is dynamically resolved via backend settings (`VLM_PROVIDER_TYPE`), ensuring zero vendor lock-in.

### 4. Model Candidate Selection & Hardware Realism

We evaluated three 2B-class candidate models:
1. `Qwen/Qwen3.5-2B`: High-efficiency text/MoE foundation model. Lacks native vision encoder specialization for fine-grained spatial traffic perception.
2. `Qwen/Qwen3-VL-2B-Instruct`: Native multimodal vision-language model featuring Interleaved-MRoPE for dynamic video understanding, spatial bounding-box grounding, and 256K native context.
3. `HuggingFaceTB/SmolVLM2-2.2B-Instruct`: Compact vision-language model pairing a SigLIP vision encoder with SmolLM2-1.7B, natively supporting multi-frame image/video inputs under Apache 2.0.

**Hardware Feasibility Conclusion**:
Our development machine (4 GB RTX 2050 Laptop GPU, 8 GB total system RAM with ~1 GB free, CPU-only active PyTorch) cannot reliably support local FP16 2B inference. Even under 4-bit quantization (requiring ~1.5 GB weights), multi-image visual token expansion ($5 \text{ keyframes} \times 729 \text{ tokens} \approx 3,645 \text{ tokens}$) pushes activation memory and KV caches beyond safe operating limits on a 4 GB GPU, while CPU execution causes severe system swapping and 30–60s latencies.

**Recommendation**:
- **Primary Architecture**: `Qwen/Qwen3-VL-2B-Instruct` deployed via an **OpenAI-compatible remote/hosted inference endpoint** (or self-hosted vLLM/Ollama instance on accelerated infrastructure, or hosted cloud multimodal API).
- **Fallback / Local Edge Option**: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` using 4-bit INT4 quantization (`bitsandbytes`) for environments requiring strictly air-gapped, zero-network operation.

### 5. Decoupled Processing Lifecycle

VLM scene interpretation is designed as an **independent, on-demand or asynchronous enrichment stage**:
- The core CV detection, tracking, and analytics pipeline executes to completion without waiting for or depending upon the VLM.
- VLM scene interpretation is invoked either:
  a. As an optional post-processing step during job execution if `enable_vlm_analysis=True` is requested.
  b. On-demand via `POST /api/v1/jobs/{job_id}/interpretations` when a user requests contextual scene understanding for an already completed job.
- Failure of the VLM provider (timeout, API error, rate limit) does not invalidate the CV tracking results or fail the primary job.

## Considered Options

1. **Directly Feeding Video Streams into an End-to-End VLM (Zero CV Pipeline)**:
   - *Rejected*: Current VLMs exhibit high error rates in vehicle counting, cannot reliably maintain multi-object tracking identities over hundreds of frames, and cannot provide calibrated spatial occupancy or tripwire metrics.
2. **In-Process Monolithic Local Inference Only**:
   - *Rejected*: Disregards the physical 4 GB VRAM / 8 GB RAM hardware reality of student and edge laptop environments, resulting in brittle crashes and poor developer experience.
3. **Hardcoding Proprietary Cloud APIs into Route Handlers**:
   - *Rejected*: Destroys modularity, prevents offline testing, and couples the repository to a single commercial vendor.

## Consequences

### Positive
- Strict epistemic grounding prevents hallucinations and preserves data integrity.
- Deterministic keyframe extraction bounds compute cost and guarantees reproducible evaluation.
- The `VLMProvider` protocol enables zero-downtime switching between local quantized models and high-throughput remote endpoints.
- The core computer vision pipeline remains fully functional, testable, and deployable completely standalone.

### Negative / Trade-offs
- Additional architectural surface area (keyframe extraction utilities, evidence packaging, output schema parsing).
- Prompt engineering and schema validation are required to enforce reliable JSON output formatting across varied models.
- Remote provider operation requires network connectivity and API credential management.
