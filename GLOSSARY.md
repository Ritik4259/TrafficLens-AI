# TrafficLens-AI

Intelligent traffic understanding and analytics platform leveraging computer vision, multi-object tracking, multimodal vision-language models, and agentic workflows.

## Language

### Core Entities

**Vehicle**:
A motorized road transport entity recognized by the detection model, classified into one of the canonical categories: `car`, `bus`, `van`, or `truck`.
_Avoid_: Target, object, auto, entity

**Detection**:
A single localized spatial prediction in an individual video frame, defined by a bounding box, confidence score, and vehicle category.
_Avoid_: Box, hit, sighting

**Tracklet**:
A unique temporal trajectory connecting detections of the same physical vehicle across consecutive video frames, identified by a persistent track ID.
_Avoid_: Track, path, trail, trajectory segment

**Tripwire**:
A designated virtual directional line segment configured across a roadway to detect and record vehicle crossings in specific directions.
_Avoid_: Virtual line, count line, trigger boundary

**Region of Interest (ROI)**:
A defined polygonal spatial area in a camera view used for measuring traffic metrics such as density, queue length, or dwell time.
_Avoid_: Zone, mask, polygon, sector

**Ignore Region**:
A polygonal zone in an evaluation frame (defined in UA-DETRAC) where vehicle detections are excluded from false positive/false negative scoring during benchmark evaluation.
_Avoid_: Mask zone, excluded area, blind spot

**Traffic Event**:
A classified spatiotemporal irregularity or traffic rule violation (such as stopped vehicle, congestion onset, wrong-way driving, or illegal lane utilization).
_Avoid_: Anomaly, alert, incident, exception

**Traffic Telemetry**:
Aggregated quantitative traffic flow metrics computed over rolling time windows, including flow rate (vehicles/minute), average velocity, and density.
_Avoid_: Stats, analytics data, summary numbers

**Keyframe**:
A single extracted video frame associated with a specific timestamp or traffic event, preserved with overlaid telemetry for audit and visual reasoning.
_Avoid_: Snapshot, thumbnail, screen grab

**Multimodal Agent**:
A stateful reasoning workflow orchestrated via LangGraph that queries traffic telemetry tools and inspects keyframes with a vision-language model to answer natural-language user queries.
_Avoid_: Chatbot, LLM wrapper, assistant

### Multimodal Understanding

**Evidence Package**:
A compact, structured bundle combining selected video keyframes, CV detection bounding boxes, tracking trajectories, and spatiotemporal telemetry windows for VLM consumption.
_Avoid_: Prompt context, input blob, multimodal payload

**Epistemic Grounding**:
The strict attribution of every analytical claim to a verified source category (`cv_fact`, `visual_observation`, `inference`, or `unverified`) with explicit citations to timestamps or track IDs.
_Avoid_: Truth labeling, fact-checking, attribution tagging

**VLM Provider**:
An interchangeable software interface abstracting local quantized transformer models, self-hosted inference servers, and remote multimodal API endpoints behind a unified protocol.
_Avoid_: Model wrapper, LLM backend, inference engine adapter

**Visual Observation**:
A directly perceivable physical attribute in a keyframe (such as wet road pavement, illuminated brake lights, emergency vehicles, or debris) as opposed to a measured numerical quantity.
_Avoid_: Visual fact, seen object, perceptual feature

**Contextual Inference**:
A plausible, model-generated hypothesis explaining a traffic phenomenon (e.g., stationary vehicle due to traffic queue) that is distinct from a deterministic CV measurement.
_Avoid_: Guess, deduction, speculation, prediction

### Evaluation Metrics

**mAP (Mean Average Precision)**:
The primary object detection metric measuring the area under the precision-recall curve across vehicle classes at specified IoU thresholds (mAP@0.5, mAP@0.5:0.95).

**MOTA (Multi-Object Tracking Accuracy)**:
A traditional tracking metric evaluating continuous tracking performance by combining false positives, false negatives, and identity switches.

**IDF1 (Identification F1-Score)**:
A tracking metric measuring how well the tracker maintains the correct identity over time, calculated as the harmonic mean of identification precision and recall.

**HOTA (Higher Order Tracking Accuracy)**:
A balanced tracking evaluation metric that explicitly balances the accuracy of object detection, association, and localization.
