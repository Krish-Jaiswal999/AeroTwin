# AeroTwin

> **One Flight. One Video. One Measurable 3D Digital Twin.**

Smart India Hackathon 2026 — SIH26158  
Single-Pass Drone Video to Accurate 3D Model Generation System

## Quick Start

### Prerequisites
- Python 3.13 (virtual environment included)
- COLMAP (in `tools/` or system PATH)
- FFmpeg (optional, OpenCV handles most video formats)

### Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (already done if you cloned fresh)
pip install -r requirements.txt
```

### Run Pipeline

#### Step 1: Extract frames from drone video
```bash
python scripts/01_extract_frames.py --input data/input/your_drone_video.mp4 --project test1
```

#### Step 2: Run COLMAP 3D reconstruction
```bash
python scripts/02_run_colmap.py --project test1
```

#### Step 3: View the 3D point cloud
```bash
python scripts/03_view_pointcloud.py --project test1
```

## Project Structure
```
AeroTwin/
├── config/
│   └── colmap_defaults.yaml    # COLMAP & extraction parameters
├── data/
│   ├── input/                  # Place drone videos here
│   └── output/                 # Reconstruction results
├── scripts/
│   ├── 01_extract_frames.py    # Video → frames
│   ├── 02_run_colmap.py        # Frames → 3D reconstruction
│   └── 03_view_pointcloud.py   # Visualize results
├── tools/                      # COLMAP, FFmpeg binaries
├── requirements.txt
└── README.md
```

## Team
- Member 1: 3D / Reconstruction (COLMAP, Open3D, georeferencing)
- Member 2: AI / Computer Vision (keyframe selection, segmentation, semantic filtering)
- Member 3: Frontend / 3D UI (React, Three.js, measurements)
- Member 4: Backend / Integration (FastAPI, pipeline, benchmarking)

## Status
- [x] Milestone 1: Core pipeline (video → frames → COLMAP → 3D)
- [ ] Milestone 2: Intelligent keyframe selection
- [ ] Milestone 3: Better reconstruction
- [ ] Milestone 4: GPS / georeferencing
- [ ] Milestone 5: Metric measurement
- [ ] Milestone 6: Semantic detection
- [ ] Milestone 7: Dynamic object filtering
- [ ] Milestone 8: Web viewer
- [ ] Milestone 9: Accuracy / benchmarking
- [ ] Milestone 10: Final integration / demo
