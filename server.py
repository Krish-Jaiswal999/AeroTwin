import os
import sys
import subprocess

# If running outside the virtual environment, automatically switch to .venv python
_venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'python.exe'))
if os.path.exists(_venv_py) and os.path.normcase(sys.executable) != os.path.normcase(_venv_py):
    ret = subprocess.call([_venv_py] + sys.argv)
    sys.exit(ret)

import json
import secrets
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='.')

# ── Config ────────────────────────────────────────────────
JOBS_DIR   = Path('jobs')
OUTPUT_BASE = Path('data/output')   # kept for legacy static serving
JOBS_DIR.mkdir(exist_ok=True)

# Use .venv python for subprocesses
_venv_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'python.exe'))
PYTHON_EXE = _venv_py_path if os.path.exists(_venv_py_path) else sys.executable

ALLOWED_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.MP4', '.MOV', '.AVI'}
MAX_FILE_SIZE_MB = 4000  # 4 GB

# ── Helpers ──────────────────────────────────────────────

def make_job_id() -> str:
    return secrets.token_hex(3)   # e.g. "8f32a1"

def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def job_status_file(job_id: str) -> Path:
    return job_dir(job_id) / 'status.json'

def write_status(job_id: str, stage: str, progress: int, message: str,
                 is_complete: bool = False, error: bool = False):
    status = {
        'job_id':      job_id,
        'stage':       stage,
        'progress':    progress,
        'message':     message,
        'is_complete': is_complete,
        'error':       error,
        'timestamp':   time.time(),
    }
    jf = job_status_file(job_id)
    jf.parent.mkdir(parents=True, exist_ok=True)
    with open(jf, 'w') as f:
        json.dump(status, f)

def load_json(path: Path) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def run_script(script: str, extra_args: list, job_id: str) -> bool:
    """Run a python script as a subprocess. Returns True on success."""
    cmd = [PYTHON_EXE, script] + extra_args
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.abspath('.'),
            timeout=7200,   # 2 hours max per stage
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        write_status(job_id, 'Error', 0, f'{script} timed out', error=True)
        return False
    except Exception as e:
        write_status(job_id, 'Error', 0, str(e), error=True)
        return False

# ── Pipeline ─────────────────────────────────────────────

def run_pipeline(job_id: str, video_path: str):
    """Full reconstruction pipeline — runs in a background thread."""
    jd = job_dir(job_id)
    project_name = job_id   # each job IS its own project

    timings = {}

    try:
        # ── Stage 1: Frame extraction ──
        write_status(job_id, 'Extract', 8, 'Extracting frames from video...')
        t0 = time.time()
        ok = run_script('scripts/01_extract_frames.py', [
            '--input', video_path,
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['frame_extraction'] = round(time.time() - t0, 1)
        if not ok:
            write_status(job_id, 'Error', 0,
                         'Frame extraction failed. Check video format.', error=True)
            return

        # ── Stage 2: Keyframe selection ──
        write_status(job_id, 'Keyframes', 20, 'Selecting optimal keyframes...')
        t0 = time.time()
        run_script('scripts/10_keyframe_analysis.py', [
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['keyframe_selection'] = round(time.time() - t0, 1)
        # Non-fatal — continue even if keyframe selection fails

        # ── Stage 3: COLMAP SfM ──
        write_status(job_id, 'SfM', 35, 'Running COLMAP feature extraction & mapping...')
        t0 = time.time()
        ok = run_script('scripts/02_run_colmap.py', [
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['colmap'] = round(time.time() - t0, 1)
        if not ok:
            write_status(job_id, 'Error', 0,
                         'COLMAP reconstruction failed. Insufficient overlap or features.',
                         error=True)
            return

        # ── Stage 3.5: Georeferencing & Flight Metric ──
        write_status(job_id, 'Georef', 60, 'Computing metric scale & flight trajectory...')
        t0 = time.time()
        run_script('scripts/09_georeferencing.py', [
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['georef'] = round(time.time() - t0, 1)

        # ── Stage 4: Mesh + analytics ──
        write_status(job_id, 'Mesh', 70, 'Building surface mesh...')
        t0 = time.time()
        run_script('scripts/06_mesh_and_analytics.py', [
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['mesh'] = round(time.time() - t0, 1)
        # Non-fatal

        # ── Stage 5: Semantic + dynamic ──
        write_status(job_id, 'Semantic', 82, 'Running semantic segmentation (mask generation)...')
        t0 = time.time()
        run_script('scripts/08_semantic_filter.py', [
            '--project', project_name,
            '--output-base', str(jd),
        ], job_id)
        timings['semantic'] = round(time.time() - t0, 1)
        # Non-fatal

        # ── Stage 5.5: 2D→3D Semantic Projection ──
        write_status(job_id, 'Semantic3D', 88,
                     'Projecting semantic labels into 3D point cloud (multi-view voting)...')
        t0 = time.time()
        run_script('scripts/semantic_2d_to_3d.py', [
            '--job-id', job_id,
            '--jobs-base', str(JOBS_DIR),
        ], job_id)
        timings['semantic_3d'] = round(time.time() - t0, 1)
        # Non-fatal

        # ── Stage 6: 3D Scene Builder ──
        write_status(job_id, 'Scene', 93, 'Building web-ready 3D digital twin scene...')
        t0 = time.time()
        run_script('scripts/11_scene_builder.py', [
            '--job-id', job_id,
            '--jobs-base', str(JOBS_DIR),
            '--max-pts', '80000',
        ], job_id)
        timings['scene_builder'] = round(time.time() - t0, 1)
        # Non-fatal

        # ── Write timing summary ──
        timing_path = jd / project_name / 'pipeline_timings.json'
        with open(timing_path, 'w') as f:
            json.dump(timings, f, indent=2)

        write_status(job_id, 'Done', 100, 'Pipeline complete!', is_complete=True)

    except Exception as e:
        write_status(job_id, 'Error', 0, f'Unexpected error: {str(e)}', error=True)


# ── Static serving ────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/results.html')
def results():
    return send_file('results.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


# ── API: Upload ───────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix
    if ext.lower() not in {e.lower() for e in ALLOWED_EXTS}:
        return jsonify({'error': f'Unsupported file type: {ext}. Use MP4, MOV, AVI, MKV.'}), 400

    # Create job
    job_id = make_job_id()
    jd = job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)

    # Save video
    input_path = jd / f'input{ext}'
    file.save(str(input_path))

    # Write initial metadata
    meta = {
        'job_id': job_id,
        'original_filename': original_name,
        'video_path': str(input_path),
        'upload_time': time.time(),
    }
    with open(jd / 'job_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    write_status(job_id, 'Received', 3, f'Video received: {original_name}')

    # Start pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, str(input_path)),
        daemon=True,
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Upload successful. Pipeline started.',
        'original_filename': original_name,
        'status_url': f'/api/jobs/{job_id}/status',
        'results_url': f'/results.html?job={job_id}',
    })


# ── API: Job status ───────────────────────────────────────

@app.route('/api/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    sf = job_status_file(job_id)
    if not sf.exists():
        return jsonify({'error': 'Job not found', 'job_id': job_id}), 404
    return jsonify(load_json(sf))


# ── API: 3D Scene Package ──────────────────────────────────

@app.route('/api/jobs/<job_id>/scene', methods=['GET'])
def get_job_scene(job_id):
    jd = job_dir(job_id)
    if not jd.exists():
        return jsonify({'error': 'Job not found', 'job_id': job_id}), 404
    scene_file = jd / 'scene' / 'scene.json'
    if not scene_file.exists():
        scene_file = jd / job_id / 'scene' / 'scene.json'
    if not scene_file.exists():
        return jsonify({'error': 'Scene not yet generated', 'job_id': job_id}), 404
    return jsonify(load_json(scene_file))


# ── API: Legacy /api/status (for existing index.html polling) ──

@app.route('/api/status', methods=['GET'])
def get_legacy_status():
    """Compatibility endpoint — returns status of most recently created job."""
    try:
        # Find the newest job
        jobs = sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not jobs:
            return jsonify({'stage': 'Idle', 'progress': 0, 'message': 'No jobs yet'})
        newest_job = jobs[0].name
        sf = job_status_file(newest_job)
        if sf.exists():
            data = load_json(sf)
            return jsonify(data)
    except Exception:
        pass
    return jsonify({'stage': 'Idle', 'progress': 0, 'message': 'Waiting for upload'})


# ── API: Job results (all real output data) ───────────────

@app.route('/api/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    jd = job_dir(job_id)
    if not jd.exists():
        return jsonify({'error': 'Job not found'}), 404

    project_dir = jd / job_id  # scripts write to jobs/<id>/<id>/

    meta            = load_json(jd / 'job_meta.json')
    frame_report    = load_json(project_dir / 'frame_report.json')
    keyframe_report = load_json(project_dir / 'keyframe_selection_report.json')
    colmap_report   = load_json(project_dir / 'colmap_report.json')
    semantic_report = load_json(project_dir / 'semantic_classification_report.json')
    dynamic_report  = load_json(project_dir / 'dynamic_object_report.json')
    georef_report   = load_json(project_dir / 'georef_report.json')
    mesh_report     = load_json(project_dir / 'mesh_report.json')
    timings         = load_json(project_dir / 'pipeline_timings.json')
    status          = load_json(jd / 'status.json')

    # Frame previews (up to 12)
    frames_dir = project_dir / 'frames'
    frame_previews = []
    if frames_dir.exists():
        jpgs = sorted(frames_dir.glob('*.jpg'))[:12]
        frame_previews = [f'/jobs/{job_id}/{job_id}/frames/{f.name}' for f in jpgs]

    # Keyframe images (up to 20)
    keyframes_dir = project_dir / 'keyframes'
    keyframe_images = []
    if keyframes_dir.exists():
        kfjpgs = sorted(keyframes_dir.glob('*.jpg'))[:20]
        keyframe_images = [f'/jobs/{job_id}/{job_id}/keyframes/{f.name}' for f in kfjpgs]

    # Semantic images (up to 6)
    semantics_dir = project_dir / 'semantics'
    semantic_images = []
    if semantics_dir.exists():
        semjpgs = sorted(semantics_dir.glob('sem_*.jpg'))[:6]
        semantic_images = [f'/jobs/{job_id}/{job_id}/semantics/{f.name}' for f in semjpgs]

    # PLY for 3D viewer
    ply_url = None
    for ply_name in ['sparse_pointcloud.ply', 'display_cloud.ply', 'aerotwin_mesh.ply']:
        if (project_dir / ply_name).exists():
            ply_url = f'/jobs/{job_id}/{job_id}/{ply_name}'
            break

    # Camera poses from images.txt (for 3D viewer flight path)
    camera_positions = []
    for model_idx in ['1', '0']:
        images_txt = project_dir / 'sparse' / model_idx / 'images.txt'
        if images_txt.exists():
            camera_positions = _parse_camera_positions(images_txt)
            break

    # Colmap sparse stats
    sparse = colmap_report.get('sparse', {})
    num_registered = sparse.get('num_images_registered', 0)
    num_frames = frame_report.get('frames_kept', frame_report.get('total_video_frames', 0))
    # Registration rate: cap at 100% (COLMAP can register more than input if multiple sub-models merged)
    if num_registered > 0 and num_frames > 0:
        reg_rate = round(min(num_registered / num_frames * 100, 100.0), 1)
    else:
        reg_rate = 0.0

    # Georeferencing
    gps_available = bool(
        georef_report.get('gps_available')
        or georef_report.get('gps_status') == 'available'
    )
    scale = georef_report.get('scale_m_per_unit', None)

    result = {
        'job_id':           job_id,
        'status':           status.get('stage', 'Unknown'),
        'is_complete':      status.get('is_complete', False),

        # Video info
        'original_filename': meta.get('original_filename', ''),
        'video_resolution':  frame_report.get('video_resolution', ''),
        'video_fps':         frame_report.get('video_fps', 0),
        'video_duration_sec': frame_report.get('video_duration_sec', 0),
        'total_video_frames': frame_report.get('total_video_frames', 0),

        # Frame extraction
        'frames_kept':       frame_report.get('frames_kept', 0),
        'frames_blurry':     frame_report.get('frames_blurry', 0),

        # Keyframe selection
        'keyframes_selected': keyframe_report.get('selected_keyframes', 0),
        'keyframes_rejected_blurry':     keyframe_report.get('rejected_blurry', 0),
        'keyframes_rejected_redundant':  keyframe_report.get('rejected_redundant', 0),
        'keyframes_rejected_low_info':   keyframe_report.get('rejected_low_information', 0),

        # Reconstruction
        'num_registered_cameras': num_registered,
        'num_input_frames':       colmap_report.get('num_input_frames', num_frames),
        'registration_rate_pct':  reg_rate,
        'num_sparse_points':      sparse.get('num_points3D', 0),
        'reconstruction_success': colmap_report.get('success', False),

        # Mesh
        'mesh_generated':    mesh_report.get('mesh_generated', False),
        'mesh_faces':        mesh_report.get('mesh_faces', 0),

        # Semantic
        'semantic_class_averages': semantic_report.get('class_averages', {}),
        'semantic_frames_analyzed': semantic_report.get('frames_analyzed', 0),

        # Dynamic
        'dynamic_high_motion_count': dynamic_report.get('high_motion_count', 0),
        'dynamic_total_frames':      dynamic_report.get('total_frames', 0),

        # Georeferencing
        'gps_available':     gps_available,
        'metric_scale':      scale,
        'trajectory_length_m': georef_report.get('trajectory_length_m'),

        # Timings
        'pipeline_timings':  timings,
        'colmap_timings':    colmap_report.get('timings', {}),

        # 3D viewer
        'ply_url':           ply_url,
        'camera_positions':  camera_positions[:200],   # limit to 200 poses for JSON size

        # Media paths
        'frame_previews':    frame_previews,
        'keyframe_images':   keyframe_images,
        'semantic_images':   semantic_images,
    }
    return jsonify(result)


def _parse_camera_positions(images_txt: Path) -> list:
    """Parse COLMAP images.txt to extract camera world positions.
    
    COLMAP stores camera-to-world transform as quaternion (qw,qx,qy,qz)
    and translation t. The camera world position is: C = -R^T @ t
    where R is the rotation matrix from quaternion.
    """
    import math
    positions = []
    try:
        with open(images_txt, encoding='utf-8', errors='replace') as f:
            lines = [l.rstrip() for l in f if not l.startswith('#') and l.strip()]
        # Every even-indexed line (0, 2, 4...) is image data:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 9:
                continue
            try:
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
                
                # Build rotation matrix from quaternion
                # R rotates from world to camera
                r00 = 1 - 2*(qy*qy + qz*qz)
                r01 = 2*(qx*qy - qz*qw)
                r02 = 2*(qx*qz + qy*qw)
                r10 = 2*(qx*qy + qz*qw)
                r11 = 1 - 2*(qx*qx + qz*qz)
                r12 = 2*(qy*qz - qx*qw)
                r20 = 2*(qx*qz - qy*qw)
                r21 = 2*(qy*qz + qx*qw)
                r22 = 1 - 2*(qx*qx + qy*qy)
                
                # Camera world position = -R^T * t
                cx = -(r00*tx + r10*ty + r20*tz)
                cy = -(r01*tx + r11*ty + r21*tz)
                cz = -(r02*tx + r12*ty + r22*tz)
                
                positions.append({'x': cx, 'y': cy, 'z': cz})
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    return positions


# ── API: Serve job files ──────────────────────────────────

@app.route('/jobs/<job_id>/<path:filepath>')
def serve_job_file(job_id, filepath):
    """Serve files from a specific job directory."""
    jd = job_dir(job_id)
    if not jd.exists():
        return jsonify({'error': 'Job not found'}), 404
    return send_from_directory(str(jd), filepath)


# ── API: List all jobs ────────────────────────────────────

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    jobs = []
    if JOBS_DIR.exists():
        for jd in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if jd.is_dir():
                sf = jd / 'status.json'
                meta_f = jd / 'job_meta.json'
                status = load_json(sf) if sf.exists() else {}
                meta = load_json(meta_f) if meta_f.exists() else {}
                jobs.append({
                    'job_id': jd.name,
                    'stage': status.get('stage', 'Unknown'),
                    'is_complete': status.get('is_complete', False),
                    'original_filename': meta.get('original_filename', ''),
                    'upload_time': meta.get('upload_time', 0),
                })
    return jsonify({'jobs': jobs})


# ── Legacy API: old /api/results route (compatibility) ────

@app.route('/api/results', methods=['GET'])
def get_legacy_results():
    """Returns results of the most recently completed job."""
    try:
        if JOBS_DIR.exists():
            completed = [
                jd for jd in JOBS_DIR.iterdir()
                if jd.is_dir() and load_json(jd / 'status.json').get('is_complete')
            ]
            if completed:
                newest = max(completed, key=lambda p: p.stat().st_mtime)
                return get_job_results(newest.name)
    except Exception:
        pass
    return jsonify({'error': 'No completed jobs yet.'}), 404


if __name__ == '__main__':
    print("\n  +==========================================+")
    print("  |  AeroTwin - Production Pipeline Server   |")
    print("  |  http://127.0.0.1:5000                   |")
    print("  +==========================================+\n")
    app.run(debug=False, port=5000, threaded=True)
