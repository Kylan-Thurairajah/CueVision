
# pool_ai_full (Jetson-ready, no-ML MVP)
Local pool-table analyzer: felt mask → top-down warp → Hough ball/cue detection → ghost-ball planner → advice + overlay.
CUDA/TensorRT hooks are ready but optional.

## Install
```bash
python3 -m pip install opencv-python fastapi uvicorn
```

## CLI
```bash
python3 -m pool_ai_full.cli /path/to/pool.jpg --overlay /tmp/overlay.png
# Felt color:
python3 -m pool_ai_full.cli /path/to/pool.jpg --felt auto
python3 -m pool_ai_full.cli /path/to/pool.jpg --felt blue
python3 -m pool_ai_full.cli /path/to/pool.jpg --felt green
```

## API
```bash
python3 -m pool_ai_full.server
curl -F "file=@/path/to/pool.jpg" -F "felt=auto" http://localhost:8080/analyze > result.json
```

## Upgrade path (CUDA/TensorRT)
- Replace cv2 ops with cv.cuda equivalents to keep data on GPU.
- Implement TRTBallsDetector to run a tiny learned detector (FP16/INT8).
- Port the planner's inner loop to a CUDA kernel for parallel shot sampling.
