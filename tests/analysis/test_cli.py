import os
from pathlib import Path
import subprocess
import sys

import cv2


ROOT = Path(__file__).resolve().parents[2]


def test_cli_writes_overlay_with_redirected_cp1252_output(tmp_path):
    overlay = tmp_path / "overlay.png"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pool_ai_full.cli",
            str(ROOT / "sample_pool_table1.jpg"),
            "--overlay",
            str(overlay),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr
    assert b"Pot ball #" in process.stdout
    image = cv2.imread(str(overlay))
    assert image is not None
    assert image.shape == (512, 1024, 3)
