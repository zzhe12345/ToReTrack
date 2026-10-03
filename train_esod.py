"""Launch the retained ESOD trainer with this project's runtime adapters."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from uav_tracking.mmcv_ops_compat import install


if __name__ == "__main__":
    install()
    esod_root = ROOT / "third_party/esod"
    sys.path.insert(0, str(esod_root))
    os.chdir(esod_root)
    sys.argv[0] = str(esod_root / "train.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
