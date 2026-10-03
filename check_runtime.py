"""Check installed dependencies and import the model registry without weights."""
from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/mia_net_official")]


def main() -> None:
    required = {
        "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
        "scipy": "scipy", "cv2": "opencv-python", "mmcv": "mmcv",
        "mmdet": "mmdet", "mmcls": "mmcls", "motmetrics": "motmetrics",
        "pandas": "pandas", "skimage": "scikit-image", "yaml": "PyYAML",
        "lap": "lap", "pycocotools": "pycocotools",
    }
    missing = [package for module, package in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit("Missing dependencies: " + ", ".join(missing))
    for package in required.values():
        print(f"{package}: {importlib.metadata.version(package)}")
    from uav_tracking.mmcv_ops_compat import install
    install()
    from mmtrack.models import build_model
    from uav_tracking.identity_topology_model import TopologyEnhancedAssociation
    assert callable(build_model) and callable(TopologyEnhancedAssociation)
    print("Runtime imports passed. Dataset and model weights are external inputs.")


if __name__ == "__main__":
    main()
