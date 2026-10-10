"""Report the hardware and software ADAPT will actually run on. Run before sizing any model."""

from __future__ import annotations

import platform
import shutil
import sys

from adapt.config import DATA_DIR


def main() -> None:
    print(f"python     {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform   {platform.platform()}")

    try:
        import torch

        print(f"torch      {torch.__version__}  cuda build: {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"gpu        {p.name}  {p.total_memory / 2**30:.1f} GiB  cc {p.major}.{p.minor}")
            print(f"bf16       {torch.cuda.is_bf16_supported()}")
        else:
            print("gpu        not available to torch (CPU-only build or driver issue)")
    except ImportError:
        print("torch      not installed")

    anchor = DATA_DIR if DATA_DIR.exists() else DATA_DIR.anchor
    usage = shutil.disk_usage(anchor)
    print(f"data dir   {DATA_DIR}  (exists: {DATA_DIR.exists()}, free {usage.free / 2**30:.0f} GiB)")


if __name__ == "__main__":
    main()
