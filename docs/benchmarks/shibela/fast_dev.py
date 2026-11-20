# Local wrapper: forces libvmaf-fast's Vulkan device (best_device() otherwise always picks the discrete GPU).
import os, sys, runpy
from pathlib import Path
lvf = sys.argv[sys.argv.index("--lvf") + 1]
sys.path.insert(0, str(Path(lvf) / "fast" / "python"))
from vmaf_fast import vulkan as VK
dev = int(os.environ["FAST_DEV"])
found = VK.devices()
print("fast devices:", [(d.index, d.name) for d in found], "-> forcing", dev, flush=True)
chosen = next(d for d in found if d.index == dev)
VK.best_device = lambda found=None: chosen
runpy.run_path("vmaf/tools/bench/vmaf_bench.py", run_name="__main__")
