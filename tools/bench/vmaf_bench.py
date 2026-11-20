#!/usr/bin/env python3
"""Scoring-speed benchmark: libvmaf CPU, libvmaf Vulkan (this fork) and libvmaf-fast, 1/2/4 v1.0.16 models, 1080p/4K.

Clips: <clips>/ref_{1080,4k}.yuv and dis_{1080,4k}.yuv, raw 8-bit 4:2:0, 120 frames each (see README).
Every run is timed twice, over N and over N_SHORT frames; steady fps = (N - N_SHORT) / (t_N - t_SHORT), which
removes process/driver start-up. Several runs per case give min / max / mean.

  vmaf_bench.py --engine cpu    --vmaf build/tools/vmaf --models model --clips clips --label "Areldir 9800X3D" --out r.json
  vmaf_bench.py --engine vulkan --vmaf build/tools/vmaf --models model --clips clips --device 0 --label "Areldir RTX 4080 SUPER" ...
  vmaf_bench.py --engine fast   --lvf libvmaf-fast-src --models model --clips clips --label ...   (Vulkan device = its default)

Needs Python 3.8+, no packages (libvmaf-fast mode needs its own python modules, numpy included).
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

V1 = ["1d5h_2160", "3d0h", "3d0h_2160", "5d0h"]
SIZES = {"1080": (1920, 1080), "4k": (3840, 2160)}
N, N_SHORT = 120, 24


def vmaf_cmd(a, res, nmodels, frames):
    w, h = SIZES[res]
    c = [a.vmaf, "-r", f"{a.clips}/ref_{res}.yuv", "-d", f"{a.clips}/dis_{res}.yuv", "-w", str(w), "-h", str(h),
         "-p", "420", "-b", "8", "--frame_cnt", str(frames), "-q"]
    for m in V1[:nmodels]:
        c += ["-m", f"path={a.models}/vmaf_v1.0.16/vmaf_v1.0.16_{m}.json:name={m}"]
    if a.engine == "vulkan":
        c += [f"--vulkan={a.device}"]
    elif a.threads:
        c += ["--threads", str(a.threads)]
    return c


def time_cli(a, res, nm, frames):
    t = time.perf_counter()
    subprocess.run(vmaf_cmd(a, res, nm, frames), check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t


def time_fast(a, res, nm, frames):
    sys.path.insert(0, str(Path(a.lvf) / "fast" / "python"))
    from vmaf_fast import v1 as V1M, vulkan as VK  # noqa
    w, h = SIZES[res]
    fb = w * h * 3 // 2
    t = time.perf_counter()
    sc = [V1M.V1Scorer(w, h, 8, Path(a.models) / "vmaf_v1.0.16" / f"vmaf_v1.0.16_{m}.json", name=m) for m in V1[:nm]]
    try:
        with open(f"{a.clips}/ref_{res}.yuv", "rb", buffering=0) as fr, open(f"{a.clips}/dis_{res}.yuv", "rb", buffering=0) as fd:
            for _ in range(frames):
                r, d = bytearray(fr.read(fb)), bytearray(fd.read(fb))
                for s in sc:
                    s.add(r, d)
        for s in sc:
            s.finish()
    finally:
        for s in sc:
            s.close()
    return time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["cpu", "vulkan", "fast"], required=True)
    p.add_argument("--vmaf"); p.add_argument("--lvf"); p.add_argument("--models", required=True)
    p.add_argument("--clips", required=True); p.add_argument("--device", default="0")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = vmaf default)")
    p.add_argument("--runs", type=int, default=7); p.add_argument("--label", required=True)
    p.add_argument("--res", default="1080,4k"); p.add_argument("--models-n", default="1,2,4")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    timer = time_fast if a.engine == "fast" else time_cli
    results = []
    for res in a.res.split(","):
        for nm in map(int, a.models_n.split(",")):
            timer(a, res, nm, N_SHORT)  # warm-up (disk cache, driver caches)
            runs = []
            for _ in range(a.runs):
                ts, tl = timer(a, res, nm, N_SHORT), timer(a, res, nm, N)
                runs.append({"t_short": ts, "t_full": tl, "fps": (N - N_SHORT) / (tl - ts), "fps_with_startup": N / tl})
            f = [r["fps"] for r in runs]
            row = {"res": res, "models": nm, "min": min(f), "max": max(f), "avg": sum(f) / len(f), "runs": runs}
            results.append(row)
            print(f"{a.label} {a.engine} {res} {nm} models: min {row['min']:.1f} avg {row['avg']:.1f} max {row['max']:.1f} fps", flush=True)
    meta = {"label": a.label, "engine": a.engine, "frames": N, "short_frames": N_SHORT, "runs": a.runs,
            "platform": platform.platform(), "cpu": platform.processor(), "device": a.device, "date": time.strftime("%Y-%m-%d %H:%M")}
    Path(a.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))


if __name__ == "__main__":
    main()
