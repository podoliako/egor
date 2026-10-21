"""Reset-only scikit-fmm stencil controls; no production or environment edits.

Run from the repository root:
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -B \
        projects/EMTomo/studies/station_seed/stencil_controls.py

Pinned upstream C++ sources are downloaded into memory. Compilation uses -pipe
and anonymous Linux memfd objects/shared libraries, not disk build directories.
Only the requested stencil_controls*.json result is written. No tolerance,
heap, seeding, speed rounding, or PyKonal changes are included.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import skfmm


TAG = "2025.06.23"
ROOT = f"https://raw.githubusercontent.com/scikit-fmm/scikit-fmm/{TAG}/skfmm/"
NAMES = (
    "base_marcher.h", "heap.h", "distance_marcher.h", "travel_time_marcher.h",
    "heap.cpp", "base_marcher.cpp", "distance_marcher.cpp", "travel_time_marcher.cpp",
)
VARIANTS = {
    "original": (),
    "travel_reset_only": ("travel_time_marcher.cpp",),
    "distance_reset_only": ("distance_marcher.cpp",),
    "both_reset_only": ("travel_time_marcher.cpp", "distance_marcher.cpp"),
}
WRAPPER = r'''
class TravelProbe : public travelTimeMarcher {
public:
  using travelTimeMarcher::travelTimeMarcher;
  double evaluate(int i) { return updatePointOrderTwo(i); }
};
class DistanceProbe : public distanceMarcher {
public:
  using distanceMarcher::distanceMarcher;
  double evaluate(int i) { return updatePointOrderTwo(i); }
};
extern "C" int run_field(double *p, double *v, double *t, int dim,
                         int *shape, double h, int is_distance) {
  double dx[12]; int n=1;
  for (int d=0;d<dim;d++) { dx[d]=h; n*=shape[d]; }
  std::vector<long long> flags(n,0);
  for (int i=0;i<n;i++) t[i]=0.;
  try {
    if (is_distance) {
      distanceMarcher m(p,dx,flags.data(),t,dim,shape,false,2,0.,0);
      m.march(); return m.getError();
    }
    travelTimeMarcher m(p,dx,flags.data(),t,dim,shape,false,2,v,0.,0);
    m.march(); return m.getError();
  } catch (...) { return 9; }
}
extern "C" double probe_stencil(double *p, double *v, double *t,
                                 long long *flags, int n, double h,
                                 int target, int is_distance) {
  double dx[1]={h}; int shape[1]={n};
  if (is_distance) {
    DistanceProbe m(p,dx,flags,t,1,shape,false,2,0.,0);
    return m.evaluate(target);
  }
  TravelProbe m(p,dx,flags,t,1,shape,false,2,v,0.,0);
  return m.evaluate(target);
}
'''


def fetch_source(name):
    with urllib.request.urlopen(ROOT + name, timeout=20) as response:
        return name, response.read().decode()


class MemoryMarcher:
    def __init__(self, sources, reset_files, compiler):
        parts = ["#include <cmath>\n#include <algorithm>"]
        for name in NAMES:
            src = sources[name]
            if name in reset_files:
                old = "value1 = distance_[naddr];"
                assert src.count(old) == 1, (name, "unexpected upstream source")
                src = src.replace(old, old + "\n          value2 = maxDouble;")
            # Separate translation units use these same global constant names.
            if name == "distance_marcher.cpp":
                src = re.sub(r"\baa\b", "distance_aa", src)
                src = re.sub(r"\boneThird\b", "distance_oneThird", src)
            src = re.sub(r'^#include "[^"]+".*$', "", src, flags=re.M)
            parts.append(src.replace("#pragma once", ""))
        code = "\n".join(parts) + WRAPPER
        self.source_sha256 = hashlib.sha256(code.encode()).hexdigest()
        self.fds = [os.memfd_create("stencil_object"), os.memfd_create("stencil_library")]
        obj, so = (f"/proc/self/fd/{fd}" for fd in self.fds)
        compile_args = [compiler, "-pipe", "-O3", "-fPIC", "-std=c++11",
                        "-x", "c++", "-c", "-", "-o", obj]
        link_args = [compiler, "-shared", obj, "-Wl,--no-as-needed", "-lstdc++",
                     "-lm", "-o", so]
        for args, text in ((compile_args, code), (link_args, None)):
            result = subprocess.run(args, input=text, text=True, capture_output=True,
                                    pass_fds=tuple(self.fds), timeout=20)
            if result.returncode:
                raise RuntimeError(result.stderr)
        self.lib = ctypes.CDLL(so)
        self.ptr = ctypes.POINTER(ctypes.c_double)
        self.field = self.lib.run_field
        self.field.argtypes = [self.ptr, self.ptr, self.ptr, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_int]
        self.field.restype = ctypes.c_int
        self.probe = self.lib.probe_stencil
        self.probe.argtypes = [self.ptr, self.ptr, self.ptr,
                               ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
                               ctypes.c_double, ctypes.c_int, ctypes.c_int]
        self.probe.restype = ctypes.c_double

    def solve(self, phi, velocity, h, distance=False):
        phi = np.ascontiguousarray(phi, dtype=np.float64)
        velocity = np.ascontiguousarray(velocity, dtype=np.float64)
        out = np.empty_like(phi)
        shape = np.array(phi.shape, dtype=np.int32)
        error = self.field(phi.ctypes.data_as(self.ptr), velocity.ctypes.data_as(self.ptr),
                           out.ctypes.data_as(self.ptr), phi.ndim,
                           shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), h, distance)
        if error:
            raise RuntimeError(f"marcher error {error}")
        return out

    def local_probe(self, distance=False, missing_second=False, reflect=False):
        # Target 2: left pair .2/.1; nearer right .15 has no admissible second.
        t = np.array([.1, .2, 0., .15, .3], dtype=np.float64)
        flags = np.array([2, 2, 0, 2, 0 if missing_second else 2], dtype=np.int64)
        if missing_second:
            t, flags = t[:4].copy(), flags[:4].copy()
        if reflect:
            t, flags = t[::-1].copy(), flags[::-1].copy()
        target = len(t) - 3 if reflect else 2
        if distance:
            t *= 1000.
        phi = np.ones_like(t)
        velocity = np.full_like(t, 1000.)
        return self.probe(phi.ctypes.data_as(self.ptr), velocity.ctypes.data_as(self.ptr),
                          t.ctypes.data_as(self.ptr),
                          flags.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                          len(t), 100., target, distance)


def stats(delta):
    delta = np.asarray(delta, dtype=np.float64)
    index = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    return {"max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta ** 2))),
            "mean": float(np.mean(delta)),
            "argmax": [int(i) for i in index]}


def speed_steps():
    v = np.float64(4000.)
    bits = v.view(np.int64)
    for n in (-1024, -64, -8, -2, -1, 1, 2, 8, 64, 1024):
        changed = np.int64(bits + n).view(np.float64)
        yield f"{n:+d}_ulp", float(changed), {"actual_ulp": n}
    for sign in (-1, 1):
        changed = v * (1. + sign * np.finfo(np.float64).eps)
        yield f"{sign:+d}_relative_eps", float(changed), {
            "actual_ulp": int(changed.view(np.int64) - bits)}


def geometries():
    h, shape = 125., (48, 32, 32)
    xyz = (np.moveaxis(np.indices(shape), 0, -1) + .5) * h
    station = np.array([1000., 875., 0.])
    r = np.linalg.norm(xyz - station, axis=-1)
    phi = r - 3 * h
    block = np.zeros(shape, dtype=bool)
    block[4:8, 4:8, :4] = True
    box = np.ones((24, 24))
    box[7:9, 7:9] = 0.
    block2 = np.zeros(box.shape, dtype=bool)
    block2[4:8, 4:8] = True
    return h, xyz, r, [
        ("sphere3d_patched", phi, block, phi <= 0),
        ("zero_box2d_no_sphere", box, block2, np.zeros(box.shape, dtype=bool)),
    ]


def run(output):
    started = time.perf_counter()
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("This disk-free build requires Linux memfd_create")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required")
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = dict(pool.map(fetch_source, NAMES))
    marchers = {name: MemoryMarcher(sources, reset, compiler)
                for name, reset in VARIANTS.items()}
    result = {
        "upstream_tag": TAG, "installed_skfmm": skfmm.__version__,
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "upstream_sha256": {n: hashlib.sha256(s.encode()).hexdigest()
                            for n, s in sources.items()},
        "compiled_sha256": {n: m.source_sha256 for n, m in marchers.items()},
        "reset_files": VARIANTS,
        "constraints": {"heap_unchanged": True, "no_tolerances": True,
                        "no_speed_rounding": True, "order": 2,
                        "build_storage": "anonymous memfd only"},
        "local_stale_probe": {}, "continuity": {}, "analytic_accuracy": {},
    }
    for name, marcher in marchers.items():
        probes = {}
        for distance in (False, True):
            for missing in (False, True):
                label = ("distance" if distance else "travel_time") + (
                    "_second_out_of_bounds" if missing else "_second_inadmissible")
                forward = marcher.local_probe(distance, missing, False)
                reflected = marcher.local_probe(distance, missing, True)
                probes[label] = {"forward": forward, "reflected": reflected,
                                 "expected": 250. if distance else .25}
        result["local_stale_probe"][name] = probes
    h, xyz, r, cases = geometries()
    for case, phi, block, inside in cases:
        v = np.full(phi.shape, 4000.)
        baselines = {name: m.solve(phi, v, h) for name, m in marchers.items()}
        installed = np.asarray(skfmm.travel_time(phi, v, dx=h, order=2))
        np.testing.assert_array_equal(baselines["original"], installed)
        rows = {name: [] for name in marchers}
        for label, speed, meta in speed_steps():
            perturbed = v.copy()
            perturbed[block] = speed
            perturbed[inside] = 4000.
            original_installed = np.asarray(skfmm.travel_time(phi, perturbed, dx=h, order=2))
            for name, marcher in marchers.items():
                field = marcher.solve(phi, perturbed, h)
                if name == "original":
                    np.testing.assert_array_equal(field, original_installed)
                rows[name].append({
                    "step": label, **meta, "speed_delta_m_s": speed - 4000.,
                    **stats(field - baselines[name]),
                    "bitwise_matches_original_perturbed": bool(
                        np.array_equal(field, original_installed)),
                    "max_abs_vs_original_perturbed": float(
                        np.max(np.abs(field - original_installed))),
                })
        result["continuity"][case] = {
            "original_bitwise_matches_installed_all_steps": True,
            "baseline_vs_original": {n: stats(b - baselines["original"])
                                     for n, b in baselines.items()},
            "steps": rows,
        }
        print(case, flush=True)
        for name in marchers:
            selected = {row["step"]: row["max_abs"] * 1000 for row in rows[name]
                        if abs(row["actual_ulp"]) in (1, 8)}
            print(" ", name, "max_delta_ms", selected, flush=True)
    phi = r - 3 * h
    v = np.full(r.shape, 4000.)
    plane = xyz[..., 2] - 3 * h
    for name, marcher in marchers.items():
        raw = marcher.solve(phi, v, h)
        field = raw + 3 * h / 4000.
        field[phi <= 0] = r[phi <= 0] / 4000.
        dist = marcher.solve(phi, v, h, distance=True)
        planar = marcher.solve(plane, v, h)
        result["analytic_accuracy"][name] = {
            "sphere_travel_time_error_s": stats(field - r / 4000.),
            "sphere_travel_time_exterior_error_s": stats((field - r / 4000.)[phi > 0]),
            "sphere_signed_distance_error_m": stats(dist - phi),
            "plane_travel_time_error_s": stats(planar - np.abs(plane) / 4000.),
            "travel_sphere_vs_original_s": stats(raw - marchers["original"].solve(phi, v, h)),
            "signed_distance_vs_original_m": stats(
                dist - marchers["original"].solve(phi, v, h, distance=True)),
        }
        if name == "original":
            np.testing.assert_array_equal(dist, np.asarray(skfmm.distance(phi, dx=h, order=2)))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "completed; continuity is measured, not asserted to pass"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved", output, "seconds", round(result["elapsed_seconds"], 3), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("stencil_controls_results.json"))
    args = parser.parse_args()
    if (args.output.resolve().parent != Path(__file__).resolve().parent
            or not args.output.name.startswith("stencil_controls")
            or args.output.suffix != ".json"):
        parser.error("output must be a stencil_controls*.json alongside this script")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result; select a new --output name")
    run(args.output)
