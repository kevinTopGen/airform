"""Score every measurement adapter against the known-warp ground truth.

    python scripts/bench.py [adapter ...]

Two numbers per (adapter, parameter, tone):

  r  correlation between requested and observed change.  THE PRIMARY METRIC.
     Does the measurement move with reality at all? r near 0 means the technique
     is reporting noise, and no amount of post-hoc correction rescues it.

  k  slope. How much of the true change the technique reports. k=1 is ideal, but
     k=0.4 with r=0.99 is perfectly usable -- a stable gain is calibratable.
     Adapters legitimately differ here because they measure slightly different
     anatomy; that is a scale difference, not an error.

An adapter is USABLE for a parameter when r > 0.95 and k > 0.25. Anything else
must not be used to fit a signature, because fitting on it is fitting on noise.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import measure as M

R_MIN, K_MIN = 0.95, 0.25


def main():
    man = json.load(open("bench/manifest.json"))
    adapters = M.load()
    want = sys.argv[1:]
    if want:
        adapters = {k: v for k, v in adapters.items() if k in want}
    print(f"adapters: {', '.join(adapters) or '(none available)'}\n")

    results, cache = {}, {}

    def get(mod, path):
        key = (mod.NAME, path)
        if key not in cache:
            try:
                cache[key] = mod.measure(path)
            except Exception as e:
                cache[key] = None
                print(f"  ! {mod.NAME} failed on {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return cache[key]

    for name, mod in adapters.items():
        results[name] = {}
        for tname, bpath in man["baselines"].items():
            b = get(mod, bpath)
            if not b:
                continue
            for p in man["params"]:
                if p not in b or not b[p]:
                    continue
                req, obs = [], []
                for im in man["images"]:
                    if im["tone"] != tname or im["param"] != p:
                        continue
                    m = get(mod, im["path"])
                    if m and m.get(p):
                        req.append(im["level"])
                        obs.append(m[p] / b[p] - 1)
                if len(req) >= 3:
                    k = float(np.polyfit(req, obs, 1)[0])
                    r = float(np.corrcoef(req, obs)[0, 1])
                    rms = float(np.sqrt(np.mean((np.array(obs) - np.array(req)) ** 2)))
                    results[name][f"{p}|{tname}"] = {"k": k, "r": r, "rms": rms,
                                                     "n": len(req)}

    hdr = f"{'adapter':<18}{'parameter':<15}{'tone':<11}{'k':>7}{'r':>8}{'rms':>8}   verdict"
    print(hdr)
    print("-" * len(hdr))
    for name in results:
        for key, v in sorted(results[name].items()):
            p, tname = key.split("|")
            ok = v["r"] > R_MIN and v["k"] > K_MIN
            print(f"{name:<18}{p:<15}{tname:<11}{v['k']:>7.3f}{v['r']:>8.3f}"
                  f"{v['rms']*100:>7.1f}%   {'USABLE' if ok else 'unusable'}")
        print()

    with open("bench/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote bench/results.json")


if __name__ == "__main__":
    main()
