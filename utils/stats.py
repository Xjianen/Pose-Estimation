"""Statistics: stage timing and pose-accuracy accumulation / reporting."""

import time

import numpy as np


class StageTimer:
    """Accumulate wall-clock time per named stage.

    Use as a context manager:  with timer("coarse"): ...
    On CUDA, pass sync=True so async kernels are not mis-attributed.
    """

    def __init__(self, sync=False):
        self.totals = {}
        self.counts = {}
        self.samples = {}
        self.sync = sync
        self._name = None
        self._t0 = None

    def _maybe_sync(self):
        if self.sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._maybe_sync()
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self._maybe_sync()
        dt = time.time() - self._t0
        self.totals[self._name] = self.totals.get(self._name, 0.0) + dt
        self.counts[self._name] = self.counts.get(self._name, 0) + 1
        self.samples.setdefault(self._name, []).append(dt)
        return False

    def last(self, name):
        return self.samples[name][-1]

    def report(self, per_query_stages=(), title="TIMING"):
        print(f"\n{title}")
        for name, total in self.totals.items():
            s = np.asarray(self.samples[name])
            if name in per_query_stages and len(s) > 1:
                unit, scale = ("ms", 1000.0) if np.median(s) < 0.5 else ("s", 1.0)
                print(f"  {name:22s}: median={np.median(s)*scale:8.2f} {unit}  "
                      f"mean={s.mean()*scale:8.2f} {unit}  total={total:7.2f} s "
                      f"(n={len(s)})")
            else:
                print(f"  {name:22s}: {total:7.2f} s")


class PoseMetrics:
    """Collect per-query pose errors and report aggregate accuracy."""

    def __init__(self, rot_target=10.0, trans_target=0.1):
        self.rot_target = rot_target
        self.trans_target = trans_target
        self.rows = []

    def add(self, name, rot_deg, trans, extra=None):
        ok = rot_deg < self.rot_target and trans < self.trans_target
        self.rows.append({"name": name, "rot": rot_deg, "trans": trans,
                          "ok": ok, **(extra or {})})
        return ok

    @property
    def rot(self):
        return np.asarray([r["rot"] for r in self.rows], np.float64)

    @property
    def trans(self):
        return np.asarray([r["trans"] for r in self.rows], np.float64)

    def header(self, extra_cols=()):
        cols = f"{'query':44s} {'rot°':>7} {'trans':>8}"
        for c in extra_cols:
            cols += f" {c:>9}"
        cols += f" {'ok':>3}"
        print(cols)
        print("-" * len(cols))

    def print_row(self, extra_cols=()):
        r = self.rows[-1]
        line = f"{r['name'][:44]:44s} {r['rot']:7.3f} {r['trans']:8.4f}"
        for c in extra_cols:
            v = r.get(c, "")
            line += f" {v:9.3f}" if isinstance(v, float) else f" {str(v):>9}"
        line += f" {'Y' if r['ok'] else 'N':>3}"
        print(line)

    def report(self):
        n = len(self.rows)
        r, t = self.rot, self.trans
        npass = sum(1 for x in self.rows if x["ok"])
        print("\n" + "=" * 74)
        print(f"ACCURACY over {n} queries "
              f"(targets: rot < {self.rot_target}°, trans < {self.trans_target})")
        print(f"  rotation °  : median={np.median(r):7.3f}  mean={r.mean():7.3f}  "
              f"max={r.max():7.3f}   pass {int((r < self.rot_target).sum())}/{n}")
        print(f"  translation : median={np.median(t):7.4f}  mean={t.mean():7.4f}  "
              f"max={t.max():7.4f}   pass {int((t < self.trans_target).sum())}/{n}")
        print(f"  BOTH pass   : {npass}/{n}  ({100.0*npass/n:.1f}%)")
        return {"n": n, "rot_median": float(np.median(r)), "rot_mean": float(r.mean()),
                "rot_max": float(r.max()), "trans_median": float(np.median(t)),
                "trans_mean": float(t.mean()), "trans_max": float(t.max()),
                "pass": npass}

    def hypothesis_summary(self, key="sel"):
        """How often the coarse top-1 was not the selected hypothesis."""
        sels = [r[key] for r in self.rows if key in r]
        if not sels:
            return
        n, not_top1 = len(sels), sum(1 for s in sels if s != 0)
        print(f"\n  hypothesis selected: " +
              ", ".join(f"#{k}:{sels.count(k)}" for k in sorted(set(sels))))
        print(f"  coarse top-1 was NOT best in {not_top1}/{n} "
              f"({100.0*not_top1/n:.0f}%) -> multi-hypothesis search matters")
