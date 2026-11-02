"""Run the full grid (outer split x variant x seed) on one or more GPUs.

Usage:
    python scripts/run_all.py --data-dir cache --gpus 0 1 --workers-per-gpu 3
Each job is a separate `run_split.py` process; jobs whose result.json exists are skipped, so the script can be
restarted. Logs go to <out>/logs/.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {  # dataset: variants
    "isu": ["CPT", "CPT-Delta", "CPMLP"],
    "hust": ["CPT", "CPT-Delta", "CPMLP"],
    "snl": ["CPT", "CPT-Delta", "CPMLP"],
}
SEEDS = [2021, 42, 2024]


def jobs(datasets, splits_dir):
    out = []
    for ds in datasets:
        ids = [s["id"] for s in json.load(open(os.path.join(splits_dir, f"{ds}.json")))["splits"]]
        out += [(ds, sp, v, s) for sp in ids for v in GRID[ds] for s in SEEDS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(GRID), choices=list(GRID))
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--splits-dir", default=os.path.join(ROOT, "splits"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    a = ap.parse_args()
    logd = os.path.join(a.out, "logs"); os.makedirs(logd, exist_ok=True)
    todo = [j for j in jobs(a.datasets, a.splits_dir)
            if not os.path.exists(os.path.join(a.out, j[0], j[1], f"{j[2]}_s{j[3]}", "result.json"))]
    print(f"{len(todo)} jobs to run on GPUs {a.gpus}", flush=True)
    running, done, failed = {}, 0, 0
    while todo or running:
        while todo and len(running) < a.workers_per_gpu * len(a.gpus):
            load = {g: sum(1 for (_, _, gg) in running.values() if gg == g) for g in a.gpus}
            g = min(a.gpus, key=lambda x: load[x])
            ds, sp, v, s = todo.pop(0)
            name = f"{ds}_{sp}_{v}_s{s}"
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_split.py"), "--dataset", ds, "--split", sp,
                   "--variant", v, "--seed", str(s), "--data-dir", a.data_dir, "--splits-dir", a.splits_dir,
                   "--out", a.out, "--device", "cuda:0"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            running[name] = (subprocess.Popen(cmd, env=env, stdout=open(os.path.join(logd, name + ".log"), "w"),
                                              stderr=subprocess.STDOUT), time.time(), g)
        time.sleep(5)
        for name, (p, t, g) in list(running.items()):
            if p.poll() is not None:
                del running[name]
                done, failed = (done + 1, failed) if p.returncode == 0 else (done, failed + 1)
                print(f"{'ok' if p.returncode == 0 else 'FAILED'} {name} GPU {g} {time.time() - t:.0f}s | "
                      f"done {done}, failed {failed}, left {len(todo) + len(running)}", flush=True)
    print(f"finished: done {done}, failed {failed}")


if __name__ == "__main__":
    main()
