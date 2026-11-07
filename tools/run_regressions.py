import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Run the fixed regression suite over a list of checkpoints, sequentially."""
import json, sys, os, time, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
torch.set_num_threads(2)
import nfcommon as nf
from regression2 import build_suite, run

RES = os.path.join(ROOT, "results")
PAT = os.path.join(ROOT, "checkpoints", "patched")

def main(ids):
    flows = build_suite()
    for pid in ids:
        out = os.path.join(RES, f"regression_{pid}.json")
        if os.path.exists(out):
            print("skip", pid, flush=True); continue
        ck = None if pid == "orig" else os.path.join(PAT, pid)
        t0 = time.time()
        model = nf.load_model(ck) if ck else nf.load_model()
        r = run(model, flows)
        r["checkpoint"] = pid
        json.dump(r, open(out, "w"), indent=2)
        print(f"{pid}: {json.dumps(r)} ({time.time()-t0:.0f}s)", flush=True)
        del model

if __name__ == "__main__":
    main(sys.argv[1:])
