import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, numpy as np, collections
FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
rows = [json.loads(l) for l in open(FLOWS)]
print("total flows:", len(rows))
print(collections.Counter(r["capture"] for r in rows))

def first_tokens(r):
    return r["tokens"][0][:12]

for cap in sorted(set(r["capture"] for r in rows)):
    rs = [r for r in rows if r["capture"] == cap]
    nb = np.array([len(r["tokens"]) for r in rs])
    npk = np.array([sum(r["counts"]) for r in rs])
    ttls = np.array([first_tokens(r)[4] for r in rs])
    wins = np.array([first_tokens(r)[6] for r in rs])
    tls = np.array([first_tokens(r)[2] for r in rs])
    flg = np.array([first_tokens(r)[5] for r in rs])
    print(f"\n{cap}")
    print(f"  bursts: med={np.median(nb):.0f} p90={np.percentile(nb,90):.0f} max={nb.max()} | packets: med={np.median(npk):.0f} max={npk.max()}")
    print(f"  ttl: uniq={len(set(ttls.tolist()))} top={collections.Counter(ttls.tolist()).most_common(5)}")
    print(f"  win: top={collections.Counter(wins.tolist()).most_common(5)}")
    print(f"  iptl: top={collections.Counter(tls.tolist()).most_common(5)}")
    print(f"  tcpflags: top={collections.Counter(flg.tolist()).most_common(5)}")
