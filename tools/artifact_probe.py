import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, numpy as np
FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
SCAN = {"ctu-idseval-6-malicious-portscan-1","ctu-idseval-6-malicious-portscan-2","ctu-idseval-6-malicious-portscan-3"}
BEN = "ctu-idseval-6-benign-user-traffic-1"
scan, ben = [], []
for line in open(FLOWS):
    r = json.loads(line)
    (scan if r["capture"] in SCAN else ben if r["capture"] == BEN else []).append(r)
print("scan:", len(scan), "benign:", len(ben))

def feats(r):
    t = r["tokens"][0]
    pk = sum(r["counts"])
    return np.array([t[4], t[6], t[2], t[5], pk, len(r["tokens"]), max(r["iats"])], dtype=np.float64)

X = np.array([feats(r) for r in scan] + [feats(r) for r in ben])
y = np.array([1]*len(scan) + [0]*len(ben))
names = ["ttl","tcp_window","ip_total_len","tcp_flags","n_packets","n_bursts","max_iat"]
print(f"{'feature':16s} {'AUC':>6s} {'scan_mean':>10s} {'benign_mean':>12s} {'overlap':>8s}")
def auc(s, y):
    pos, neg = s[y==1], s[y==0]
    allv = np.concatenate([pos,neg]); order = allv.argsort(); ranks = np.empty_like(order, float); ranks[order]=np.arange(1,len(allv)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
for i,n in enumerate(names):
    a = auc(X[:,i], y)
    rng_overlap = ((X[:,i].min(), X[:,i].max()))
    print(f"{n:16s} {max(a,1-a):6.3f} {X[y==1,i].mean():10.2f} {X[y==0,i].mean():12.2f}  scan[{X[y==1,i].min():.0f},{X[y==1,i].max():.0f}] ben[{X[y==0,i].min():.0f},{X[y==0,i].max():.0f}]")

# single-rule ceilings
for name, rule in [("win==1024", X[:,1]==1024), ("tl==44", X[:,2]==44), ("ttl<64", X[:,0]<64), ("pkts==1", X[:,4]==1)]:
    pred = rule.astype(int)
    tp = ((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); tn=((pred==0)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-9)
    pn = tn/max(tn+fn,1); rn = tn/max(tn+fp,1); f1n = 2*pn*rn/max(pn+rn,1e-9)
    print(f"rule {name:12s} macroF1={(f1+f1n)/2:.4f} prec={p:.3f} rec={r:.3f} FPR={fp/max(fp+tn,1):.3f}")
