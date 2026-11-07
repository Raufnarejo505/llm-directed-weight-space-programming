import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, struct, urllib.request
url = "https://huggingface.co/snlucsb/netFound-small/resolve/main/model.safetensors"
req = urllib.request.Request(url, headers={"Range": "bytes=0-800000"})
with urllib.request.urlopen(req, timeout=60) as r:
    data = r.read()
print("bytes:", len(data), "status:", r.status)
n = struct.unpack("<Q", data[:8])[0]
hdr = json.loads(data[8:8+n].decode("utf-8"))
meta = hdr.pop("__metadata__", None)
print("HEADER BYTES:", n, "TENSORS:", len(hdr), "META:", meta)
rows = []
tot = 0
for k, v in hdr.items():
    num = 1
    for d in v["shape"]: num *= d
    tot += num
    rows.append((k, v["dtype"], tuple(v["shape"]), num))
print("TOTAL PARAMS:", f"{tot:,}")
rows.sort()
for k, dt, shp, num in rows:
    print(f"{k:78s} {dt:5s} {str(shp):20s} {num:>10,}")
