import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Convert CTU-IDSEVAL-6 pcaps into netFound flow records.

Faithful port of pre_process_src/Tokenize.py + 3_field_extraction semantics:
  * bidirectional conversations keyed by 5-tuple (pcap_splitter behaviour)
  * direction = +1 for the initiator (first packet's source), -1 for the reverse
  * bursts split on a >10 ms gap between packets, per direction, then merged by start time
  * max 12 bursts per flow, max 6 packets per burst
  * 12 header tokens per TCP packet (payload dropped, strip_payload=True config)
  * seq/ack made relative to the flow start (3_field_extraction.cpp)
"""
import json, os, sys, glob
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP, ICMP

BURST_SPLIT_NS = 10_000_000     # tokenize.py: 10 ms
MAX_BURSTS, MAX_PKTS = 12, 6


def flow_key(ip):
    if IP not in ip:
        return None
    l4 = ip.payload
    if TCP in l4 or UDP in l4:
        sport, dport = int(l4.sport), int(l4.dport)
    elif ICMP in l4:
        sport, dport = 0, 0
    else:
        return None
    a, b = (ip.src, sport), (ip.dst, dport)
    return (min(a, b), max(a, b), int(ip.proto))


def parse_pcap(path, capture):
    flows = {}
    for pkt in PcapReader(path):
        if IP not in pkt:
            continue
        ip = pkt[IP]
        k = flow_key(ip)
        if k is None:
            continue
        ts = int(round(float(pkt.time) * 1e9))
        rec = dict(ts=ts, src=ip.src, dst=ip.dst, ihl=ip.ihl * 4, tos=int(ip.tos),
                   tl=int(ip.len), ipflags=int(ip.flags), ttl=int(ip.ttl), proto=int(ip.proto),
                   tcpflags=0, win=0, seq=0, ack=0, urp=0, payload=0)
        l4 = ip.payload
        if TCP in l4:
            t = l4[TCP]
            rec.update(tcpflags=int(t.flags), win=int(t.window), seq=int(t.seq),
                       ack=int(t.ack), urp=int(t.urgptr), payload=len(bytes(t.payload)))
        elif UDP in l4:
            rec["payload"] = len(bytes(l4[UDP].payload))
        elif ICMP in l4:
            c = l4[ICMP]
            rec.update(tcpflags=int(c.type) * 256 + int(c.code), payload=len(bytes(c.payload)))
        flows.setdefault(k, []).append(rec)
    return flows


def normalize_seq(pkts):
    """3_field_extraction.cpp: subtract the absolute seq/ack seen at flow start."""
    abs_seq = pkts[0]["seq"]
    abs_ack = pkts[0]["ack"]
    if abs_ack == 0 and len(pkts) > 1:
        abs_ack = pkts[1]["seq"]
    init_src = pkts[0]["src"]
    out = []
    for p in pkts:
        q = dict(p)
        if p["proto"] == 6:
            if p["src"] == init_src:
                q["seq"] = (p["seq"] - abs_seq) & 0xFFFFFFFF
                q["ack"] = 0 if abs_ack == 0 else (p["ack"] - abs_ack) & 0xFFFFFFFF
            else:
                q["seq"] = (p["seq"] - abs_ack) & 0xFFFFFFFF
                q["ack"] = (p["ack"] - abs_seq) & 0xFFFFFFFF
        out.append(q)
    return out


def packets_to_tokens(p):
    """12 header tokens for TCP/UDP/ICMP per the tokenizer config."""
    seq, ack = p["seq"], p["ack"]
    toks = [p["ihl"], p["tos"], p["tl"] & 0xFFFF, p["ipflags"], p["ttl"]]
    if p["proto"] == 6:
        toks += [p["tcpflags"], p["win"] & 0xFFFF,
                 (seq >> 16) & 0xFFFF, seq & 0xFFFF,
                 (ack >> 16) & 0xFFFF, ack & 0xFFFF,
                 p["urp"] & 0xFFFF]
    elif p["proto"] == 17:
        toks += [p["payload"] & 0xFFFF]      # UDP length token
    else:
        toks += [p["tcpflags"] >> 8, p["tcpflags"] & 0xFF]  # icmp type, code
    return toks


def split_bursts(pkts):
    bursts, cur = [], []
    for p in pkts:
        if cur and p["ts"] - cur[-1]["ts"] > BURST_SPLIT_NS:
            bursts.append(cur); cur = []
        cur.append(p)
    if cur:
        bursts.append(cur)
    return bursts


def build_flow(pkts, meta):
    pkts = normalize_seq(pkts)
    init_src = pkts[0]["src"]
    fwd = [p for p in pkts if p["src"] == init_src]
    bwd = [p for p in pkts if p["src"] != init_src]
    bursts = []
    for direction, group in ((1, fwd), (-1, bwd)):
        for b in split_bursts(group):
            bursts.append((b[0]["ts"], direction, b[:MAX_PKTS]))
    bursts.sort(key=lambda x: x[0])
    bursts = bursts[:MAX_BURSTS]
    if not bursts:
        return None
    tokens, directions, byte_list, counts, iats = [], [], [], [], []
    prev_start = None
    for start, direction, b in bursts:
        t = []
        for p in b:
            t += packets_to_tokens(p)
        tokens.append(t)
        directions.append(direction)
        byte_list.append(int(sum(p["tl"] for p in b)))
        counts.append(len(b))
        iats.append(0 if prev_start is None else int(round((start - prev_start) / 1e6)))
        prev_start = start
    rec = dict(capture=meta["capture"], label=meta["label"], tokens=tokens,
               directions=directions, bytes=byte_list, counts=counts, iats=iats,
               protocol=int(bursts[0][2][0]["proto"]),
               flow_duration=int(round((pkts[-1]["ts"] - pkts[0]["ts"]) / 1e6)),
               t0=pkts[0]["ts"],
               sip=pkts[0]["src"], sport=0, dip=pkts[0]["dst"])
    return rec


def main(pcap_glob, out_path):
    recs = []
    for path in sorted(glob.glob(pcap_glob)):
        cap = os.path.basename(path).replace(".pcap", "")
        t0 = __import__("time").time()
        flows = parse_pcap(path, cap)
        n = 0
        for k, pkts in flows.items():
            if pkts[0]["proto"] != 6:   # TCP only for this study
                continue
            pkts.sort(key=lambda p: p["ts"])
            r = build_flow(pkts, dict(capture=cap, label=None))
            if r is None:
                continue
            r["sport"] = int(k[0][1]); r["dip"] = k[1][0]; r["dport"] = int(k[1][1])
            r["sip"] = k[0][0]
            recs.append(r); n += 1
        print(f"{cap}: {len(flows)} flows, kept {n} TCP flows, {__import__('time').time()-t0:.1f}s")
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(recs), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
