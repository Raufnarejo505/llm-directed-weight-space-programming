# Dataset

Nothing in this repository redistributes network traffic. The pipeline consumes:

## CTU-IDSEVAL-6

* Zenodo record: https://zenodo.org/records/21027042
* files used: `pcap.zip` (38 MB, six captures) and `zeek.zip` (7 MB, labelled conn logs)
* licence: see the Zenodo record; cite the dataset authors if you use it

Captures used in this study (TCP flows after conversion):

| capture | packets | TCP flows | label |
|---|---|---|---|
| ctu-idseval-6-benign-user-traffic-1 | 75,356 | 761 | benign / background |
| ctu-idseval-6-malicious-portscan-1 | 2,009 | 1,999 | malicious (src 147.32.80.37) |
| ctu-idseval-6-malicious-portscan-2 | 206 | 200 | malicious |
| ctu-idseval-6-malicious-portscan-3 | 131,237 | 131,148 | malicious |

Layout expected by `tools/build_dataset.py`:

    data/ctu/pcap/pcap/ctu-idseval-6-*.pcap
    data/ctu/zeek/zeek/*.conn-labeled.log

## Conversion

`tools/build_dataset.py` is a Python port of the upstream preprocessing
(`pre_process_src/Tokenize.py` + `3_field_extraction.cpp`):

* bidirectional conversation keyed by 5-tuple (pcap_splitter behaviour);
* direction +1 for the initiator (source of the first packet);
* bursts split on a >10 ms inter-packet gap, per direction, merged by start time;
* max 12 bursts, max 6 packets per burst, 12 header tokens per TCP packet;
* seq/ack made relative to the flow start;
* payload dropped, matching the released `strip_payload=True` configuration.

## Why the corpus cannot support a capability claim

The two classes differ by IP TTL (37-59 vs 128), TCP window (1024 vs 8192) and IP total
length (40-44 vs 48-52) with **disjoint ranges**. A single rule `ttl < 64` scores macro F1
1.0000. The corpus is used strictly as a controlled testbed for representation geometry;
see docs/02_dataset_and_baseline.md.
