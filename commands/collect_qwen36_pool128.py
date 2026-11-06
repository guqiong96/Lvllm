import json, os, time, hashlib, urllib.request, sys
from collections import Counter

URL = "http://127.0.0.1:8070"
ARM = os.environ.get("ARM", "plain")
OUT_DIR = os.environ.get(
    "OUT_DIR",
    f"/home/guqiong/Downloads/opencode/ab_20261019/qwen36-pool128-subm32768-{ARM}",
)
DET_PROMPT = "用一个长段落介绍杭州西湖的历史、景色和美食，要求信息密集。"
FISH = ("Make an html animation of fishes in an aquarium. The aquarium is pretty, "
        "the fishes vary in colors and sizes and swim realistically. You can left "
        "click to place a piece of fish food in aquarium. Each fish chases a food "
        "piece closest to it, trying to eat it. Once there are no more food pieces, "
        "fishes resume swimming as usual.")
WORDS = ["alphabet","bicycle","compass","diamond","elephant","falcon","galaxy","harvest",
         "island","jaguar","kettle","lantern","mountain","nebula","orchard","puzzle"]


def post(path, payload, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application-type/json"})
    req.add_header("Content-Type", "application/json")
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return out, time.time() - t


def gen(text, n, temp=0.0, **extra):
    sp = {"temperature": temp, "max_new_tokens": n}
    sp.update(extra)
    return post("/generate", {"text": text, "sampling_params": sp})


def h(txt):
    return hashlib.sha256(txt.encode("utf-8", "replace")).hexdigest()[:16]


def probe_det(n, reps, tag):
    seqs = []
    for _ in range(reps):
        out, dt = gen(DET_PROMPT, n)
        seqs.append(out["text"])
        time.sleep(0.4)
    uniq = len(set(seqs))
    return {"tag": tag, "n_tokens": n, "reps": reps, "uniq_full_text": uniq,
            "hashes": dict(Counter(h(s) for s in seqs)),
            "lens": sorted(len(s) for s in seqs)}


def probe_bs2(tag):
    import threading
    res = []
    def one():
        out, dt = gen(DET_PROMPT, 48)
        res.append((h(out["text"]), round(dt, 2)))
    pairs = []
    for _ in range(3):
        res.clear()
        ts = [threading.Thread(target=one) for _ in range(2)]
        [t.start() for t in ts]; [t.join() for t in ts]
        pairs.append(list(res))
        time.sleep(0.5)
    return {"tag": tag, "pairs": pairs,
            "intra_pair_identical": all(a[0] == b[0] for a, b in pairs),
            "pair_hashes": [p[0][0] for p in pairs]}


def probe_speed():
    out, dt = gen(DET_PROMPT, 512)
    n = out["meta_info"].get("completion_tokens", 0)
    return {"tag": "speed", "completion_tokens": n, "wall_s": round(dt, 2),
            "tok_s": round(n / dt, 1)}


def probe_fish(reps, n=4096):
    rows = []
    for _ in range(reps):
        out, dt = gen(FISH, n, temp=1.0)
        txt = out["text"]
        grams = [txt[k:k+24] for k in range(0, max(len(txt)-24, 1), 8)]
        rep = 1 - len(set(grams)) / max(len(grams), 1)
        rows.append({"len": len(txt), "rep_ratio": round(rep, 4),
                     "finish": out["meta_info"].get("finish_reason", {}),
                     "hash": h(txt), "wall_s": round(dt, 1)})
        time.sleep(0.5)
    return {"tag": "fish_quality", "samples": rows}


def probe_long(words_n):
    import random
    random.seed(42)
    words = WORDS * (words_n // len(WORDS) + 1)
    prompt = " ".join(random.sample(words, words_n)) + "\nQuestion: list the first and last word. Answer:"
    out, dt = post("/generate", {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 16}}, timeout=1200)
    mi = out["meta_info"]
    pt = mi.get("prompt_tokens", 0)
    return {"tag": "long_prefill", "words": words_n, "prompt_tokens": pt,
            "wall_s": round(dt, 1), "prefill_tok_s": round(pt / dt, 1),
            "text": out["text"][:120]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    info = json.loads(urllib.request.urlopen(URL + "/model_info", timeout=10).read())
    data = {"server": {"launch": os.environ.get("LAUNCH", f"{ARM} qwen3.6 pool128 subm32768"),
                       "base_sha": os.environ.get("BASE_SHA", "worktree"), "side": ARM,
                       "model": (info.get("model_path") or info.get("path") or info.get("model")
                   or (info.get("server_args") or {}).get("model_path") or "?").split("/")[-1],
            "model_info": info, "collected_at": time.strftime("%F %T")},
            "probes": []}
    steps = [lambda: probe_det(32, 12, "det32x12"),
             lambda: probe_det(512, 4, "det512x4_track_boundary"),
             lambda: probe_bs2("bs2_pairs"),
             probe_speed,
             lambda: probe_long(6000),
             lambda: probe_fish(3)]
    for i, fn in enumerate(steps):
        try:
            r = fn()
        except Exception as e:
            r = {"error": repr(e), "step": i}
        data["probes"].append(r)
        print(json.dumps(r, ensure_ascii=False)[:300], flush=True)
        with open(OUT_DIR + "/results.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    print("DONE")


main()
