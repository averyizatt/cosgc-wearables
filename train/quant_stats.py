"""How much of a fine-tune survives int8 weight rounding, and how much a better per-row scale could help (CPU only).

python quant_stats.py <fine-tune last.pt> [<starting weights .pt>]

Prints, per layer type: rounding error against the size of the fine-tune's change (ratio near or above 1 means the rounding noise
is as large as what training changed), the row's max against its rms, and the error of the current scale (max/127) against
the row-by-row MSE-optimal scale (docs/EXPERIMENTS.md, E8b).
"""
import re
import sys
import torch


def state(path):
    st = torch.load(path, map_location="cpu")
    sd = st["ema"] if isinstance(st, dict) and "ema" in st else st["model"] if isinstance(st, dict) and "model" in st else st
    return {k.replace("_orig_mod.", ""): v.float() for k, v in sd.items() if hasattr(v, "dtype") and v.dtype.is_floating_point}


def rows(w):
    return w.reshape(w.shape[0], -1)


def rnd_err(w, s):
    return ((torch.clamp(torch.round(w / s), -127, 127) * s - w) ** 2).sum(1)


def best_scale(w):
    mx = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
    best, bs = rnd_err(w, mx), mx.clone()
    for t in torch.linspace(0.30, 0.99, 70):
        e = rnd_err(w, mx * t)
        better = e < best
        best = torch.where(better, e, best)
        bs = torch.where(better[:, None], mx * t, bs)
    return bs, best


def kind(name):
    m = re.search(r"(ff1_l1|ff1_l2|ff2_l1|ff2_l2|linear_q|linear_k|linear_v|linear_out|pw1|pw2|pos|head|ctc|decoder|sub|conv\d?)", name)
    return m.group(1) if m else "other"


def main():
    a = state(sys.argv[1])
    b = state(sys.argv[2]) if len(sys.argv) > 2 else None
    groups = {}
    for name, w in a.items():
        if not name.endswith("weight") or w.dim() != 2 or min(w.shape) < 32:
            continue
        w = rows(w)
        mx = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        e_max = rnd_err(w, mx).sum().item()
        _, e_best = best_scale(w)
        g = groups.setdefault(kind(name), dict(n=0, sq=0.0, max=0.0, best=0.0, delta=0.0, hasd=0, peak=[]))
        g["n"] += w.numel()
        g["sq"] += (w ** 2).sum().item()
        g["max"] += e_max
        g["best"] += e_best.sum().item()
        g["peak"].append((w.abs().amax(1) / w.pow(2).mean(1).sqrt().clamp(min=1e-12)).median().item())
        if b is not None and name in b and b[name].shape[0] == a[name].shape[0] and b[name].numel() == a[name].numel():
            g["delta"] += ((w - rows(b[name])) ** 2).sum().item()
            g["hasd"] += w.numel()
    print("%-11s %9s %9s %12s %10s %12s %14s" % ("layer", "weights", "max/rms", "SNR max dB", "SNR best", "gain dB", "noise/change"))
    tot = dict(sq=0.0, max=0.0, best=0.0, delta=0.0)
    for k in sorted(groups):
        g = groups[k]
        for t in tot:
            tot[t] += g[t]
        r = (g["max"] / g["delta"]) ** 0.5 if g["delta"] > 0 else float("nan")
        import math
        print("%-11s %9d %9.1f %12.1f %10.1f %12.2f %14.2f" % (k, g["n"], sum(g["peak"]) / len(g["peak"]), 10 * math.log10(g["sq"] / g["max"]),
              10 * math.log10(g["sq"] / g["best"]), 10 * math.log10(g["max"] / g["best"]), r))
    import math
    print("%-11s %9s %9s %12.1f %10.1f %12.2f %14.2f" % ("ALL", "", "", 10 * math.log10(tot["sq"] / tot["max"]), 10 * math.log10(tot["sq"] / tot["best"]),
          10 * math.log10(tot["max"] / tot["best"]), (tot["max"] / tot["delta"]) ** 0.5 if tot["delta"] else float("nan")))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
