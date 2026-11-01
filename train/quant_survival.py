"""How much of a fine-tune's weight change survives int8 rounding, for different grids (CPU only).

python quant_survival.py <fine-tune last.pt> <starting weights .pt> [qmax ...]

survive = (q - base) . (w - base) / |w - base|^2 over all rows: 1 means the rounded weights keep the whole change, 0 means they
lost it. moved = share of weights whose rounded value differs from the rounded starting value. snr = |w|^2 / |q - w|^2 in dB.
With qmax 127 the shipped model's weights are already on the grid, so changes under half a step round back to where they
started. Any other qmax misaligns the grid (docs/EXPERIMENTS.md, E8b).
"""
import math
import sys
import torch


def state(path):
    st = torch.load(path, map_location="cpu")
    sd = st["ema"] if isinstance(st, dict) and "ema" in st else st["model"] if isinstance(st, dict) and "model" in st else st
    return {k.replace("_orig_mod.", ""): v.float() for k, v in sd.items() if hasattr(v, "dtype") and v.dtype.is_floating_point}


def quant(w, qmax):
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s, s


def main():
    a, b = state(sys.argv[1]), state(sys.argv[2])
    names = [k for k, w in a.items() if k.endswith("weight") and w.dim() == 2 and min(w.shape) >= 32 and k in b and b[k].shape == w.shape]
    print("%d weight matrices compared" % len(names))
    print("%6s %9s %8s %8s" % ("qmax", "survive", "moved", "snr dB"))
    for qmax in [float(x) for x in sys.argv[3:]] or [127.0, 125.0, 121.0, 113.0]:
        dot = dd = sq = err = moved = n = 0.0
        for k in names:
            w, base = a[k], b[k]
            q, s = quant(w, qmax)
            qb, _ = quant(base, qmax)
            d = w - base
            dot += ((q - base) * d).sum().item()
            dd += (d * d).sum().item()
            sq += (w * w).sum().item()
            err += ((q - w) ** 2).sum().item()
            moved += ((q - qb).abs() > s * 0.5).sum().item()
            n += w.numel()
        print("%6.1f %9.3f %8.3f %8.1f" % (qmax, dot / dd, moved / n, 10 * math.log10(sq / err)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
