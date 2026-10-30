#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Make a deeper Conformer from a trained one by inserting new layers (depth up-scaling), for the ESP32-P4.

    python grow_nemo.py --nemo G:\\cosgc\\exp\\base --init G:\\cosgc\\exp\\base\\init.pt --layers 24 --out G:\\cosgc\\exp\\base24

Writes <out>/state_dict_plain.pt (NeMo key names, which every script reads the model shape from), <out>/init.pt (the
weights, with the fixed feature normalization of --init) and copies tokenizer.model. Then train with
--nemo <out> --init <out>/init.pt as usual; export_nemo.py writes the new depth into the .tnm header and the engine reads it.

The new layers are spread evenly between the old ones. Each starts as a copy of the layer before it with the output
projections of its four residual branches (FF1, attention, convolution, FF2) set to zero, so it only re-applies that
layer's final LayerNorm: the grown model starts close to the old one (measure with eval_manifest.py) and the new
layers learn from there. Compute and size grow with the depth (16 -> 24 layers: 1.5x, 13 -> 19 M parameters).
"""
import argparse, copy, os, re, shutil
import torch
from nemo_small import NemoSmall

ZERO = ["ff1_l2", "linear_out", "pw2", "ff2_l2"]  # residual branch outputs (NemoSmall names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="dir with state_dict_plain.pt and tokenizer.model")
    ap.add_argument("--init", help="trained weights (init.pt or a run's last.pt); default: the --nemo weights")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = torch.load(f"{a.nemo}/state_dict_plain.pt", map_location="cpu")
    m = NemoSmall(sd)
    init = torch.load(a.init, map_location="cpu") if a.init else {}
    if init:
        m.load_state_dict(init["model"])
    n = len(m.layers)
    assert a.layers > n, f"--layers must be more than the current {n}"
    extra = a.layers - n
    after = {round((i + 1) * n / extra) - 1 for i in range(extra)}  # insert a new layer after these old ones
    assert len(after) == extra
    order = []  # (source old layer, is new)
    for i in range(n):
        order.append((i, False))
        if i in after:
            order.append((i, True))

    # NeMo-format state dict: only its shapes matter (the weights come from init.pt), so new layers reuse old keys
    new_sd = {k: v for k, v in sd.items() if not k.startswith("encoder.layers.")}
    for j, (i, _) in enumerate(order):
        for k, v in sd.items():
            if k.startswith(f"encoder.layers.{i}."):
                new_sd[f"encoder.layers.{j}." + k[len(f"encoder.layers.{i}."):]] = v.clone()
    g = NemoSmall(new_sd)
    assert len(g.layers) == a.layers
    with torch.no_grad():
        for name, mod in m.named_children():
            if name != "layers":
                getattr(g, name).load_state_dict(mod.state_dict())
        for j, (i, new) in enumerate(order):
            L = copy.deepcopy(m.layers[i])
            if new:
                for z in ZERO:
                    getattr(L, z).weight.zero_()
                    if getattr(L, z).bias is not None:
                        getattr(L, z).bias.zero_()
            g.layers[j] = L

    os.makedirs(a.out, exist_ok=True)
    torch.save(new_sd, f"{a.out}/state_dict_plain.pt")
    st = {k: v for k, v in init.items() if k in ("norm_mean", "norm_std", "stream")}
    torch.save({"model": g.state_dict(), "grown_from": a.init or a.nemo, "new_layers": [j for j, (_, nw) in enumerate(order) if nw],
                **st}, f"{a.out}/init.pt")
    shutil.copy(f"{a.nemo}/tokenizer.model", a.out)
    p0, p1 = sum(p.numel() for p in m.parameters()), sum(p.numel() for p in g.parameters())
    print(f"{n} -> {a.layers} layers, {p0/1e6:.1f} -> {p1/1e6:.1f} M parameters; new layers at "
          f"{[j for j, (_, nw) in enumerate(order) if nw]}; wrote {a.out}")


if __name__ == "__main__":
    main()
