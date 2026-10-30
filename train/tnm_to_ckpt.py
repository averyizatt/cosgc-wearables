#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Turn an exported engine model (TNM1, e.g. models/oido_stream.tnm) back into trainable float weights.

The repository ships exported models, not training checkpoints, and every fine-tuning or distillation run needs a float
starting point. This reads the int8 weights and their per-row scales, multiplies them back out, and writes the result in
NVIDIA NeMo's key names, so the existing training code loads it unchanged:

    python tnm_to_ckpt.py ../models/oido_stream.tnm out_dir            # writes out_dir/state_dict_plain.npz (+ .pt with torch)
    python tnm_to_ckpt.py ../models/oido_stream.tnm out_dir --verify   # also re-exports and compares with the original bytes

With PyTorch installed it also writes out_dir/state_dict_plain.pt (use as --nemo out_dir, with tokenizer.model copied
next to it) and out_dir/init.pt ({"model", "norm_mean", "norm_std", "stream"}; use as --init for streaming models so
training keeps the model's fixed feature normalization).

The weights are the int8 values times their scale: within half a quantization step of the trained floats. Fine-tuning
repairs that. int4 models (nemo4.tnm) work the same way but start further from the original floats.
"""
import argparse, os, struct, sys
import numpy as np


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 64

    def align(self, a=16):
        self.p = (self.p + a - 1) // a * a

    def f32(self, n):
        self.align()
        v = np.frombuffer(self.b, "<f4", n, self.p).copy()
        self.p += 4 * n
        return v

    def qlin(self, n, k, bits):
        """inverse of export.W.qlin: returns (w float (n, k), bias (n,), raw int weights (n, kp))"""
        blocked = bits == 4 and n % 16 == 0
        blk = 16 if (bits == 8 or blocked) else 32
        kp = (k + blk - 1) // blk * blk
        self.align()
        if bits == 8:
            q = np.frombuffer(self.b, np.int8, n * kp, self.p).reshape(n, kp).astype(np.int16)
            self.p += n * kp
        elif blocked:  # (B, kpair, j) bytes: lo = w[n0+j][2*kpair], hi = w[n0+j][2*kpair+1]
            u = np.frombuffer(self.b, np.uint8, n * kp // 2, self.p).reshape(n // 16, kp // 2, 16)
            self.p += n * kp // 2
            lo, hi = (u & 15).astype(np.int16) - 8, (u >> 4).astype(np.int16) - 8
            q = np.stack([lo, hi], -1).transpose(0, 2, 1, 3).reshape(n, kp)
        else:  # per row, per 32-block: byte j = w[j] | w[j+16] << 4
            u = np.frombuffer(self.b, np.uint8, n * kp // 2, self.p).reshape(n, kp // 32, 16)
            self.p += n * kp // 2
            lo, hi = (u & 15).astype(np.int16) - 8, (u >> 4).astype(np.int16) - 8
            q = np.concatenate([lo, hi], -1).reshape(n, kp)
        s = self.f32(n)
        b = self.f32(n)
        return (q[:, :k] * s[:, None]).astype(np.float32), b, q


def parse(buf):
    assert buf[:4] == b"TNM1", "not a TNM1 model"
    hd = struct.unpack("<15I", buf[4:64])
    _, d, h, ff, K, nl, C, V, bits, rnnt, flags, chunk, left = hd[:13]
    assert not rnnt, "transducer models are not supported here"
    r = Reader(buf)
    sd = {}
    WIN, NMEL, NBIN = 400, 80, 257
    sd["preprocessor.featurizer.window"] = r.f32(WIN)
    sd["preprocessor.featurizer.fb"] = r.f32(NBIN * NMEL).reshape(NBIN, NMEL).T[None].copy()  # NeMo: (1, 80, 257)
    sd["encoder.pre_encode.conv.0.weight"] = r.f32(C * 9).reshape(C, 1, 3, 3)
    sd["encoder.pre_encode.conv.0.bias"] = r.f32(C)
    w, b, _ = r.qlin(C, C * 9, 8)
    sd["encoder.pre_encode.conv.2.weight"], sd["encoder.pre_encode.conv.2.bias"] = w.reshape(C, C, 3, 3), b
    f1 = (NMEL + 2 - 3) // 2 + 1
    f2 = (f1 + 2 - 3) // 2 + 1
    w, b, _ = r.qlin(d, C * f2, 8)
    sd["encoder.pre_encode.out.weight"], sd["encoder.pre_encode.out.bias"] = w, b
    for i in range(nl):
        p = f"encoder.layers.{i}."

        def ln(name):
            sd[p + name + ".weight"] = r.f32(d)
            sd[p + name + ".bias"] = r.f32(d)

        def lin(name, n, k, bt):
            w, b, _ = r.qlin(n, k, bt)
            sd[p + name + ".weight"], sd[p + name + ".bias"] = w, b

        ln("norm_feed_forward1")
        lin("feed_forward1.linear1", ff, d, bits); lin("feed_forward1.linear2", d, ff, bits)
        ln("norm_self_att")
        w, b, _ = r.qlin(3 * d, d, bits)
        for j, n in enumerate("qkv"):
            sd[p + f"self_attn.linear_{n}.weight"] = w[j * d:(j + 1) * d].copy()
            sd[p + f"self_attn.linear_{n}.bias"] = b[j * d:(j + 1) * d].copy()
        lin("self_attn.linear_out", d, d, bits)
        w, _, _ = r.qlin(d, d, 8)
        sd[p + "self_attn.linear_pos.weight"] = w
        sd[p + "self_attn.pos_bias_u"] = r.f32(d).reshape(h, d // h)
        sd[p + "self_attn.pos_bias_v"] = r.f32(d).reshape(h, d // h)
        ln("norm_conv")
        w, b, _ = r.qlin(2 * d, d, bits)
        sd[p + "conv.pointwise_conv1.weight"], sd[p + "conv.pointwise_conv1.bias"] = w[:, :, None], b
        sd[p + "conv.depthwise_conv.weight"] = r.f32(d * K).reshape(d, 1, K)
        sd[p + "conv.depthwise_conv.bias"] = r.f32(d)
        # the export folded batch norm into the depthwise conv; an identity batch norm keeps NemoSmall's folding exact
        sd[p + "conv.batch_norm.weight"] = np.ones(d, np.float32)
        sd[p + "conv.batch_norm.bias"] = np.zeros(d, np.float32)
        sd[p + "conv.batch_norm.running_mean"] = np.zeros(d, np.float32)
        sd[p + "conv.batch_norm.running_var"] = np.full(d, 1.0 - 1e-5, np.float32)
        w, b, _ = r.qlin(d, d, bits)
        sd[p + "conv.pointwise_conv2.weight"], sd[p + "conv.pointwise_conv2.bias"] = w[:, :, None], b
        ln("norm_feed_forward2")
        lin("feed_forward2.linear1", ff, d, bits); lin("feed_forward2.linear2", d, ff, bits)
        ln("norm_out")
    w, b, _ = r.qlin(V + 1, d, 8)
    sd["decoder.decoder_layers.0.weight"], sd["decoder.decoder_layers.0.bias"] = w[:, :, None], b
    r.align()
    toks = []
    for _ in range(V):
        n = buf[r.p]; toks.append(buf[r.p + 1:r.p + 1 + n].decode()); r.p += 1 + n
    extra = {"flags": flags, "chunk": chunk, "left": left, "bits": bits, "vocab": V, "d": d, "layers": nl}
    if flags & 1:
        extra["norm_mean"], extra["norm_std"] = r.f32(NMEL), r.f32(NMEL)
    assert r.p == len(buf), f"parsed {r.p} of {len(buf)} bytes"
    return sd, toks, extra


def reexport(buf, sd, extra):
    """Re-quantize the floats with export.W.qlin's arithmetic (numpy) and rebuild the file: must equal the original."""
    out = bytearray(buf[:64])

    def align():
        while len(out) % 16: out.extend(b"\0")

    def f32(v):
        align(); out.extend(np.ascontiguousarray(v, "<f4").tobytes())

    def qlin(w, b, bits):
        w = w.reshape(w.shape[0], -1).astype(np.float32); n, k = w.shape
        qmax = 2 ** (bits - 1) - 1
        s = np.maximum(np.abs(w).max(1), 1e-8).astype(np.float32) / np.float32(qmax)
        q = np.clip(np.round(w / s[:, None]), -qmax, qmax).astype(np.int16)
        assert bits == 8, "verify supports int8 models"
        kp = (k + 15) // 16 * 16
        qp = np.zeros((n, kp), np.int8); qp[:, :k] = q
        align(); out.extend(qp.tobytes()); f32(s); f32(b if b is not None else np.zeros(n, np.float32))

    g = sd.__getitem__
    f32(g("preprocessor.featurizer.window")); f32(g("preprocessor.featurizer.fb")[0].T)
    f32(g("encoder.pre_encode.conv.0.weight")); f32(g("encoder.pre_encode.conv.0.bias"))
    qlin(g("encoder.pre_encode.conv.2.weight"), g("encoder.pre_encode.conv.2.bias"), 8)
    qlin(g("encoder.pre_encode.out.weight"), g("encoder.pre_encode.out.bias"), 8)
    bits = extra["bits"]
    for i in range(extra["layers"]):
        p = f"encoder.layers.{i}."
        ln = lambda n: (f32(g(p + n + ".weight")), f32(g(p + n + ".bias")))
        lin = lambda n, bt: qlin(g(p + n + ".weight"), g(p + n + ".bias"), bt)
        ln("norm_feed_forward1"); lin("feed_forward1.linear1", bits); lin("feed_forward1.linear2", bits)
        ln("norm_self_att")
        qlin(np.concatenate([g(p + f"self_attn.linear_{n}.weight") for n in "qkv"]),
             np.concatenate([g(p + f"self_attn.linear_{n}.bias") for n in "qkv"]), bits)
        lin("self_attn.linear_out", bits)
        qlin(g(p + "self_attn.linear_pos.weight"), None, 8)
        f32(g(p + "self_attn.pos_bias_u")); f32(g(p + "self_attn.pos_bias_v"))
        ln("norm_conv")
        qlin(g(p + "conv.pointwise_conv1.weight")[:, :, 0], g(p + "conv.pointwise_conv1.bias"), bits)
        f32(g(p + "conv.depthwise_conv.weight")); f32(g(p + "conv.depthwise_conv.bias"))
        qlin(g(p + "conv.pointwise_conv2.weight")[:, :, 0], g(p + "conv.pointwise_conv2.bias"), bits)
        ln("norm_feed_forward2"); lin("feed_forward2.linear1", bits); lin("feed_forward2.linear2", bits)
        ln("norm_out")
    qlin(g("decoder.decoder_layers.0.weight")[:, :, 0], g("decoder.decoder_layers.0.bias"), 8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tnm")
    ap.add_argument("out_dir")
    ap.add_argument("--verify", action="store_true", help="re-export (int8 models) and compare with the original bytes")
    a = ap.parse_args()
    buf = open(a.tnm, "rb").read()
    sd, toks, extra = parse(buf)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez(os.path.join(a.out_dir, "state_dict_plain.npz"), **sd)
    meta = {k: v for k, v in extra.items() if not k.startswith("norm")}
    n_par = sum(v.size for k, v in sd.items() if not k.startswith("preprocessor"))
    print(f"{a.tnm}: {meta}, {n_par / 1e6:.1f} M parameters -> {a.out_dir}/state_dict_plain.npz")
    if a.verify:
        out = reexport(buf, sd, extra)
        n = len(out)
        same = bytes(out) == buf[:n]
        diff = sum(x != y for x, y in zip(out, buf[:n]))
        print(f"verify: re-exported weights {'IDENTICAL' if same else 'DIFFER'} to the original ({n} bytes compared, {diff} differ)")
    try:
        import torch
    except ImportError:
        print("PyTorch not installed: wrote the .npz only (convert on the training machine: rerun this script there)")
        return
    tsd = {k: torch.from_numpy(v) for k, v in sd.items()}
    torch.save(tsd, os.path.join(a.out_dir, "state_dict_plain.pt"))
    from nemo_small import NemoSmall  # train_nemo_stream.py --init loads init["model"] into NemoSmall
    init = {"model": NemoSmall(tsd).state_dict(), "stream": bool(extra["flags"] & 2)}
    if "norm_mean" in extra:
        init["norm_mean"], init["norm_std"] = torch.from_numpy(extra["norm_mean"]), torch.from_numpy(extra["norm_std"])
    torch.save(init, os.path.join(a.out_dir, "init.pt"))
    print("wrote state_dict_plain.pt and init.pt; copy the model's tokenizer.model into", a.out_dir)


if __name__ == "__main__":
    main()
