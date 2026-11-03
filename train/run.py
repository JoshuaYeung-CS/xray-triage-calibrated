"""Train, select and export the pneumonia triage model. Runs as one short GPU job.

1. Frozen ImageNet ResNet-18 features + L2 logistic regression (baseline).
2. Full fine-tune of ResNet-18, three seeds, best epoch by validation NLL.
3. Pick the candidate with the lowest temperature-scaled validation NLL.
4. Export it to ONNX, check ONNX against PyTorch on every test image, time CPU inference.

Only validation numbers are printed. Test logits are saved for the evaluation script, which
computes every test metric once (scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "train"))

from triage.calibration import auroc, fit_temperature, nll  # noqa: E402
from resnet import TriageNet  # noqa: E402


def use_bf16(x: torch.Tensor) -> bool:
    """bf16 autocast only on Ampere or newer; older cards (2080 Ti, Titan RTX) train in fp32."""
    return x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 8


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_split(z, name: str, device, limit: int | None):
    x = torch.from_numpy(z[f"{name}_images"].astype(np.float32) / 255.0)[:, None]
    y = torch.from_numpy(z[f"{name}_labels"].reshape(-1).astype(np.float32))
    if limit:
        x, y = x[:limit], y[:limit]
    return x.to(device), y.to(device)


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Small random affine (rotate +-10 deg, scale 0.9-1.1, shift +-6%) and brightness/contrast.
    No horizontal flips: the heart sits on the left, so a flipped film is not a real film."""
    n = x.shape[0]
    dev = x.device

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=g, device="cpu").to(dev)

    ang = u(-10, 10) * math.pi / 180
    sc = u(0.9, 1.1)
    tx, ty = u(-0.06, 0.06), u(-0.06, 0.06)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = u(0.85, 1.15).view(n, 1, 1, 1)
    b = u(-0.08, 0.08).view(n, 1, 1, 1)
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


@torch.no_grad()
def predict(model, x, bs: int = 256, with_cam: bool = False):
    model.eval()
    logits, cams = [], []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16(x)):
            lg, cam = model(x[i : i + bs])
        logits.append(lg.float().cpu())
        if with_cam:
            cams.append(cam.float().cpu())
    out = torch.cat(logits).numpy().astype(np.float64)
    return (out, torch.cat(cams).numpy()) if with_cam else out


@torch.no_grad()
def pooled_features(model, x, bs: int = 256):
    model.eval()
    feats = []
    for i in range(0, len(x), bs):
        feats.append(model.features(x[i : i + bs]).mean(dim=(2, 3)).float())
    return torch.cat(feats)


def logistic_baseline(model, data, lambdas=(1e-4, 1e-3, 1e-2, 1e-1)):
    (xtr, ytr), (xva, yva), (xte, _) = data
    ftr, fva, fte = (pooled_features(model, x) for x in (xtr, xva, xte))
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva, fte = ((f - mu) / sd for f in (ftr, fva, fte))
    best = None
    for lam in lambdas:
        w = torch.zeros(ftr.shape[1], device=ftr.device, requires_grad=True)
        b = torch.zeros(1, device=ftr.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(ftr @ w + b, ytr) + lam * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            va = (fva @ w + b).double().cpu().numpy()
            te = (fte @ w + b).double().cpu().numpy()
        v = nll(va, yva.cpu().numpy())
        log(f"  logistic lambda={lam:g}  val NLL={v:.4f}  val AUROC={auroc(va, yva.cpu().numpy()):.4f}")
        if best is None or v < best["val_nll"]:
            best = {"lambda": lam, "val_nll": v, "val": va, "test": te}
    return best


def finetune(seed: int, args, data, device):
    (xtr, ytr), (xva, yva), _ = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = TriageNet().to(device)
    model.trunk.load_imagenet(args.weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(xtr) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    )
    yva_np = yva.cpu().numpy()
    best, history = None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        run_loss = 0.0
        for i in range(0, len(xtr), args.batch):
            idx = perm[i : i + args.batch]
            xb = augment(xtr[idx], g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16(xb)):
                logit, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit.float(), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += float(loss) * len(idx)
        va = predict(model, xva)
        rec = {"epoch": ep + 1, "train_loss": run_loss / len(xtr), "val_nll": nll(va, yva_np), "val_auroc": auroc(va, yva_np)}
        history.append(rec)
        log(f"  seed {seed} epoch {ep + 1:2d}  train {rec['train_loss']:.4f}  val NLL {rec['val_nll']:.4f}  val AUROC {rec['val_auroc']:.4f}")
        if best is None or rec["val_nll"] < best["val_nll"]:
            best = {"epoch": ep + 1, "val_nll": rec["val_nll"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, best, history, time.time() - t0


def run_onnx(sess, x: np.ndarray, bs: int = 64):
    logits, cams = [], []
    for i in range(0, len(x), bs):
        lg, cam = sess.run(None, {"image": x[i : i + bs]})
        logits.append(lg)
        cams.append(cam)
    return np.concatenate(logits), np.concatenate(cams)


def export_onnx(model, path: Path, x_val: np.ndarray, x_test: np.ndarray):
    """Export on CPU in float32, then check onnxruntime against PyTorch on every test image.
    Validation and test logits for the shipped model come from the ONNX file the app serves."""
    import onnxruntime as ort

    model = model.float().cpu().eval()
    dummy = torch.from_numpy(x_test[:2])
    kwargs = dict(
        input_names=["image"],
        output_names=["logit", "cam"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
        exporter = "torchscript"
    except Exception as e:  # newer torch may only ship the dynamo exporter
        log(f"  torchscript exporter failed ({type(e).__name__}: {e}); trying dynamo")
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, **kwargs)
        exporter = "dynamo"
    full = ort.SessionOptions()
    full.intra_op_num_threads = 4  # explicit count: the default tries to pin threads and fails in Slurm cgroups
    sess = ort.InferenceSession(str(path), full, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_logit, ref_cam = model(torch.from_numpy(x_test))
    o_logit, o_cam = run_onnx(sess, x_test)
    v_logit, _ = run_onnx(sess, x_val)
    parity = {
        "exporter": exporter,
        "max_abs_logit_diff": float(np.max(np.abs(o_logit - ref_logit.numpy()))),
        "max_abs_cam_diff": float(np.max(np.abs(o_cam - ref_cam.numpy()))),
        "same_side_of_zero": float(np.mean(np.sign(o_logit) == np.sign(ref_logit.numpy()))),
    }
    # Single-image CPU latency with the same thread count the app uses.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    s2 = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    one = x_test[:1]
    for _ in range(5):
        s2.run(None, {"image": one})
    times = []
    for i in range(50):
        t = time.perf_counter()
        s2.run(None, {"image": x_test[i : i + 1]})
        times.append(time.perf_counter() - t)
    parity["cpu_latency_ms_median"] = 1000 * float(np.median(times))
    parity["cpu_latency_ms_p90"] = 1000 * float(np.percentile(times, 90))
    parity["onnxruntime"] = ort.__version__
    return v_logit.astype(np.float64), o_logit.astype(np.float64), o_cam, parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, 1 seed")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} torch={torch.__version__}")
    torch.backends.cudnn.benchmark = True

    z = np.load(args.data)
    limit = 256 if args.smoke else None
    data = tuple(load_split(z, s, device, limit) for s in ("train", "val", "test"))
    (_, yva), (xte, yte) = data[1], data[2]
    yva_np, yte_np = yva.cpu().numpy(), yte.cpu().numpy()
    log(f"train {len(data[0][0])}  val {len(yva_np)}  test {len(yte_np)}")

    record = {"args": vars(args), "python": platform.python_version(), "torch": torch.__version__, "candidates": {}}
    preds = {"val_labels": yva_np, "test_labels": yte_np}

    log("baseline: frozen ImageNet features + logistic regression")
    frozen = TriageNet().to(device)
    frozen.trunk.load_imagenet(args.weights)
    lr = logistic_baseline(frozen, data)
    del frozen
    preds["val_logits_frozen_lr"], preds["test_logits_frozen_lr"] = lr["val"], lr["test"]
    t = fit_temperature(lr["val"], yva_np)
    record["candidates"]["frozen_lr"] = {"lambda": lr["lambda"], "val_nll": lr["val_nll"], "val_nll_scaled": nll(lr["val"], yva_np, t), "val_auroc": auroc(lr["val"], yva_np)}

    best_name, best_model, best_scaled = None, None, None
    for seed in args.seeds:
        log(f"fine-tune seed {seed}")
        model, best, history, secs = finetune(seed, args, data, device)
        va = predict(model, data[1][0])
        te = predict(model, xte)
        name = f"finetune_s{seed}"
        preds[f"val_logits_{name}"], preds[f"test_logits_{name}"] = va, te
        t = fit_temperature(va, yva_np)
        scaled = nll(va, yva_np, t)
        record["candidates"][name] = {
            "best_epoch": best["epoch"], "val_nll": best["val_nll"], "val_nll_scaled": scaled,
            "val_auroc": auroc(va, yva_np), "train_seconds": secs, "history": history,
        }
        log(f"  seed {seed}: best epoch {best['epoch']}, val NLL {best['val_nll']:.4f} -> {scaled:.4f} after T={t:.3f}, {secs:.0f}s")
        if best_scaled is None or scaled < best_scaled:
            best_name, best_model, best_scaled = name, model, scaled
        else:
            del model
        torch.cuda.empty_cache()

    # The deployed model must be a fine-tuned network (the app needs its heatmap), so selection is
    # among fine-tune seeds; the frozen baseline is reported alongside for comparison.
    record["shipped"] = best_name
    log(f"shipping {best_name} (lowest temperature-scaled val NLL among fine-tune seeds)")
    torch.save(best_model.state_dict(), out / "triage_state.pt")

    x_test_np = xte.cpu().numpy().astype(np.float32)
    x_val_np = data[1][0].cpu().numpy().astype(np.float32)
    v_logit, o_logit, o_cam, parity = export_onnx(best_model, out / "triage.onnx", x_val_np, x_test_np)
    record["onnx"] = parity
    log(f"onnx parity: {json.dumps(parity)}")
    preds["val_logits_onnx"] = v_logit
    preds["test_logits_onnx"] = o_logit
    preds["test_cam_onnx"] = o_cam.astype(np.float32)
    np.savez_compressed(out / "predictions.npz", **preds)
    (out / "train_record.json").write_text(json.dumps(record, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main()
