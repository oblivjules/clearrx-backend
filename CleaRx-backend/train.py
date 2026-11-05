"""
train.py

Fine-tunes microsoft/trocr-base-handwritten on the CleaRx prescription dataset.

Expected layout (output of dataset_builder.py):
  dataset_clean/
  ├── train/
  │   ├── images/
  │   └── labels.csv   (columns: file_name, text)
  ├── val/
  │   ├── images/
  │   └── labels.csv
  └── test/
      ├── images/
      └── labels.csv

Usage:
    python train.py --data dataset_clean [--epochs 20] [--batch-size 8]
                    [--lr 5e-5] [--encoder-lr 1e-5] [--decoder-lr 5e-5]
                    [--output runs/trocr-rx] [--resume runs/trocr-rx/checkpoint-epoch-5]
                    [--unfreeze-epoch 2] [--early-stopping-patience 3]
                    [--gradient-checkpointing] [--compile] [--freeze-encoder]
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

MODEL_ID = "microsoft/trocr-base-handwritten"


# ── CER (no external dependency) ──────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = (
                prev[j - 1]
                if a[i - 1] == b[j - 1]
                else 1 + min(prev[j], dp[j - 1], prev[j - 1])
            )
    return dp[n]


def character_error_rate(refs: list[str], hyps: list[str]) -> float:
    total_chars = sum(len(r) for r in refs)
    total_edits = sum(_levenshtein(r, h) for r, h in zip(refs, hyps))
    return total_edits / max(total_chars, 1)


# ── Augmentation (train only) ─────────────────────────────────────────────────

class _CLAHEBilateral:
    """CLAHE + bilateral filter on PIL image, matching inference preprocessing."""
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=20, sigmaSpace=20)
        return Image.fromarray(arr).convert("RGB")


def build_train_augmentation() -> transforms.Compose:
    return transforms.Compose([
        # CLAHE + bilateral filter with p=0.4 aligns training with inference preprocessing
        transforms.RandomApply([_CLAHEBilateral()], p=0.4),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
            p=0.3,
        ),
        # Brightness/contrast jitter simulates uneven phone lighting and shadow
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.4)],
            p=0.4,
        ),
        transforms.RandomAffine(
            degrees=8,               # real prescriptions are often tilted more than ±2°
            translate=(0.05, 0.05),
            scale=(0.90, 1.10),      # wider zoom range for phone capture variation
            shear=5,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=255,
        ),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class PrescriptionDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        augment: bool = False,
    ) -> None:
        self.image_dir         = split_dir / "images"
        self.processor         = processor
        self.max_target_length = max_target_length
        self.augment           = augment
        self.transform         = build_train_augmentation() if augment else None
        self.samples: list[dict[str, str]] = []

        csv_path = split_dir / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"labels.csv not found in {split_dir}")

        # utf-8-sig handles BOM from Excel-exported CSVs
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = row["file_name"].strip()
                text  = row["text"].strip()
                if fname and text:
                    self.samples.append({"file_name": fname, "text": text})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image  = Image.open(self.image_dir / sample["file_name"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        encoding = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = encoding.input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Encoder freeze / unfreeze helpers ─────────────────────────────────────────

def freeze_encoder(model: VisionEncoderDecoderModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: VisionEncoderDecoderModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True


# ── Build optimizer with differential LRs ────────────────────────────────────

def build_optimizer(
    model: VisionEncoderDecoderModel,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    encoder_params = list(model.encoder.named_parameters())
    decoder_params = list(model.decoder.named_parameters())

    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    param_groups = [
        {
            "params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in decoder_params if not any(nd in n for nd in no_decay)],
            "lr": decoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in decoder_params if any(nd in n for nd in no_decay)],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(param_groups)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    num_beams: int = 1,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_refs: list[str] = []
    all_hyps: list[str] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss = model(pixel_values=pixel_values, labels=labels).loss
        total_loss += loss.item()

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=num_beams,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        hyps = processor.batch_decode(generated_ids, skip_special_tokens=True)

        ref_ids = labels.clone()
        ref_ids[ref_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(ref_ids, skip_special_tokens=True)

        all_hyps.extend(hyps)
        all_refs.extend(refs)

    return total_loss / len(loader), character_error_rate(all_refs, all_hyps)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "best", "timestamp"])

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"[INFO] Device      : {device}")
    if amp_enabled:
        print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] AMP         : enabled")

    # Resolve LRs: --encoder-lr / --decoder-lr take priority; fall back to --lr
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.lr
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr * 0.2

    print(f"[INFO] Encoder LR  : {encoder_lr:.2e}")
    print(f"[INFO] Decoder LR  : {decoder_lr:.2e}")

    # ── Load model ─────────────────────────────────────────────────────────────
    resume_path = Path(args.resume) if args.resume else None
    source      = str(resume_path) if resume_path else MODEL_ID
    print(f"[INFO] Loading from: {source}")

    processor = TrOCRProcessor.from_pretrained(source)
    model     = VisionEncoderDecoderModel.from_pretrained(source).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    model.generation_config.max_new_tokens       = args.max_target_length
    model.generation_config.no_repeat_ngram_size = 0    # drug names legitimately repeat tokens
    model.generation_config.length_penalty       = 1.0  # neutral — prevents over-generation
    model.generation_config.num_beams            = 4

    # Gradient checkpointing reduces VRAM at slight speed cost
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing: enabled")

    # torch.compile (PyTorch 2.0+)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile: enabled")
        except Exception as e:
            print(f"[WARN] torch.compile not available ({e}), skipping")

    # Initial freeze state
    encoder_frozen = args.freeze_encoder or (args.unfreeze_epoch > 0)
    if encoder_frozen:
        freeze_encoder(model)
        reason = "--freeze-encoder" if args.freeze_encoder else f"gradual unfreeze at epoch {args.unfreeze_epoch}"
        print(f"[INFO] Encoder frozen ({reason})")
    else:
        print("[INFO] Encoder unfrozen — training encoder + decoder")

    start_epoch = 0
    if resume_path and (resume_path / "trainer_state.txt").exists():
        with open(resume_path / "trainer_state.txt") as f:
            start_epoch = int(f.read().strip()) + 1
        print(f"[INFO] Resuming from epoch {start_epoch}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = PrescriptionDataset(
        data_dir / "train", processor, args.max_target_length, augment=True
    )
    val_ds = PrescriptionDataset(
        data_dir / "val", processor, args.max_target_length, augment=False
    )

    persistent = args.workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=amp_enabled,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=amp_enabled,
        persistent_workers=persistent,
    )

    print(f"[INFO] Train       : {len(train_ds)} samples")
    print(f"[INFO] Val         : {len(val_ds)} samples")
    print()

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer    = build_optimizer(model, encoder_lr, decoder_lr)
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── Loop ───────────────────────────────────────────────────────────────────
    best_cer          = args.initial_best_cer
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):

        # Gradual unfreeze: turn on encoder after N warm-up epochs
        if (
            not args.freeze_encoder
            and encoder_frozen
            and epoch >= args.unfreeze_epoch
        ):
            unfreeze_encoder(model)
            encoder_frozen = False
            print(f"[INFO] Encoder unfrozen at epoch {epoch + 1}")
            # Encoder params were already in optimizer groups from build_optimizer()
            # but had requires_grad=False, so no Adam state was accumulated for them
            # (correct: they start with fresh state).  Decoder Adam state is preserved
            # by NOT rebuilding the optimizer.  Only the scheduler needs rebuilding.
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                0,
                (len(train_loader) // args.grad_accum) * (args.epochs - epoch),
            )

        model.train()
        optimizer.zero_grad()
        epoch_loss  = 0.0
        accum_steps = 0

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = model(pixel_values=pixel_values, labels=labels).loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Only step scheduler when optimizer actually stepped (no grad overflow)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad()
                accum_steps += 1

                if accum_steps % args.log_steps == 0:
                    avg = epoch_loss / step
                    lrs = scheduler.get_last_lr()
                    enc_lr_str = f"{lrs[0]:.2e}" if len(lrs) > 0 else "n/a"
                    dec_lr_str = f"{lrs[2]:.2e}" if len(lrs) > 2 else f"{lrs[0]:.2e}"
                    print(
                        f"  Epoch {epoch + 1}/{args.epochs}  "
                        f"Step {step}/{len(train_loader)}  "
                        f"loss={avg:.4f}  "
                        f"enc_lr={enc_lr_str}  "
                        f"dec_lr={dec_lr_str}"
                    )

        # Flush any remaining gradients from a partial accumulation batch
        if len(train_loader) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            optimizer.zero_grad()
            accum_steps += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        val_loss, val_cer = evaluate(model, processor, val_loader, device, amp_enabled, num_beams=args.val_beams)
        improved = val_cer < best_cer
        print(
            f"[Epoch {epoch + 1:>3}/{args.epochs}]  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_CER={val_cer:.4f}"
            + (" *" if improved else "")
        )

        # ── Save best / early stopping ─────────────────────────────────────────
        if improved:
            best_cer          = val_cer
            epochs_no_improve = 0
            best_dir          = out_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  [BEST] CER={best_cer:.4f} → {best_dir}")
        else:
            epochs_no_improve += 1
            print(f"  [NO IMPROVE] patience {epochs_no_improve}/{args.early_stopping_patience}")
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"[STOP] Early stopping triggered after {epoch + 1} epochs.")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{val_cer:.4f}", improved, time.strftime("%Y-%m-%d %H:%M:%S")])
                break

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{val_cer:.4f}", improved, time.strftime("%Y-%m-%d %H:%M:%S")])

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_dir = out_dir / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            with open(ckpt_dir / "trainer_state.txt", "w") as f:
                f.write(str(epoch))
            print(f"  [CKPT] Saved → {ckpt_dir}")

    print(f"\n[DONE] Best val CER : {best_cer:.4f}")
    print(f"       Best model   : {out_dir / 'best'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune TrOCR on the CleaRx prescription handwriting dataset."
    )
    # ── Existing args (unchanged) ──────────────────────────────────────────────
    p.add_argument("--data",              default="dataset_synth",
                   help="Path to dataset directory (default: dataset_synth)")
    p.add_argument("--output",            default="runs/trocr-rx",
                   help="Directory for checkpoints (default: runs/trocr-rx)")
    p.add_argument("--resume",            default=None,
                   help="Checkpoint directory to resume from")
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=5e-5,
                   help="Decoder peak LR; encoder gets 0.2x unless --encoder-lr is set")
    p.add_argument("--warmup-ratio",      type=float, default=0.1,
                   help="Fraction of total steps for LR warmup (default: 0.1)")
    p.add_argument("--max-target-length", type=int,   default=128)
    p.add_argument("--max-grad-norm",     type=float, default=1.0)
    p.add_argument("--grad-accum",        type=int,   default=1)
    p.add_argument("--save-every",        type=int,   default=5)
    p.add_argument("--log-steps",         type=int,   default=10)
    p.add_argument("--workers",           type=int,   default=2)
    # ── New args ───────────────────────────────────────────────────────────────
    p.add_argument("--encoder-lr",        type=float, default=None,
                   help="Encoder peak LR (default: --lr * 0.2)")
    p.add_argument("--decoder-lr",        type=float, default=None,
                   help="Decoder peak LR (default: --lr)")
    p.add_argument("--freeze-encoder",    action="store_true",
                   help="Keep encoder frozen for all epochs (old behaviour)")
    p.add_argument("--unfreeze-epoch",    type=int,   default=2,
                   help="Unfreeze encoder after this many epochs (default: 2, 0=never freeze)")
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="Stop if val CER does not improve for N epochs (default: 5)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable encoder gradient checkpointing to save VRAM")
    p.add_argument("--compile",           action="store_true",
                   help="Enable torch.compile (PyTorch 2.0+)")
    p.add_argument("--val-beams",         type=int,   default=1,
                   help="Beam width during validation (1=greedy, faster; default: 1)")
    p.add_argument("--initial-best-cer",  type=float, default=float("inf"),
                   help="Starting best CER threshold — best/ is only overwritten if val CER beats this. "
                        "Pass your current best CER when resuming so the saved model is never replaced "
                        "by a worse one (default: inf, meaning first epoch always saves).")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
