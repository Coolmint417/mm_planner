from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PROJECT_PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import cfg as default_cfg
from dataloader import build_dataloader
from models.planner import MultimodalPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior cloning training for the mm_planner model."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bc"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--vision-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--mock-vision",
        action="store_true",
        help="Use the lightweight local mock vision backbone instead of torch.hub DINOv2.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--normal-flight-only-continuous-loss",
        action="store_true",
        help="Apply waypoint/velocity/acceleration losses only when target_mode_id == 0. "
        "By default these losses are applied to all modes because every clip has future trajectory supervision.",
    )
    return parser.parse_args()


def gaussian_nll(
    pred: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mu = pred["mu"]
    sigma = pred["sigma"].clamp_min(1e-6)
    nll = 0.5 * ((target - mu) / sigma).pow(2) + sigma.log()
    nll = nll.sum(dim=(-1, -2))
    if mask is not None:
        if mask.any():
            nll = nll[mask]
        else:
            return mu.new_zeros(())
    return nll.mean()


def trajectory_smoothness_loss(waypoints: torch.Tensor) -> torch.Tensor:
    if waypoints.size(1) < 3:
        return waypoints.new_zeros(())
    velocity = waypoints[:, 1:] - waypoints[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return acceleration.pow(2).mean()


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    moved: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [item.to(device, non_blocking=True) for item in value]
        else:
            moved[key] = value.to(device, non_blocking=True)
    return moved


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
    vision_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    vision_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_encoder."):
            vision_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": learning_rate})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": vision_learning_rate})

    return AdamW(param_groups, weight_decay=weight_decay)


def compute_bc_loss(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    training_cfg: Any,
    normal_flight_only_continuous_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    mode_loss = F.cross_entropy(outputs["mode_logits"], batch["target_mode_id"])

    continuous_mask = None
    if normal_flight_only_continuous_loss:
        # Current collected clips use mode ids 0 normal, 1 takeoff, 2 landing.
        # Keep this as a conservative hook for future semantic FLY-only masks.
        continuous_mask = batch["target_mode_id"] == 0

    waypoint_loss = gaussian_nll(
        outputs["flight_waypoints"],
        batch["teacher_flight_waypoints"],
        continuous_mask,
    )
    velocity_delta_loss = gaussian_nll(
        outputs["flight_velocity_deltas"],
        batch["teacher_flight_velocity_deltas"],
        continuous_mask,
    )
    acceleration_delta_loss = gaussian_nll(
        outputs["flight_acceleration_deltas"],
        batch["teacher_flight_acceleration_deltas"],
        continuous_mask,
    )

    smooth_loss = trajectory_smoothness_loss(outputs["flight_waypoints"]["mu"])

    crawl_loss = outputs["mode_logits"].new_zeros(())
    if "teacher_crawl_action" in batch:
        crawl_mask = batch["target_mode_id"] == 1
        if crawl_mask.any():
            crawl_loss = F.smooth_l1_loss(
                outputs["crawl_action"][crawl_mask],
                batch["teacher_crawl_action"][crawl_mask],
            )

    total = (
        training_cfg.lambda_mode * mode_loss
        + training_cfg.lambda_waypoint * waypoint_loss
        + training_cfg.lambda_velocity_delta * velocity_delta_loss
        + training_cfg.lambda_acceleration_delta * acceleration_delta_loss
        + training_cfg.lambda_crawl * crawl_loss
        + training_cfg.lambda_smooth * smooth_loss
    )

    metrics = {
        "loss": float(total.detach().cpu()),
        "mode": float(mode_loss.detach().cpu()),
        "waypoint": float(waypoint_loss.detach().cpu()),
        "velocity_delta": float(velocity_delta_loss.detach().cpu()),
        "acceleration_delta": float(acceleration_delta_loss.detach().cpu()),
        "crawl": float(crawl_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
    }
    return total, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    cfg: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = copy.deepcopy(default_cfg)

    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.samples_per_epoch is not None:
        cfg.data.samples_per_epoch = args.samples_per_epoch
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.vision_learning_rate is not None:
        cfg.training.vision_learning_rate = args.vision_learning_rate
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.teacher_forcing_ratio is not None:
        cfg.training.teacher_forcing_ratio = args.teacher_forcing_ratio
    if args.mock_vision:
        cfg.model.allow_mock_vision_encoder = True

    device_name = args.device or cfg.model.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    torch.manual_seed(cfg.model.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.model.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    loader = build_dataloader(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=cfg.model.seed,
    )

    model = MultimodalPlanner(cfg.model).to(device)
    optimizer = make_optimizer(
        model=model,
        learning_rate=cfg.training.learning_rate,
        vision_learning_rate=cfg.training.vision_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    steps_per_epoch = len(loader)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        running: dict[str, float] = {}

        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(
                    rgb=batch["rgb"],
                    traj_mode_ids=batch["traj_mode_ids"],
                    traj_continuous=batch["traj_continuous"],
                    task_waypoints=batch["task_waypoints"],
                    teacher_flight_waypoints=batch["teacher_flight_waypoints"],
                    teacher_flight_velocity_deltas=batch[
                        "teacher_flight_velocity_deltas"
                    ],
                    teacher_flight_acceleration_deltas=batch[
                        "teacher_flight_acceleration_deltas"
                    ],
                    teacher_forcing_ratio=cfg.training.teacher_forcing_ratio,
                )
                loss, metrics = compute_bc_loss(
                    outputs=outputs,
                    batch=batch,
                    training_cfg=cfg.training,
                    normal_flight_only_continuous_loss=(
                        args.normal_flight_only_continuous_loss
                    ),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if step % args.log_interval == 0 or step == steps_per_epoch:
                averaged = {
                    key: value / step
                    for key, value in running.items()
                }
                metric_text = " ".join(
                    f"{key}={value:.4f}" for key, value in averaged.items()
                )
                print(
                    f"epoch={epoch + 1}/{cfg.training.num_epochs} "
                    f"step={step}/{steps_per_epoch} {metric_text}",
                    flush=True,
                )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
            )
        save_checkpoint(
            args.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
