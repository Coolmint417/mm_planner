# Instruction File: Route-Conditioned Multimodal Flying-Crawling Drone Planner

## 1. Project Goal

Please implement a PyTorch project for a route-conditioned multimodal planner for a flying-crawling drone.

The model receives:

1. RGB image observations from an onboard camera.
2. Historical self-motion trajectory information over the past `n_traj_encoder` steps.
3. A sequence of global task waypoints that the drone must visit in order.

The model outputs:

1. A discrete motion mode prediction.
2. A future flight trajectory generated autoregressively.
3. A crawling action prediction.
4. A transition action prediction for landing, takeoff, attachment, detachment, etc.

The project should be modular, configurable, and easy to extend for imitation learning and later reinforcement learning.

---

## 2. Required Project Structure

Please create the following structure:

```text
mm_planner/
│
├── config.py
├── main_test.py
│
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py
│   ├── trajectory_encoder.py
│   ├── waypoint_encoder.py
│   ├── fusion_transformer.py
│   ├── heads.py
│   └── planner.py
│
├── utils/
│   ├── __init__.py
│   └── tensor_utils.py
│
└── README.md
```

The implementation should be clean, type-annotated where reasonable, and contain shape comments in the forward functions.

---

## 3. Global Configuration File

Create a `config.py` file containing a dataclass-style configuration object.

The config should allow convenient adjustment of all important hyperparameters.

Suggested content:

```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # -------------------------
    # General
    # -------------------------
    device: str = "cuda"
    seed: int = 42

    # -------------------------
    # Vision encoder
    # -------------------------
    vision_model_name: str = "dinov2_vitb14"
    vision_output_dim: int = 256
    freeze_vision_encoder: bool = True
    use_patch_tokens: bool = True
    image_size: int = 224

    # DINOv2 ViT-B/14 feature dimension is usually 768.
    dinov2_feature_dim: int = 768

    # -------------------------
    # Historical trajectory encoder
    # -------------------------
    n_traj_encoder: int = 8

    num_motion_modes: int = 7

    # Per historical point:
    # motion mode one-hot or embedding index
    # position
    # velocity
    # acceleration
    # attitude
    # angular velocity
    #
    # Recommended raw dimensions:
    # mode_id: 1 if using embedding, or num_motion_modes if one-hot
    # position: 3
    # velocity: 3
    # acceleration: 3
    # attitude quaternion: 4
    # angular velocity: 3
    #
    # If using mode embedding, continuous state dim = 3 + 3 + 3 + 4 + 3 = 16
    traj_continuous_dim: int = 16
    mode_embedding_dim: int = 16
    traj_token_dim: int = 256
    traj_encoder_layers: int = 2
    traj_encoder_heads: int = 4
    traj_encoder_ff_dim: int = 512
    traj_dropout: float = 0.1

    # -------------------------
    # Task waypoint encoder
    # -------------------------
    n_waypoints: int = 5

    # Each task waypoint feature:
    # relative position in body frame: 3
    # relative yaw or heading: 1
    # distance to waypoint: 1
    # is_current_goal flag: 1
    # goal_type id or scalar: 1
    waypoint_dim: int = 7

    waypoint_token_dim: int = 256
    waypoint_encoder_layers: int = 2
    waypoint_encoder_heads: int = 4
    waypoint_encoder_ff_dim: int = 512
    waypoint_dropout: float = 0.1

    # -------------------------
    # Fusion transformer
    # -------------------------
    fusion_dim: int = 256
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_ff_dim: int = 1024
    fusion_dropout: float = 0.1

    # -------------------------
    # Prediction heads
    # -------------------------
    hidden_dim: int = 256

    # Motion modes:
    # 0 FLY
    # 1 CRAWL
    # 2 LAND
    # 3 TAKEOFF
    # 4 ATTACH
    # 5 DETACH
    # 6 HOVER
    num_modes: int = 7

    # Flight trajectory prediction
    pred_num_flight_waypoints: int = 10
    pred_waypoint_dim: int = 4  # [dx_body, dy_body, dz_body, dyaw]
    autoregressive_hidden_dim: int = 256
    autoregressive_predict_delta: bool = True
    max_waypoint_delta: float = 1.0

    # Crawling action
    crawl_action_dim: int = 3  # [vx_surface, vy_surface, yaw_rate_surface]
    max_crawl_vx: float = 0.2
    max_crawl_vy: float = 0.2
    max_crawl_yaw_rate: float = 0.5

    # Transition action
    # contact_pos: 3
    # surface_normal: 3
    # yaw: 1
    # approach_speed: 1
    transition_dim: int = 8
    max_approach_speed: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    vision_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    num_epochs: int = 100

    lambda_mode: float = 1.0
    lambda_waypoint: float = 1.0
    lambda_crawl: float = 1.0
    lambda_transition: float = 1.0
    lambda_smooth: float = 0.1

    teacher_forcing_ratio: float = 0.5


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


cfg = Config()
```

Please fix any dataclass mutable default issues if necessary by using `field(default_factory=...)`.

---

## 4. Input Tensor Definitions

The main model should accept a dictionary or explicit arguments. Prefer explicit arguments.

### 4.1 RGB Image

```python
rgb: Tensor
shape: [B, 3, H, W]
```

Use DINOv2 ViT-B/14 as the vision encoder.

Default input size:

```text
224 × 224
```

### 4.2 Historical Motion Trajectory

The trajectory encoder receives the past `n_traj_encoder` states.

Use:

```python
traj_mode_ids: LongTensor
shape: [B, n_traj_encoder]
```

Each value is a discrete mode id:

```text
0 FLY
1 CRAWL
2 LAND
3 TAKEOFF
4 ATTACH
5 DETACH
6 HOVER
```

And:

```python
traj_continuous: Tensor
shape: [B, n_traj_encoder, traj_continuous_dim]
```

Where each historical point contains:

```text
position:         3
velocity:         3
acceleration:     3
attitude quat:    4
angular velocity: 3
```

So:

```text
traj_continuous_dim = 16
```

### 4.3 Task Waypoints

The drone receives a sequence of future task waypoints that it must pass in order.

Use:

```python
task_waypoints: Tensor
shape: [B, n_waypoints, waypoint_dim]
```

Each waypoint should be represented relative to the current drone body frame.

Recommended waypoint feature:

```text
dx_body:          1
dy_body:          1
dz_body:          1
relative_yaw:     1
distance:         1
is_current_goal:  1
goal_type:        1
```

So:

```text
waypoint_dim = 7
```

Optional mask:

```python
task_waypoint_mask: BoolTensor | None
shape: [B, n_waypoints]
```

`True` means padding and should be ignored by the transformer.

---

## 5. Visual Encoder Requirements

Implement `models/vision_encoder.py`.

Class name:

```python
class DINOv2VisionEncoder(nn.Module):
    ...
```

Requirements:

1. Load DINOv2 ViT-B/14 using:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```

2. Support frozen and trainable modes using `freeze_vision_encoder`.

3. Support both:

   * global feature output
   * patch-token attention pooling

4. DINOv2 ViT-B/14 feature dimension should be configurable, default `768`.

5. Output shape must be:

```python
z_img: Tensor
shape: [B, vision_output_dim]
```

Suggested implementation behavior:

```python
features = backbone.forward_features(rgb)
patch_tokens = features["x_norm_patchtokens"]  # [B, N, C]
cls_token = features["x_norm_clstoken"]        # [B, C]
```

If `use_patch_tokens=True`, apply attention pooling over patch tokens:

```text
patch_tokens → Linear(C, 1) → softmax over patches → weighted sum
```

Then project to `vision_output_dim`.

If `use_patch_tokens=False`, use the CLS token or global output.

Add projection:

```text
Linear(dinov2_feature_dim, vision_output_dim)
ReLU
LayerNorm
```

Important: if frozen, use `torch.no_grad()` around the DINOv2 forward pass.

---

## 6. Historical Trajectory Encoder Requirements

Implement `models/trajectory_encoder.py`.

Class name:

```python
class HistoricalTrajectoryEncoder(nn.Module):
    ...
```

Input:

```python
traj_mode_ids: [B, T]
traj_continuous: [B, T, traj_continuous_dim]
```

Where:

```text
T = n_traj_encoder
```

Architecture:

1. Use an embedding layer for mode ids:

```python
nn.Embedding(num_motion_modes, mode_embedding_dim)
```

2. Concatenate mode embedding with continuous trajectory features:

```text
[mode_embedding, position, velocity, acceleration, attitude, angular_velocity]
```

3. Project each time step to `traj_token_dim`.

4. Add learnable positional embedding of shape:

```python
[1, n_traj_encoder, traj_token_dim]
```

5. Encode the sequence using `nn.TransformerEncoder`.

6. Pool the sequence to a single trajectory feature.

Use either:

* mean pooling
* attention pooling

Prefer attention pooling.

Output:

```python
z_traj: Tensor
shape: [B, traj_token_dim]
```

---

## 7. Task Waypoint Encoder Requirements

Implement `models/waypoint_encoder.py`.

Class name:

```python
class TaskWaypointEncoder(nn.Module):
    ...
```

Input:

```python
task_waypoints: [B, M, waypoint_dim]
task_waypoint_mask: [B, M] or None
```

Where:

```text
M = n_waypoints
```

Architecture:

1. Project waypoint features to `waypoint_token_dim`.
2. Add learnable positional embedding of shape:

```python
[1, n_waypoints, waypoint_token_dim]
```

3. Encode with `nn.TransformerEncoder`.
4. Use attention pooling to get a single route feature.

Output:

```python
z_waypoint: Tensor
shape: [B, waypoint_token_dim]
```

When mask is provided, make sure the transformer ignores padding tokens.

---

## 8. Fusion Transformer Requirements

Implement `models/fusion_transformer.py`.

Class name:

```python
class MultimodalFusionTransformer(nn.Module):
    ...
```

Inputs:

```python
z_img:      [B, vision_output_dim]
z_traj:     [B, traj_token_dim]
z_waypoint: [B, waypoint_token_dim]
```

Architecture:

1. Project each modality feature to a common `fusion_dim`.
2. Create modality tokens:

```text
image token
trajectory token
waypoint token
```

3. Add learnable modality embeddings.

4. Pass the 3 tokens through a transformer encoder.

5. Pool or select a learnable `[CLS]` token.

Preferred version:

Use a learnable `cls_token`:

```text
[CLS], image_token, traj_token, waypoint_token
```

Then pass through transformer encoder and use CLS output as fused representation.

Output:

```python
h: Tensor
shape: [B, fusion_dim]
```

This fused representation is passed to prediction heads.

---

## 9. Prediction Heads Requirements

Implement `models/heads.py`.

Need the following heads:

1. `ModePredictionHead`
2. `AutoregressiveWaypointHead`
3. `CrawlActionHead`
4. `TransitionHead`

---

### 9.1 ModePredictionHead

Input:

```python
h: [B, fusion_dim]
```

Output:

```python
mode_logits: [B, num_modes]
```

Architecture:

```text
Linear
ReLU
LayerNorm
Linear to num_modes
```

---

### 9.2 AutoregressiveWaypointHead

This head generates future flight waypoints autoregressively.

Input:

```python
h: [B, fusion_dim]
```

Optional training input:

```python
teacher_waypoints: [B, pred_num_flight_waypoints, pred_waypoint_dim]
teacher_forcing_ratio: float
```

Output:

```python
pred_flight_waypoints: [B, pred_num_flight_waypoints, pred_waypoint_dim]
```

Each waypoint is:

```text
[dx_body, dy_body, dz_body, dyaw]
```

Use `nn.GRUCell`.

Recommended architecture:

```text
start token
previous waypoint
    ↓
Linear waypoint_dim → autoregressive_hidden_dim
    ↓
GRUCell
    ↓
MLP
    ↓
waypoint or waypoint delta
```

Support two modes:

1. Direct waypoint prediction.
2. Delta waypoint prediction.

If `autoregressive_predict_delta=True`, each step predicts:

```text
delta_wp_t
```

Then accumulate:

```python
wp_t = wp_{t-1} + delta_wp_t
```

Optionally bound the delta using:

```python
delta = tanh(delta) * max_waypoint_delta
```

During training, support teacher forcing:

```python
if training and teacher_waypoints is not None:
    with probability teacher_forcing_ratio:
        previous waypoint = teacher waypoint at current step
    otherwise:
        previous waypoint = predicted waypoint
```

Please implement teacher forcing carefully and make sure the tensor shapes remain correct.

---

### 9.3 CrawlActionHead

Input:

```python
h: [B, fusion_dim]
```

Output:

```python
crawl_action: [B, 3]
```

Meaning:

```text
vx_surface
vy_surface
yaw_rate_surface
```

Architecture:

```text
MLP → Tanh
```

Scale outputs by:

```text
max_crawl_vx
max_crawl_vy
max_crawl_yaw_rate
```

---

### 9.4 TransitionHead

Input:

```python
h: [B, fusion_dim]
```

Output a dictionary:

```python
{
    "contact_pos": [B, 3],
    "surface_normal": [B, 3],
    "yaw": [B, 1],
    "approach_speed": [B, 1],
}
```

Architecture:

1. Shared MLP.
2. Separate linear heads.

For `surface_normal`, normalize using:

```python
F.normalize(normal, dim=-1)
```

For `approach_speed`, use:

```python
sigmoid(raw_speed) * max_approach_speed
```

---

## 10. Complete Planner Model

Implement `models/planner.py`.

Class name:

```python
class MultimodalPlanner(nn.Module):
    ...
```

Constructor:

```python
def __init__(self, cfg: ModelConfig):
    ...
```

Forward signature:

```python
def forward(
    self,
    rgb: torch.Tensor,
    traj_mode_ids: torch.Tensor,
    traj_continuous: torch.Tensor,
    task_waypoints: torch.Tensor,
    task_waypoint_mask: torch.Tensor | None = None,
    teacher_flight_waypoints: torch.Tensor | None = None,
    teacher_forcing_ratio: float = 0.0,
) -> dict:
    ...
```

Return:

```python
{
    "mode_logits": mode_logits,
    "flight_waypoints": flight_waypoints,
    "crawl_action": crawl_action,
    "transition": {
        "contact_pos": contact_pos,
        "surface_normal": surface_normal,
        "yaw": yaw,
        "approach_speed": approach_speed,
    },
    "fused_feature": h,
}
```

Expected shapes:

```text
mode_logits:       [B, num_modes]
flight_waypoints:  [B, pred_num_flight_waypoints, pred_waypoint_dim]
crawl_action:      [B, 3]
contact_pos:       [B, 3]
surface_normal:    [B, 3]
yaw:               [B, 1]
approach_speed:    [B, 1]
fused_feature:     [B, fusion_dim]
```

---

## 11. Loss Function Utilities

In `utils/tensor_utils.py`, implement helper functions:

```python
def waypoint_smoothness_loss(waypoints: torch.Tensor) -> torch.Tensor:
    ...
```

Input:

```python
waypoints: [B, N, D]
```

Compute second-order smoothness:

```python
velocity = waypoints[:, 1:] - waypoints[:, :-1]
acceleration = velocity[:, 1:] - velocity[:, :-1]
loss = mean(acceleration ** 2)
```

Also implement:

```python
def count_trainable_parameters(model: nn.Module) -> int:
    ...
```

and:

```python
def count_total_parameters(model: nn.Module) -> int:
    ...
```

---

## 12. Main Test Script

Create `main_test.py` that:

1. Imports config.
2. Instantiates the full planner.
3. Creates dummy tensors.
4. Runs a forward pass.
5. Prints all output shapes.
6. Prints total and trainable parameter counts.

Use dummy tensors:

```python
B = 2
H = W = cfg.model.image_size

rgb: [B, 3, H, W]
traj_mode_ids: [B, n_traj_encoder]
traj_continuous: [B, n_traj_encoder, traj_continuous_dim]
task_waypoints: [B, n_waypoints, waypoint_dim]
task_waypoint_mask: None
teacher_flight_waypoints: [B, pred_num_flight_waypoints, pred_waypoint_dim]
```

Example output should include:

```text
mode_logits shape:
flight_waypoints shape:
crawl_action shape:
transition.contact_pos shape:
transition.surface_normal shape:
transition.yaw shape:
transition.approach_speed shape:
fused_feature shape:
total parameters:
trainable parameters:
```

---

## 13. Coding Style Requirements

Please follow these coding requirements:

1. Use PyTorch.
2. Keep modules independent and easy to replace.
3. Add clear shape comments in every forward method.
4. Avoid hard-coded dimensions; use `cfg`.
5. Make the model runnable with dummy data using `main_test.py`.
6. Use `torch.hub.load` for DINOv2 by default.
7. Make sure DINOv2 can be frozen.
8. The code should support CPU fallback.
9. Keep implementation concise but complete.
10. Avoid implementing training loops for now unless necessary.
11. Do not implement simulation environment yet.
12. Do not implement dataset loading yet.
13. Do not implement reinforcement learning yet.

---

## 14. Important Implementation Notes

### 14.1 DINOv2 Loading

Use:

```python
self.backbone = torch.hub.load(
    "facebookresearch/dinov2",
    cfg.vision_model_name,
)
```

The default model name should be:

```python
"dinov2_vitb14"
```

DINOv2 ViT-B/14 feature dimension should be:

```python
768
```

### 14.2 Patch Tokens

When using patch tokens:

```python
features = self.backbone.forward_features(rgb)
patch_tokens = features["x_norm_patchtokens"]
```

Use attention pooling:

```python
score = self.attn_pool(patch_tokens)
weight = torch.softmax(score, dim=1)
pooled = (patch_tokens * weight).sum(dim=1)
```

### 14.3 Transformer Batch Format

Use:

```python
batch_first=True
```

for all transformer encoder layers.

### 14.4 Masks

For task waypoint masks:

```python
src_key_padding_mask=task_waypoint_mask
```

where `True` means masked / ignored.

### 14.5 Teacher Forcing in Autoregressive Head

Teacher forcing should only be used during training and only if `teacher_waypoints` is not None.

If using delta prediction and teacher forcing, when the teacher waypoint is used as previous waypoint, also update the internal current waypoint accordingly.

---

## 15. README Requirements

Create a `README.md` explaining:

1. Project purpose.
2. Model architecture.
3. Input and output tensor shapes.
4. How to run the dummy test:

```bash
python main_test.py
```

5. How to change hyperparameters in `config.py`.

---

## 16. Expected Final Behavior

After implementation, I should be able to run:

```bash
python main_test.py
```

and see successful forward pass output shapes.

The code should instantiate:

```python
MultimodalPlanner(cfg.model)
```

and produce all required outputs without shape errors.

The model architecture should be:

```text
RGB image
    ↓
DINOv2 ViT-B/14 Vision Encoder
    ↓
z_img

Past motion trajectory
    ↓
HistoricalTrajectoryEncoder
    ↓
z_traj

Ordered task waypoint sequence
    ↓
TaskWaypointEncoder
    ↓
z_waypoint

z_img + z_traj + z_waypoint
    ↓
MultimodalFusionTransformer
    ↓
fused feature h

h
    ├── ModePredictionHead
    ├── AutoregressiveWaypointHead
    ├── CrawlActionHead
    └── TransitionHead
```

Please implement the project accordingly.

