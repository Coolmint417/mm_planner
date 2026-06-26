from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # General
    device: str = "cuda"
    seed: int = 42

    # Vision encoder
    vision_model_name: str = "dinov2_vitb14"
    vision_output_dim: int = 256
    freeze_vision_encoder: bool = True
    use_patch_tokens: bool = True
    image_size: int = 224
    dinov2_feature_dim: int = 768
    allow_mock_vision_encoder: bool = False

    # Historical trajectory encoder
    n_traj_encoder: int = 8
    num_motion_modes: int = 4  # flying/takeoff/landing/crawling/
    traj_continuous_dim: int = 16
    mode_embedding_dim: int = 16
    traj_token_dim: int = 256
    traj_encoder_layers: int = 2
    traj_encoder_heads: int = 4
    traj_encoder_ff_dim: int = 512
    traj_dropout: float = 0.1

    # Task waypoint encoder
    waypoint_dim: int = 7  
    # task_point x, y, z, yaw
    # (optional nx, ny, nz, if the point is on a plane)
    waypoint_token_dim: int = 256
    waypoint_encoder_layers: int = 2
    waypoint_encoder_heads: int = 4
    waypoint_encoder_ff_dim: int = 512
    waypoint_dropout: float = 0.1

    # Fusion transformer
    fusion_dim: int = 256
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_ff_dim: int = 1024
    fusion_dropout: float = 0.1

    # Prediction heads
    hidden_dim: int = 256
    num_modes: int = 4 # flying/takeoff/landing/crawling/

    # Flight trajectory prediction
    pred_num_flight_waypoints: int = 5
    pred_flight_waypoint_dt: float = 0.5
    pred_waypoint_dim: int = 4
    pred_velocity_delta_dim: int = 4  # [dv_x, dv_y, dv_z, ddyaw]
    pred_acceleration_delta_dim: int = 3  # [da_x, da_y, da_z]
    autoregressive_hidden_dim: int = 256
    autoregressive_predict_delta: bool = True
    max_waypoint_delta: float = 1.0
    min_action_sigma: float = 1e-4

    # Crawling action
    crawl_action_dim: int = 3
    max_crawl_vx: float = 0.2
    max_crawl_vy: float = 0.2
    max_crawl_yaw_rate: float = 0.5

    # Transition action
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
    lambda_velocity_delta: float = 0.5
    lambda_acceleration_delta: float = 0.5
    lambda_crawl: float = 1.0
    lambda_transition: float = 1.0
    lambda_smooth: float = 0.1

    teacher_forcing_ratio: float = 0.5


@dataclass
class DataConfig:
    data_dir: str = "data/demos"
    clip_pattern: str = "clip_*.npz"
    samples_per_epoch: int = 10000
    cache_clips: bool = True

    # Temporal sampling. History/future tensors use fixed offsets around a
    # randomly sampled anchor timestamp inside each clip.
    history_dt: float = 0.1

    # Input image preprocessing.
    rgb_key: str = "rgb"
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Stored clip keys.
    mode_key: str = "position_auto_mode"
    fallback_mode_key: str = "mode_id"
    task_waypoints_key: str = "task_waypoints"

    # Targets are expressed relative to the anchor state by default, which
    # keeps the autoregressive action heads centered around zero.
    relative_targets: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


cfg = Config()
