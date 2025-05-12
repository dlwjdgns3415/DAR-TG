import math
from os.path import join
import numpy as np
import yaml
from easydict import EasyDict as edict


#########################################################################
# Configuration of Dataset and Data Loader
#########################################################################
class DataDict:
    pose = "pose"
    time = "time"
    camera = "camera"
    scan = "scan"
    lidar2d = "lidar2d"
    local_map = "local_map"
    all_paths = "all_paths"
    targets = "targets"
    trajectories = "trajectories"

    target = "target"
    raw_target = "raw_target"
    path = "path"
    lidar = "lidar"
    vel = "vel"
    imu = "imu"
    heuristic = "heuristic"

    file_names = "file_names"
    all_positions = "all_positions"
    network = "network"
    frame_name = "frame_name"
    images = "images"
    running_time = "running_time"
    ori_trajectories = "ori_trajectories"
    traversability_gt = "traversability_gt"

    prediction = "prediction"
    time_steps = "time_steps"
    traversable_step = "traversable_step"
    predict_path = "predict_path"
    ground_truth = "ground_truth"
    embedding = "embedding"
    trajetory = "trajetory"
    zmu = "mu"
    zvar = "logvar"
    noise = "noise"
    all_trajectories = "all_trajectories"
    all_variances = "all_variances"
    traversability_pred = "traversability_pred"
    traversability_mu = "traversability_mu"
    traversability_var = "traversability_var"

    A = "A"
    b = "b"

class CameraType:
    realsense_d435i = 0
    realsense_l515 = 1


class DatasetType:
    test = "test"
    train = "train"


class GeneratorType:
    diffusion = 0
    cvae = 1


# DDIM과 DDPM 스케줄러 타입을 구분하기 위한 클래스 추가
class SchedulerType:
    ddpm = "ddpm"
    ddim = "ddim"


Lidar_cfg = edict()
Lidar_cfg.threshold = 100
Lidar_cfg.channels = 16
Lidar_cfg.horizons = 1824
Lidar_cfg.angle_range = 200

DatasetConfig = edict()  # Configuration of data loaders
DatasetConfig.name = ""
DatasetConfig.root = "/home/dke/jhlee/DTG/data_sample"
DatasetConfig.batch_size = 16
DatasetConfig.num_workers = 8
DatasetConfig.shuffle = False
DatasetConfig.distributed = False
DatasetConfig.training_data_percentage = 0.95

DatasetConfig.lidar_cfg = Lidar_cfg
DatasetConfig.vel_num = 10
DatasetConfig.imu_num = 0


#########################################################################
# Configuration of Models
#########################################################################
class RNNType:
    gru = 0
    lstm = 1


class DiffusionModelType:
    unet = "unet"
    crnn = "crnn"


class CRNNType:
    lstm = "lstm"
    gru = "gru"


Perception = edict()
Perception.fix_perception = False
Perception.vel_dim = 20
Perception.vel_out = 256
Perception.lidar_out = 512
Perception.lidar_norm_layer = False
Perception.lidar_num = 3

CVAE = edict()
CVAE.activation_func = None
CVAE.rnn_type = RNNType.gru
CVAE.perception_in = Perception.lidar_out + Perception.vel_out + 2
CVAE.vae_zd = 512
CVAE.vae_output_threshold = 1
CVAE.paths_num = 5
CVAE.waypoints_num = 16
CVAE.waypoint_dim = 2
CVAE.estimate_traversability = True
CVAE.fix_first = False
CVAE.cvae_file = None

CRNN = edict()
CRNN.type = CRNNType.gru
CRNN.waypoint_num = 16

Diffusion = edict()
# 기본 설정
Diffusion.beta_start = 0.0001
Diffusion.beta_end = 0.02
Diffusion.beta_schedule = "squaredcos_cap_v2"
Diffusion.clip_sample = True  # default clip range = 1
Diffusion.clip_sample_range = 1.0  # default clip range = 1
Diffusion.num_train_timesteps = 100  # 학습 시 timestep 수
Diffusion.variance_type = "fixed_small"

# 모델 설정
Diffusion.perception_in = Perception.lidar_out + Perception.vel_out + 2
Diffusion.diffusion_zd = 512
Diffusion.waypoint_dim = 2
Diffusion.waypoints_num = 16
Diffusion.rnn_type = RNNType.gru
Diffusion.rnn_output_threshold = 1
Diffusion.diffusion_step_embed_dim = 256
Diffusion.down_dims = [512, 1024, 2048]
Diffusion.kernel_size = 5
Diffusion.cond_predict_scale = True
Diffusion.use_traversability = True
Diffusion.estimate_traversability = True
Diffusion.traversable_steps = 10
Diffusion.traversable_steps_buffer = 5
Diffusion.n_groups = 8
Diffusion.model_type = DiffusionModelType.crnn
Diffusion.use_all_paths = True
Diffusion.sample_times = -1
Diffusion.crnn = CRNN
Diffusion.use_adaptive_steps = True

# DDIM 관련 설정
Diffusion.scheduler_type = SchedulerType.ddim  # DDIM 스케줄러 사용 설정 (ddpm 또는 ddim)
Diffusion.inference_steps = 50  # DDIM 샘플링 시 사용할 스텝 수 (일반적으로 DDPM보다 훨씬 작은 값 사용)
Diffusion.eta = 0.0  # DDIM에서 stochastic sampling 조절 (0: deterministic, 1: fully stochastic like DDPM)
Diffusion.use_clipped_model_output = True  # 모델 출력을 clipping할지 여부

ModelConfig = edict()
ModelConfig.generator_type = GeneratorType.diffusion  # CVAE에서 Diffusion으로 변경
ModelConfig.cvae = CVAE
ModelConfig.diffusion = Diffusion
ModelConfig.perception = Perception
ModelConfig.scale_waypoints = 1.0


#########################################################################
# Configuration of Loss
#########################################################################
class Hausdorff:
    average = "average"
    max = "max"


class LossNames:
    kld = "kld"
    path_dis = "path_dis"
    last_dis = "last_dis"
    traversability = "traversability"
    est_tra_rec = "est_tra_rec"
    est_tra_kld = "est_tra_kld"
    path_risk = "path_risk"

    evaluate_last_dis = "evaluate_last_dis"
    evaluate_path_dis = "evaluate_path_dis"
    evaluate_traversability = "evaluate_traversability"
    evaluate_estimation_traversability = "evaluate_estimation_traversability"

    loss = "loss"


LossConfig = edict()
LossConfig.train_poses = True
LossConfig.scale_waypoints = 1.0
LossConfig.use_traversability = True
LossConfig.distance_type = Hausdorff.average
LossConfig.distance_ratio = 10.0
LossConfig.last_ratio = 2.0
LossConfig.vae_kld_ratio = 1.0
LossConfig.traversability_ratio = 10.0
LossConfig.generator_type = ModelConfig.generator_type
LossConfig.traversability_estimation_reconstruct_ratio = 10.0
LossConfig.root = "/home/jing/Documents/gn/database/datasets/regular_data"

LossConfig.map_resolution = 0.1
LossConfig.map_range = 300
LossConfig.image_separate = 20
LossConfig.lossoutput_dir = "/home/dke/jhlee/DTG/results"


#########################################################################
# Configuration of Training
#########################################################################
class ScheduleMethods:
    step = "step"
    cosine = "cosine"


class LogNames:
    step_time = "step_time"
    lr = "learning_rate"


class LogTypes:
    train = "train"
    others = "evaluation"


TrainingConfig = edict()
TrainingConfig.name = "ddim_training"  # 이름 변경하여 DDIM 모델임을 표시
TrainingConfig.wandb_api = ""
TrainingConfig.only_model = False
TrainingConfig.output_dir = "./results"
TrainingConfig.snapshot = ""
TrainingConfig.max_epoch = 150
TrainingConfig.evaluation_freq = 5
TrainingConfig.train_time_steps = 32
TrainingConfig.scheduler = ScheduleMethods.cosine
TrainingConfig.lr = 1e-4
TrainingConfig.weight_decay = 0
# for cosine scheduler
TrainingConfig.lr_t0 = 1
TrainingConfig.lr_tm = 5
TrainingConfig.lr_min = 1e-7
TrainingConfig.traversability_threshold = 1e-7

TrainingConfig.gpus = edict()
TrainingConfig.gpus.channels_last = False
TrainingConfig.gpus.local_rank = 0
TrainingConfig.gpus.sync_bn = False
TrainingConfig.gpus.no_ddp_bb = False
TrainingConfig.gpus.device = "cuda"

TrainingConfig.data = DatasetConfig
TrainingConfig.model = ModelConfig
TrainingConfig.loss = LossConfig