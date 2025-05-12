import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess
import numpy as np
import logging
import math

from warnings import warn
import torch
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import datetime, timedelta

from src.utils.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes, DataDict, GeneratorType
from src.loss import Loss
from src.models.diffusion import Diffusion
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader


class Trainer:
    def __init__(self, cfgs: TrainingConfig):
        """
        This class is the trainner
        Args:
            cfgs: the configuration of the training class
        """
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.train_time_steps = cfgs.train_time_steps

        self.iteration = 0
        self.epoch = 0
        self.training = False

        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
            # log_name = self.name + "-" + str(int(os.environ['WORLD_SIZE'])) + "-" + str(
            #     int(os.environ['LOCAL_RANK'])) + "/" + datetime.now().strftime("%m-%d-%Y-%H-%M")
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data.distributed = False
            # log_name = self.name + "-" + datetime.now().strftime("%m-%d-%Y-%H-%M")

        # model
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)

        # set up loggers
        self.output_dir = cfgs.output_dir
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
            "epochs": self.max_epoch
        }
        wandb.login(key=cfgs.wandb_api)
        if self.distributed:
            self.wandb_run = wandb.init(project=self.name, config=configs, group="DDP")
        else:
            self.wandb_run = wandb.init(project=self.name, config=configs)

        # loss, optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay)
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, eta_min=cfgs.lr_min,
                                                                                  T_0=cfgs.lr_t0, T_mult=cfgs.lr_tm)
        else:
            raise ValueError("the current scheduler is not defined")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(state_dict)

        # loss functions
        if self.device == "cuda":
            self.loss_func = Loss(cfg=cfgs.loss).cuda()
        else:
            self.loss_func = Loss(cfg=cfgs.loss).to(self.device)

        # datasets:
        self.training_data_loader = train_data_loader(cfg=cfgs.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)

        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.time_step_number = cfgs.model.diffusion.traversable_steps
        self.traversability_threshold = cfgs.traversability_threshold

    def _set_model_gpus(self, cfg):
        # self.current_rank = 0  # global rank
        # cfg.local_rank = os.environ['LOCAL_RANK']
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            # this will make all .cuda() calls work properly
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            # self.logger.info\
            print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                  % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logger.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logger.info(
        #         f'Model created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # move model to GPU, enable channels last layout if set
        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # setup distributed training
        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(self.model, device_ids=[cfg.local_rank],
                             broadcast_buffers=not cfg.no_ddp_bb,
                             find_unused_parameters=True)
            # NOTE: EMA model does not need to be wrapped by DDP

        # # setup exponential moving average of model weights, SWA could be used here too
        # model_ema = None
        # if args.model_ema:
        #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        #     model_ema = ModelEmaV2(
        #         self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch'] + 1
            print('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def save_snapshot(self, filename):
        """
        save the snapshot of the model and other training parameters
        Args:
            filename: the output filename that is the full directory
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)
        # print('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(self.output_dir, str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        # print('Snapshot saved to "{}"'.format(snapshot_filename))

    def cleanup(self):
        dist.destroy_process_group()
        self.wandb_run.finish()

    def set_train_mode(self):
        """
        set the model to the training mode: parameters are differentiable
        """
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def optimizer_step(self):
        """
        run one step of the optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def step(self, data_dict, train=True) -> Tuple[dict, dict]:
        """
        모델의 한 스텝을 실행하는 메서드. 경로 위험도를 먼저 계산한 후 환경 복잡성에 따라 디퓨전 단계 조정.
        
        Args:
            data_dict: 입력 데이터 딕셔너리
            train: 훈련 모드 여부 (기본값: True)
            
        Returns:
            output_dict: 모델 출력 및 손실 딕셔너리
        """
        # 데이터를 현재 디바이스로 이동
        data_dict = to_device(data_dict, device=self.device)
        
        if train:
            # 훈련 모드에서는 일반적인 처리 수행
            output_dict = self.model(data_dict, sample=False)
            torch.cuda.empty_cache()
            
            risk_output_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()}

            # 경로 위험도 계산 (재사용 가능한 grid maps 반환 받기)
            path_risk_score, grid_maps = self.compute_path_risk_score_grid(data_dict, risk_output_dict, return_grid_maps=True)
            output_dict["path_risk_score"] = path_risk_score
            
            # 손실 계산
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)
            torch.cuda.empty_cache()
        else:
            # 평가 모드에서는 먼저 기본 디퓨전 단계로 경로 생성
            
            # 1. 우선 기본 디퓨전 단계 설정 (중간 값으로 설정)
            if hasattr(self.model, 'diffusion') and hasattr(self.model.diffusion, 'use_adaptive_steps') and self.model.diffusion.use_adaptive_steps:
                # 기본값으로 중간 단계를 설정 (min_steps와 max_steps의 평균)
                default_steps = 25  # 또는 모델의 기본값 사용
                self.model.diffusion.set_adaptive_inference_steps(default_steps)
            
            # 2. 경로 생성 및 위험도 계산
            initial_output = self.model(data_dict, sample=True)
            path_risk_score, grid_maps = self.compute_path_risk_score_grid(data_dict, initial_output, return_grid_maps=True)
            initial_output["path_risk_score"] = path_risk_score
            
            # 3. 경로 위험도를 포함한 정확한 복잡성 점수 계산
            complexity_score, complexity_details = self.simplified_environment_complexity(
                data_dict, initial_output, path_risk_score
            )
            
            # 복잡성 정보 기록
            initial_output["complexity_score"] = complexity_score
            initial_output["complexity_details"] = self.filter_json_serializable(complexity_details)
            
            # 4. 계산된 복잡성 점수에 따라 디퓨전 단계 조정
            if hasattr(self.model, 'diffusion') and hasattr(self.model.diffusion, 'use_adaptive_steps') and self.model.diffusion.use_adaptive_steps:
                adaptive_steps = self.compute_adaptive_diffusion_steps(complexity_score)
                
                # 로깅
                self.wandb_run.log({
                    'adaptive/complexity_score': complexity_score,
                    'adaptive/diffusion_steps': adaptive_steps,
                    'adaptive/initial_path_risk': path_risk_score
                })
                
                # 기존 단계와 다른 경우에만 재계산
                if adaptive_steps != default_steps:
                    # 새로운 디퓨전 단계 설정
                    self.model.diffusion.set_adaptive_inference_steps(adaptive_steps)
                    
                    # 5. 필요시 개선된 경로 생성 (높은 위험도와 복잡성일 때)
                    if path_risk_score > 0.3 and complexity_score > 0.5:  # 임계값은 조정 가능
                        # 캐시된 복잡성 점수와 그리드 맵을 전달하여 효율적 계산
                        output_dict = self.predict_safe_trajectory(
                            data_dict, 
                            max_attempts=3,
                            risk_threshold=0.4,
                            cached_complexity_score=complexity_score,
                            cached_complexity_details=complexity_details,
                            cached_grid_maps=grid_maps
                        )
                        torch.cuda.empty_cache()
                    else:
                        # 낮은 위험도와 복잡성인 경우 초기 결과 사용
                        output_dict = initial_output
                else:
                    # 디퓨전 단계가 동일한 경우 초기 결과 사용
                    output_dict = initial_output
            else:
                # 적응형 단계를 사용하지 않는 경우 초기 결과 사용
                output_dict = initial_output
            
            # 평가 메트릭 계산
            eval_dict = self.loss_func.evaluate(output_dict)
            output_dict.update(eval_dict)
        
        return output_dict

    def update_log(self, results, timestep=None, log_name=None):
        """
        결과를 wandb에 로깅하는 함수 - JSON 직렬화 불가능한 객체 필터링
        
        Args:
            results: 로깅할 결과 딕셔너리
            timestep: 시간 측정값 (선택적)
            log_name: 로그 그룹 이름 (선택적)
        """
        # 입력 결과를 직렬화 가능한 형태로 변환
        serializable_results = self.filter_json_serializable(results)
        
        if timestep is not None:
            self.wandb_run.log({LogNames.step_time: timestep})
        
        if log_name == LogTypes.train:
            value = self.scheduler.get_last_lr()
            self.wandb_run.log({log_name + "/" + LogNames.lr: value[-1]})

        if log_name is None:
            for key, value in serializable_results.items():
                self.wandb_run.log({key: value})
        else:
            for key, value in serializable_results.items():
                self.wandb_run.log({log_name + "/" + key: value})

    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        # with open(self.output_file, "a") as f:
        #     print("Training CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1
            data_dict[DataDict.traversable_step] = self.time_step_number
            for step_iteration in range(self.train_time_steps):
                output_dict = self.step(data_dict=data_dict)
                torch.cuda.empty_cache()

                output_dict[LossNames.loss].backward()
                self.optimizer_step()
                optimize_time = time.time()

                output_dict = release_cuda(output_dict)
                self.update_log(results=output_dict, timestep=optimize_time - last_time, log_name=LogTypes.train)
                last_time = time.time()
        self.scheduler.step()

        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    def inference_epoch(self):
        # if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):    
        # if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0):
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                # if iteration % self.max_evaluation_iteration_per_epoch == 0 and iteration != 0:
                #     break
                start_time = time.time()
                output_dict = self.step(data_dict, train=False)
                torch.cuda.synchronize()
                step_time = time.time()
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(results=output_dict, timestep=step_time - start_time, log_name=LogTypes.others)

    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        self.set_eval_mode()
        self.inference_epoch()
        for self.epoch in range(self.epoch, self.max_epoch, 1):
            self.set_eval_mode()
            self.inference_epoch()

            self.set_train_mode()
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
        self.cleanup()
    
    def compute_path_risk_score_grid(self, data_dict, output_dict, grid_size=0.5, grid_map_size=100, return_grid_maps=False):
        """
        LiDAR 데이터를 2D Grid로 변환하여 Path Risk Score 계산 - 배치 크기 불일치 해결 버전
        
        Args:
            data_dict: 입력 데이터 딕셔너리
            output_dict: 출력 데이터 딕셔너리
            grid_size: 그리드 셀 크기 (미터 단위)
            grid_map_size: 그리드 맵 크기 (셀 개수)
            return_grid_maps: 계산된 그리드 맵 반환 여부 (기본값: False)
            
        Returns:
            float: Path Risk Score
            torch.Tensor (선택적): 계산된 그리드 맵 (return_grid_maps=True인 경우)
        """
        # 데이터를 현재 디바이스로 이동
        data_dict = to_device(data_dict, device=self.device)
        
        # 1. LiDAR 데이터 처리 - 최대한 벡터화된 연산 사용
        if DataDict.lidar not in data_dict:
            raise KeyError("LiDAR 데이터가 data_dict에 포함되지 않았습니다.")
        
        lidar_tensor = data_dict[DataDict.lidar]
        
        if len(lidar_tensor.shape) == 4:
            batch_size, channels, height, width = lidar_tensor.shape
            
            # 그리드 맵 초기화 - GPU 메모리 할당 최소화
            lidar_grid_maps = torch.zeros((batch_size, grid_map_size, grid_map_size), device=self.device)
            
            # 스케일링 계수 계산
            h_scale = grid_map_size / height
            w_scale = grid_map_size / width
            
            # 병렬 처리를 위한 인덱스 사전 계산
            batch_indices = torch.arange(batch_size, device=self.device)
            
            # 최적화 1: scatter_ 연산을 사용하여 그리드 맵 업데이트 - 배치별 반복 대신 벡터화
            non_zero_mask = lidar_tensor > 0.0
            non_zero_count = non_zero_mask.sum().item()
            
            if non_zero_count > 0:
                # 최적화를 위한 임계값 - 너무 많은 포인트가 있는 경우 기존 방식으로 처리
                if non_zero_count < 10_000_000:  # 적절한 임계값 조정 필요
                    # 모든 배치의 비어 있지 않은 포인트 한 번에 찾기
                    b_indices, c_indices, r_indices, col_indices = torch.nonzero(non_zero_mask, as_tuple=True)
                    
                    # 행/열 인덱스를 그리드 맵 인덱스로 변환
                    x_scaled = (r_indices * h_scale).floor()
                    y_scaled = (col_indices * w_scale).floor()
                    
                    # 클리핑으로 모든 인덱스를 유효 범위 내로 제한
                    x_clamped = torch.clamp(x_scaled, 0, grid_map_size - 1).long()
                    y_clamped = torch.clamp(y_scaled, 0, grid_map_size - 1).long()
                    
                    # CUDA 메모리 최적화: 큰 텐서를 작은 청크로 나누어 처리
                    chunk_size = 1_000_000  # 메모리 사용량에 따라 조정
                    num_chunks = (len(b_indices) + chunk_size - 1) // chunk_size
                    
                    for chunk in range(num_chunks):
                        start_idx = chunk * chunk_size
                        end_idx = min((chunk + 1) * chunk_size, len(b_indices))
                        
                        chunk_b = b_indices[start_idx:end_idx]
                        chunk_x = x_clamped[start_idx:end_idx]
                        chunk_y = y_clamped[start_idx:end_idx]
                        
                        # 각 청크별로 scatter_add_ 연산 사용
                        # 이 방식은 원자적 연산으로 충돌 없이 동시에 값을 누적할 수 있음
                        coordinate_indices = torch.stack([chunk_b, chunk_x, chunk_y], dim=0)
                        values = torch.ones(len(chunk_b), device=self.device)
                        
                        lidar_grid_maps.index_put_(
                            tuple(coordinate_indices), 
                            values, 
                            accumulate=True
                        )
                else:
                    # 포인트가 너무 많은 경우 배치별 처리 (원래 코드)
                    for b in range(batch_size):
                        # 모든 채널의 비어 있지 않은 포인트 찾기
                        non_zero_indices = torch.nonzero(lidar_tensor[b] > 0.0, as_tuple=True)
                        
                        if len(non_zero_indices[0]) > 0:
                            c_idx, r_idx, col_idx = non_zero_indices
                            
                            # 행/열 인덱스를 그리드 맵 인덱스로 변환
                            x_scaled = (r_idx * h_scale).floor()
                            y_scaled = (col_idx * w_scale).floor()
                            
                            # 클리핑
                            x_clamped = torch.clamp(x_scaled, 0, grid_map_size - 1).long()
                            y_clamped = torch.clamp(y_scaled, 0, grid_map_size - 1).long()
                            
                            # 최적화 2: bincount 사용하여 빠르게 카운트
                            coords = x_clamped * grid_map_size + y_clamped
                            max_idx = grid_map_size * grid_map_size
                            
                            if coords.max() < max_idx:
                                counts = torch.bincount(coords, minlength=max_idx)
                                counts_reshaped = counts.reshape(grid_map_size, grid_map_size)
                                lidar_grid_maps[b] += counts_reshaped
                            else:
                                # 기존 코드 (fallback)
                                unique_coords, counts = torch.unique(coords, return_counts=True)
                                unique_x = (unique_coords // grid_map_size).long()
                                unique_y = (unique_coords % grid_map_size).long()
                                
                                for i in range(len(unique_x)):
                                    lidar_grid_maps[b, unique_x[i], unique_y[i]] += counts[i]
        else:
            raise ValueError("예상치 못한 LiDAR 텐서 형태입니다. 형태: {}".format(lidar_tensor.shape))
        
        # 2. 좌표계 설정 - 중심점 기준 좌표계 정의
        grid_center = grid_map_size // 2
        
        # 3. 예측 경로 처리
        if DataDict.prediction not in output_dict:
            raise KeyError("prediction 데이터가 output_dict에 포함되지 않았습니다.")
        
        # 이미 앞에서 path_tensor가 조정된 경우를 고려
        if 'path_tensor' not in locals() or path_tensor is None:
            path_tensor = output_dict[DataDict.prediction]
        
        if len(path_tensor.shape) != 3 or path_tensor.shape[2] < 2:
            raise ValueError("예상치 못한 예측 경로 텐서 형태입니다. 형태: {}".format(path_tensor.shape))
        
        # 배치 크기 조정 - 트레버서빌리티 조정 이후의 배치 크기 사용
        pred_batch_size = path_tensor.shape[0]
        lidar_batch_size = lidar_grid_maps.shape[0]
        
        # 배치 크기 불일치 처리 - 양방향 처리
        if pred_batch_size != lidar_batch_size:
            # 1. pred_batch_size > lidar_batch_size 인 경우
            if pred_batch_size > lidar_batch_size:
                # 1:N 매핑이 명확한지 확인
                if pred_batch_size % lidar_batch_size == 0:
                    # 완벽한 배수 관계인 경우 (예: 1:4 매핑)
                    samples_per_input = pred_batch_size // lidar_batch_size
                    lidar_grid_maps = lidar_grid_maps.repeat_interleave(samples_per_input, dim=0)
                else:
                    # 불완전한 매핑 - 로그 출력
                    logging.warning(
                        f"배치 크기 불일치: pred_batch_size({pred_batch_size})가 lidar_batch_size({lidar_batch_size})의 "
                        f"정수 배수가 아닙니다. 근사적 매핑을 적용합니다."
                    )
                    
                    # 가능한 많은 항목에 대해 1:N 매핑 후 나머지는 순환적 매핑
                    samples_per_input = pred_batch_size // lidar_batch_size
                    if samples_per_input > 0:
                        # 기본 반복
                        lidar_grid_maps = lidar_grid_maps.repeat_interleave(samples_per_input, dim=0)
                        
                        # 나머지 처리 - 순환적 매핑
                        remaining = pred_batch_size - lidar_grid_maps.shape[0]
                        if remaining > 0:
                            mapping_indices = torch.arange(remaining, device=self.device) % lidar_batch_size
                            extra_maps = lidar_grid_maps[mapping_indices]
                            lidar_grid_maps = torch.cat([lidar_grid_maps, extra_maps], dim=0)
                    else:
                        # pred_batch_size < lidar_batch_size인 특수 케이스
                        lidar_grid_maps = lidar_grid_maps[:pred_batch_size]
            
            # 2. lidar_batch_size > pred_batch_size 인 경우
            else:
                # N:1 관계인 경우 처리
                if lidar_batch_size % pred_batch_size == 0:
                    # 첫 pred_batch_size 개만 사용
                    lidar_grid_maps = lidar_grid_maps[:pred_batch_size]
                else:
                    # 불완전한 N:1 매핑 - 로그 출력
                    logging.warning(
                        f"배치 크기 불일치: lidar_batch_size({lidar_batch_size})가 pred_batch_size({pred_batch_size})의 "
                        f"정수 배수가 아닙니다. lidar 데이터를 잘라냅니다."
                    )
                    # 처음 pred_batch_size 개의 lidar 맵만 사용
                    lidar_grid_maps = lidar_grid_maps[:pred_batch_size]
        
        # 4. 경로 좌표를 그리드 인덱스로 변환
        path_x_grid = (path_tensor[:, :, 0] / grid_size + grid_center)
        path_y_grid = (path_tensor[:, :, 1] / grid_size + grid_center)
        
        # 클리핑
        path_x_clamped = torch.clamp(path_x_grid, 0, grid_map_size - 1).long()
        path_y_clamped = torch.clamp(path_y_grid, 0, grid_map_size - 1).long()
        
        # 5. 경로 위험도 계산 - 완전 벡터화
        # 이제는 배치 크기가 일치해야 함
        assert path_tensor.shape[0] == lidar_grid_maps.shape[0], "배치 크기 불일치가 해결되지 않았습니다."
        batch_size = path_tensor.shape[0]
        
        # 경로의 각 점에 대한 밀도 값 가져오기
        densities = torch.zeros(batch_size, path_x_clamped.shape[1], device=self.device)
        
        # 메모리 효율을 위해 배치별로 처리하되, 내부 반복문은 벡터화
        for b in range(batch_size):
            densities[b] = lidar_grid_maps[b][path_x_clamped[b], path_y_clamped[b]]
        
        # 배치별 평균 계산 (완전 벡터화)
        density_values = densities.mean(dim=1)
        
        # 정규화 함수
        def improved_normalization(values):
            if torch.all(values == 0):
                return values
            
            # 중앙값 기반 정규화 - 이상치에 더 견고함
            median_val = torch.median(values[values > 0]) if torch.any(values > 0) else torch.tensor(1.0, device=values.device)
            
            # 로그 스케일 변환 - 값 범위를 압축
            if median_val > 0:
                log_values = torch.log1p(values / median_val)  # log(1+x)
                # 0-1 범위로 조정
                normalized = log_values / torch.log1p(torch.tensor(5.0, device=values.device))
                return torch.clamp(normalized, 0, 1)
            return values
        
        # 정규화 적용
        density_values = improved_normalization(density_values)
        
        # 최종 Path Risk Score
        P_risk = density_values.mean().item()
        
        # diffusion 모델의 traversability 사용 시 원래 배치 크기로 예측 복원
        if original_pred_batch_size is not None and use_traversability:
            # output_dict에 원래 배치 크기로 조절된 값 포함 (필요시)
            # output_dict[DataDict.prediction] = output_dict[DataDict.prediction][:original_pred_batch_size]
            pass
        
        if return_grid_maps:
            return P_risk, lidar_grid_maps
        else:
            return P_risk
    
    def predict_safe_trajectory(self, data_dict, max_attempts=3, risk_threshold=0.4, 
                      cached_complexity_score=None, cached_complexity_details=None,
                      cached_grid_maps=None):
        """
        위험도와 환경 복잡성에 기반한 안전 경로 생성 함수
        
        Args:
            data_dict: 입력 데이터 딕셔너리
            max_attempts: 최대 재시도 횟수 (기본값: 3)
            risk_threshold: 위험도 임계값 (기본값: 0.4)
            cached_complexity_score: 이미 계산된 환경 복잡성 점수
            cached_complexity_details: 이미 계산된 복잡성 세부 정보
            cached_grid_maps: 이미 계산된 그리드 맵 (계산 효율성 향상)
            
        Returns:
            output_dict: 안전한 경로가 포함된 출력 딕셔너리
        """
        # 재시도 횟수 기록용 카운터
        attempt = 0
        best_output = None
        lowest_risk = float('inf')
        
        # 환경 복잡성 정보 기록 (미리 계산된 경우)
        if cached_complexity_score is not None:
            complexity_score = cached_complexity_score
            # 복잡성 세부 정보도 캐시된 경우 사용
            if cached_complexity_details is not None:
                complexity_details = cached_complexity_details
        
        # 그리드 맵 캐싱 (제공된 경우)
        if cached_grid_maps is not None:
            self._cached_grid_maps = cached_grid_maps
        
        # 경로 생성 시도 - 위험도가 낮은 경로 찾기
        while attempt < max_attempts:
            # 경로 예측 수행 (각 시도마다 다른 시드 사용)
            current_seed = int(time.time() * 1000) + attempt
            torch.manual_seed(current_seed)
            
            output_dict = self.model(data_dict, sample=True)
            torch.cuda.empty_cache()
            
            # 경로 위험도 계산
            if attempt == 0 and cached_grid_maps is None:
                # 첫 번째 시도이고 캐시된 그리드 맵이 없는 경우 맵 생성
                path_risk_score, grid_maps = self.compute_path_risk_score_grid(data_dict, output_dict, return_grid_maps=True)
                self._cached_grid_maps = grid_maps
            elif hasattr(self, '_cached_grid_maps') and self._cached_grid_maps is not None:
                # 그리드 맵이 캐시된 경우, 이를 이용해 경로 위험도만 계산
                # 이 부분은 compute_path_risk_score_grid 함수의 그리드 맵 재사용 기능 구현이 필요
                # 간단한 구현으로는 다음을 사용할 수 있음:
                path_tensor = output_dict[DataDict.prediction]
                batch_size = path_tensor.shape[0]
                grid_map_size = self._cached_grid_maps.shape[1]  # 그리드 맵 크기
                grid_size = 0.5  # 그리드 셀 크기
                grid_center = grid_map_size // 2
                
                # 경로 좌표를 그리드 인덱스로 변환
                path_x_grid = (path_tensor[:, :, 0] / grid_size + grid_center).clamp(0, grid_map_size - 1).long()
                path_y_grid = (path_tensor[:, :, 1] / grid_size + grid_center).clamp(0, grid_map_size - 1).long()
                
                # 경로 위험도 계산 (배치 크기 맞추기)
                if self._cached_grid_maps.shape[0] != batch_size:
                    # 그리드 맵과 경로의 배치 크기가 다른 경우 처리
                    if self._cached_grid_maps.shape[0] == 1:
                        # 단일 그리드 맵을 배치 크기에 맞게 복제
                        grid_maps = self._cached_grid_maps.repeat(batch_size, 1, 1)
                    elif batch_size == 1:
                        # 여러 그리드 맵 중 첫 번째만 사용
                        grid_maps = self._cached_grid_maps[:1]
                    else:
                        # 배치 크기 불일치 경고 및 처리
                        print(f"경고: 그리드 맵 배치 크기({self._cached_grid_maps.shape[0]})와 경로 배치 크기({batch_size})가 일치하지 않습니다.")
                        # 가능한 경우 크기 조정 (예: 짜름 또는 반복)
                        if self._cached_grid_maps.shape[0] > batch_size:
                            grid_maps = self._cached_grid_maps[:batch_size]
                        else:
                            repeat_times = (batch_size + self._cached_grid_maps.shape[0] - 1) // self._cached_grid_maps.shape[0]
                            grid_maps = self._cached_grid_maps.repeat(repeat_times, 1, 1)[:batch_size]
                else:
                    grid_maps = self._cached_grid_maps
                
                # 경로의 각 점에 대한 밀도 값 가져오기
                densities = torch.zeros(batch_size, path_x_grid.shape[1], device=path_tensor.device)
                for b in range(batch_size):
                    densities[b] = grid_maps[b][path_x_grid[b], path_y_grid[b]]
                
                # 정규화 및 평균 계산
                def improved_normalization(values):
                    if torch.all(values == 0):
                        return values
                    median_val = torch.median(values[values > 0]) if torch.any(values > 0) else torch.tensor(1.0, device=values.device)
                    if median_val > 0:
                        log_values = torch.log1p(values / median_val)
                        normalized = log_values / torch.log1p(torch.tensor(5.0, device=values.device))
                        return torch.clamp(normalized, 0, 1)
                    return values
                
                # 배치별 평균 계산
                density_values = densities.mean(dim=1)
                density_values = improved_normalization(density_values)
                path_risk_score = density_values.mean().item()
            else:
                # 그리드 맵이 캐시되지 않은 경우 전체 계산
                path_risk_score = self.compute_path_risk_score_grid(data_dict, output_dict)
                    
            output_dict["path_risk_score"] = path_risk_score
            
            # 가장 안전한 경로 기록
            if path_risk_score < lowest_risk:
                lowest_risk = path_risk_score
                best_output = output_dict.copy()
            
            # 위험도가 임계값보다 낮으면 현재 예측 사용
            if path_risk_score < risk_threshold:
                # 복잡성 점수 있는 경우 추가
                if cached_complexity_score is not None:
                    output_dict["complexity_score"] = cached_complexity_score
                    
                    # 복잡성 세부 정보도 있는 경우 추가
                    if cached_complexity_details is not None:
                        output_dict["complexity_details"] = self.filter_json_serializable(cached_complexity_details)
                    elif hasattr(self, '_cached_complexity_details'):
                        output_dict["complexity_details"] = self.filter_json_serializable(self._cached_complexity_details)
                # 아직 복잡성 점수가 계산되지 않은 경우, 경로 위험도를 이용해 계산
                else:
                    complexity_score, complexity_details = self.simplified_environment_complexity(
                        data_dict, 
                        output_dict,
                        cached_path_risk=path_risk_score
                    )
                    output_dict["complexity_score"] = complexity_score
                    output_dict["complexity_details"] = self.filter_json_serializable(complexity_details)
                
                return output_dict
            
            attempt += 1
        
        # 최대 시도 횟수를 초과했을 때 가장 안전한 경로 반환
        if best_output is not None:
            # 복잡성 점수 있는 경우 추가
            if cached_complexity_score is not None:
                best_output["complexity_score"] = cached_complexity_score
                
                # 복잡성 세부 정보도 있는 경우 추가
                if cached_complexity_details is not None:
                    best_output["complexity_details"] = self.filter_json_serializable(cached_complexity_details)
                elif hasattr(self, '_cached_complexity_details'):
                    best_output["complexity_details"] = self.filter_json_serializable(self._cached_complexity_details)
            # 아직 복잡성 점수가 계산되지 않은 경우, 경로 위험도를 이용해 계산
            else:
                complexity_score, complexity_details = self.simplified_environment_complexity(
                    data_dict, 
                    best_output,
                    cached_path_risk=best_output.get("path_risk_score")
                )
                best_output["complexity_score"] = complexity_score
                best_output["complexity_details"] = self.filter_json_serializable(complexity_details)
            
            return best_output
        
        # 예외 케이스: 모든 시도가 실패하고 best_output도 없는 경우
        # 마지막 output_dict에 필요한 정보 추가
        if cached_complexity_score is not None:
            output_dict["complexity_score"] = cached_complexity_score
            
            if cached_complexity_details is not None:
                output_dict["complexity_details"] = self.filter_json_serializable(cached_complexity_details)
        else:
            # 경로 위험도를 이용해 복잡성 점수 계산
            complexity_score, complexity_details = self.simplified_environment_complexity(
                data_dict, 
                output_dict,
                cached_path_risk=output_dict.get("path_risk_score")
            )
            output_dict["complexity_score"] = complexity_score
            output_dict["complexity_details"] = self.filter_json_serializable(complexity_details)
        
        return output_dict
    
    def simplified_environment_complexity(self, data_dict, output_dict=None, cached_path_risk=None):
        """
        장애물 밀도와 경로 위험도를 사용하여 환경 복잡성을 계산하는 함수
        
        Args:
            data_dict: 입력 데이터 딕셔너리
            output_dict: 출력 데이터 딕셔너리 (선택적)
            cached_path_risk: 이미 계산된 경로 위험도 (선택적)
            
        Returns:
            float: 환경 복잡성 점수
            dict: 복잡성 세부 정보
        """
        device = next(iter(data_dict.values())).device
        
        # 1. 장애물 밀도 계산 (60% 반영)
        if DataDict.local_map in data_dict:
            # 로컬 맵에서 장애물 밀도 계산
            local_map = data_dict[DataDict.local_map]
            occupied_cells = (local_map > 0).float()
            total_cells = local_map.shape[1] * local_map.shape[2]
            occupied_ratio = occupied_cells.sum(dim=(1, 2)) / total_cells
        elif DataDict.lidar in data_dict:
            # LiDAR 데이터에서 장애물 밀도 계산
            lidar_tensor = data_dict[DataDict.lidar]
            # LiDAR 포인트 밀도 계산 (점이 있는 공간의 비율)
            non_zero_points = (lidar_tensor > 0).float().sum(dim=(1, 2, 3))
            total_points = torch.prod(torch.tensor(lidar_tensor.shape[1:], device=device)).float()
            occupied_ratio = non_zero_points / total_points
            
            # LiDAR 점의 공간적 분포 분석 (점들이 얼마나 밀집되어 있는지)
            if len(lidar_tensor.shape) == 4:
                batch_size, channels, height, width = lidar_tensor.shape
                
                # 그리드화로 점 밀집도 계산
                grid_size = 50  # 저해상도 그리드로 축소
                
                # 각 배치 항목에 대해 밀집도 계산
                concentration_scores = []
                
                for b in range(batch_size):
                    # 그리드별 점 수 계산
                    grid_counts = torch.zeros((grid_size, grid_size), device=device)
                    
                    # 비어 있지 않은 포인트 찾기
                    non_zero_indices = torch.nonzero(lidar_tensor[b] > 0, as_tuple=True)
                    
                    if len(non_zero_indices[0]) > 0:
                        # LiDAR 인덱스를 그리드 인덱스로 변환
                        c_idx, r_idx, col_idx = non_zero_indices
                        
                        # 행/열 인덱스를 그리드 맵 인덱스로 변환
                        h_scale = grid_size / height
                        w_scale = grid_size / width
                        
                        x_scaled = (r_idx * h_scale).floor()
                        y_scaled = (col_idx * w_scale).floor()
                        
                        # 클리핑
                        x_clamped = torch.clamp(x_scaled, 0, grid_size - 1).long()
                        y_clamped = torch.clamp(y_scaled, 0, grid_size - 1).long()
                        
                        # 좌표별로 그룹화하여 각 셀의 점 수 계산
                        coords = x_clamped * grid_size + y_clamped
                        max_idx = grid_size * grid_size
                        
                        if coords.max() < max_idx:
                            counts = torch.bincount(coords, minlength=max_idx)
                            counts_reshaped = counts.reshape(grid_size, grid_size)
                            grid_counts = counts_reshaped
                        
                        # 점이 있는 셀의 수
                        occupied_cells = (grid_counts > 0).sum()
                        
                        # 셀당 평균 점 수 (점이 있는 셀만 고려)
                        if occupied_cells > 0:
                            avg_points_per_cell = grid_counts.sum() / occupied_cells
                        else:
                            avg_points_per_cell = torch.tensor(0., device=device)
                        
                        # 변동 계수 (표준편차/평균) - 점의 분산 정도
                        non_zero_counts = grid_counts[grid_counts > 0]
                        if len(non_zero_counts) > 1:
                            cv = torch.std(non_zero_counts) / (torch.mean(non_zero_counts) + 1e-5)
                        else:
                            cv = torch.tensor(0., device=device)
                        
                        # 밀집도 점수 계산 (변동 계수와 평균 점수의 조합)
                        concentration_score = (cv * 0.7 + (avg_points_per_cell / 100) * 0.3).clamp(0, 1)
                        concentration_scores.append(concentration_score)
                    else:
                        concentration_scores.append(torch.tensor(0., device=device))
                
                # 밀집도 점수 평균
                if concentration_scores:
                    concentration = torch.stack(concentration_scores).mean()
                    # 최종 점수: 밀도와 밀집도 조합
                    occupied_ratio = 0.7 * occupied_ratio + 0.3 * concentration
        else:
            # 데이터가 없는 경우 기본값 사용
            occupied_ratio = torch.tensor([0.5], device=device)
        
        # 2. 경로 위험도 계산 (40% 반영)
        have_path_risk = False
        
        if cached_path_risk is not None:
            # 캐시된 경로 위험도 사용
            if isinstance(cached_path_risk, (float, int)):
                path_risk = torch.tensor([cached_path_risk], device=device)
            else:
                path_risk = cached_path_risk
            have_path_risk = True
        elif output_dict is not None and "path_risk_score" in output_dict:
            # output_dict에서 경로 위험도 가져오기
            path_risk = output_dict["path_risk_score"]
            if isinstance(path_risk, (float, int)):
                path_risk = torch.tensor([path_risk], device=device)
            have_path_risk = True
        
        # 3. 복잡성 점수 계산
        # 경로 위험도가 있는 경우 가중 평균 사용, 없는 경우 장애물 밀도만 사용
        if have_path_risk:
            # 텐서 형태 확인 및 변환
            if len(occupied_ratio.shape) == 0:
                occupied_ratio = occupied_ratio.unsqueeze(0)
            if len(path_risk.shape) == 0:
                path_risk = path_risk.unsqueeze(0)
            
            # 배치 크기 맞추기
            if occupied_ratio.shape[0] > 1 and path_risk.shape[0] == 1:
                path_risk = path_risk.repeat(occupied_ratio.shape[0])
            elif path_risk.shape[0] > 1 and occupied_ratio.shape[0] == 1:
                occupied_ratio = occupied_ratio.repeat(path_risk.shape[0])
            
            # 가중 평균으로 복잡성 점수 계산
            complexity_scores = 0.6 * occupied_ratio + 0.4 * path_risk
            avg_complexity = complexity_scores.mean().item()
        else:
            # 경로 위험도가 없는 경우 장애물 밀도만 사용
            avg_complexity = occupied_ratio.mean().item()
        
        # 세부 정보 구성
        details = {
            'occupied_ratio': float(occupied_ratio.mean().item()) if isinstance(occupied_ratio, torch.Tensor) else float(occupied_ratio),
        }
        
        # 경로 위험도가 있는 경우에만 추가
        if have_path_risk:
            details['path_risk'] = float(path_risk.mean().item()) if isinstance(path_risk, torch.Tensor) else float(path_risk)
        
        # 복잡성 세부 정보 캐싱
        self._cached_complexity_details = details
        
        return avg_complexity, details
    
    def compute_adaptive_diffusion_steps(self, complexity_score, min_steps=5, max_steps=25):
        """
        복잡성 점수에 기반하여 적절한 디퓨전 단계 수를 계산 - 시스템 상태 의존성 제거
        
        Args:
            complexity_score (float): 0-1 사이의 환경 복잡성 점수
            min_steps (int): 최소 디퓨전 단계 수
            max_steps (int): 최대 디퓨전 단계 수
            
        Returns:
            int: 계산된 디퓨전 단계 수
        """
        # 메모이제이션 캐시 초기화 (클래스 속성으로 저장)
        if not hasattr(self, '_steps_cache'):
            self._steps_cache = {}
        
        # 캐시 키 생성 (입력값의 조합) - 단순화
        key_complexity = round(complexity_score, 2)
        cache_key = (key_complexity, min_steps, max_steps)
        
        # 캐시에 있으면 바로 반환
        if cache_key in self._steps_cache:
            return self._steps_cache[cache_key]
        
        # 시그모이드 함수를 사용한 매핑
        # k 값이 클수록 복잡성 점수의 중간 영역에서 단계 변화가 급격해짐
        k = 10.0
        midpoint = 0.5
        
        normalized_steps = 1.0 / (1.0 + math.exp(-k * (complexity_score - midpoint)))
        steps_range = max_steps - min_steps
        raw_steps = min_steps + normalized_steps * steps_range
        
        # 최종 단계 수 (정수로 반올림)
        final_steps = max(min_steps, min(max_steps, round(raw_steps)))
        
        # 결과 캐싱
        self._steps_cache[cache_key] = final_steps
        
        # 캐시 크기 제한 (선택적)
        if len(self._steps_cache) > 1000:  # 적절한 크기로 조정
            # 가장 오래된 항목 제거 (간단한 FIFO)
            self._steps_cache.pop(next(iter(self._steps_cache)))
        
        return final_steps


    def get_system_status(self):
        """
        시스템 상태 정보 수집 (배터리, CPU 사용량 등) - 캐싱 적용
        
        Returns:
            dict: 시스템 상태 정보
        """
        # 시스템 상태는 짧은 시간 내에 크게 변하지 않으므로 캐싱 적용
        # 마지막 호출 시간과 데이터 저장
        current_time = time.time()
        
        # 캐시 만료 시간 설정 (3초)
        cache_expiry = 3.0  # 초 단위
        
        # 캐시된 데이터가 있고 만료되지 않았으면 재사용
        if hasattr(self, '_system_status_cache') and hasattr(self, '_system_status_time'):
            if current_time - self._system_status_time < cache_expiry:
                return self._system_status_cache
        
        try:
            # 실제 구현에서는 적절한 시스템 모니터링 라이브러리 활용
            import psutil
            
            # CPU 사용량 (0-1 범위)
            cpu_usage = psutil.cpu_percent() / 100.0
            
            # 배터리 상태 확인 (노트북/모바일 기기에서 실행 시)
            battery = 1.0  # 기본값 (전원 연결 상태 가정)
            
            if hasattr(psutil, 'sensors_battery'):
                battery_info = psutil.sensors_battery()
                if battery_info is not None:
                    battery = battery_info.percent / 100.0
                    # 전원 연결 상태면 배터리 걱정 없음
                    if battery_info.power_plugged:
                        battery = 1.0
            
            # GPU 사용량 (CUDA 사용 가능 시)
            gpu_usage = 0.5  # 기본값
            if torch.cuda.is_available():
                # 실제 구현에서는 nvidia-smi 또는 pynvml 활용
                # 간단한 예시로 대체
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0.5
            
            system_status = {
                'battery': battery,
                'cpu_usage': cpu_usage,
                'gpu_usage': gpu_usage
            }
            
            # 결과 캐싱
            self._system_status_cache = system_status
            self._system_status_time = current_time
            
            return system_status
        except:
            # 시스템 상태 확인 실패 시 기본값 반환
            default_status = {
                'battery': 1.0,
                'cpu_usage': 0.5,
                'gpu_usage': 0.5
            }
            
            # 기본값도 캐싱
            self._system_status_cache = default_status
            self._system_status_time = current_time
            
            return default_status
        
        
    def filter_json_serializable(self, obj):
        """
        객체를 JSON 직렬화 가능한 형태로 변환
        
        Args:
            obj: 변환할 객체 (딕셔너리, 리스트, 기본 타입 등)
            
        Returns:
            JSON 직렬화 가능한 객체
        """
        import threading
        import json
        
        # 객체가 기본 타입인 경우 그대로 반환
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # 리스트인 경우 각 항목을 개별적으로 처리
        if isinstance(obj, list):
            return [self.filter_json_serializable(item) for item in obj]
        
        # 딕셔너리인 경우 각 키-값 쌍을 개별적으로 처리
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Event 객체인 경우 Boolean으로 변환
                if isinstance(v, threading.Event):
                    result[k] = v.is_set()
                # 기타 직렬화 불가능한 객체 필터링
                else:
                    try:
                        # 직렬화 가능한지 테스트
                        json.dumps(v)
                        result[k] = v
                    except (TypeError, OverflowError):
                        # 직렬화 불가능한 경우 문자열로 변환
                        try:
                            result[k] = str(v)
                        except:
                            result[k] = "unserializable_object"
            return result
        
        # 텐서나 기타 객체는 문자열로 변환 시도
        try:
            # PyTorch 텐서의 경우 item() 또는 tolist() 변환 시도
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            
            # 그 외의 경우 직렬화 테스트
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError, RuntimeError):
            # 직렬화 불가능한 경우 문자열로 변환
            try:
                return str(obj)
            except:
                return "unserializable_object"