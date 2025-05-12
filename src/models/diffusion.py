import torch
from torch import nn
# DDPM에서 DDIM으로 스케줄러 변경
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.models.backbones.rnn import RNNDiffusion
from src.models.backbones.unet import ConditionalUnet1D
from src.utils.configs import DataDict, DiffusionModelType


class Diffusion(nn.Module):
    def __init__(self, cfg, activation_func=nn.Softsign):
        super(Diffusion, self).__init__()
        self.model_type = cfg.model_type
        self.use_all_paths = cfg.use_all_paths
        self.sample_times = cfg.sample_times
        # DDPM 스케줄러에서 DDIM 스케줄러로 변경
        self.noise_scheduler = DDIMScheduler(beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                                             prediction_type="sample", num_train_timesteps=cfg.num_train_timesteps,
                                             clip_sample_range=cfg.clip_sample_range, clip_sample=cfg.clip_sample,
                                             beta_schedule=cfg.beta_schedule)
        self.time_steps = cfg.num_train_timesteps
        self.use_traversability = cfg.use_traversability
        self.estimate_traversability = cfg.estimate_traversability
        self.traversable_steps = cfg.traversable_steps

        # DDIM 관련 설정 추가
        self.inference_steps = cfg.inference_steps if hasattr(cfg, 'inference_steps') else 50  # DDIM의 추론 단계 수
        self.eta = cfg.eta if hasattr(cfg, 'eta') else 0.0  # DDIM의 stochasticity 파라미터
        
        # 적응형 디퓨전 단계 관련 설정 추가
        self.use_adaptive_steps = cfg.use_adaptive_steps if hasattr(cfg, 'use_adaptive_steps') else False
        self.min_inference_steps = cfg.min_inference_steps if hasattr(cfg, 'min_inference_steps') else 5
        self.max_inference_steps = cfg.max_inference_steps if hasattr(cfg, 'max_inference_steps') else 50
        self.last_used_steps = self.inference_steps  # 마지막으로 사용된 단계 수 추적
        
        # 환경 복잡성 점수 캐싱
        self.last_complexity_score = None
        self.complexity_history = []

        self.zd = cfg.diffusion_zd
        self.waypoint_dim = cfg.waypoint_dim
        self.waypoints_num = cfg.waypoints_num
        if activation_func is None:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), nn.LeakyReLU(0.1),
                                         nn.Linear(1024, 2048), nn.LeakyReLU(0.2),
                                         nn.Linear(2048, 512), nn.LeakyReLU(0.2),
                                         nn.Linear(512, self.zd), nn.LeakyReLU(0.2))
        else:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), activation_func(),
                                         nn.Linear(1024, 2048), activation_func(),
                                         nn.Linear(2048, 512), activation_func(),
                                         nn.Linear(512, self.zd), activation_func())
        self.trajectory_condition = nn.Linear(self.zd, self.zd)

        if self.model_type == DiffusionModelType.crnn:
            rnn_threshold = cfg.rnn_output_threshold
            self.diff_model = RNNDiffusion(in_dim=self.waypoint_dim * self.waypoints_num, out_dim=self.waypoint_dim,
                                           hidden_dim=self.zd, diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                                           steps=self.waypoints_num, rnn_type=cfg.rnn_type,
                                           output_threshold=rnn_threshold, activation_func=nn.Softsign)
        elif self.model_type == DiffusionModelType.unet:
            self.diff_model = ConditionalUnet1D(input_dim=self.waypoint_dim, global_cond_dim=self.zd,
                                                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                                                down_dims=cfg.down_dims, kernel_size=cfg.kernel_size,
                                                cond_predict_scale=cfg.cond_predict_scale, n_groups=cfg.n_groups)
        else:
            raise Exception("the diffusion model type is not defined")

    def add_trajectory_noise(self, trajectory):
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return noise

    def add_time_step_noise(self, trajectory, traversable_steps=None):
        if traversable_steps is None:
            time_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            time_steps = traversable_steps
        time_step = torch.randint(0, time_steps, (trajectory.shape[0],), device=trajectory.device).long()
        return time_step

    def add_trajectory_step_noise(self, trajectory, traversable_step=None):
        noise = self.add_trajectory_noise(trajectory=trajectory)
        time_step = self.add_time_step_noise(trajectory=trajectory)
        noisy_trajectory = self.noise_scheduler.add_noise(original_samples=trajectory, noise=noise, timesteps=time_step)
        if self.use_traversability:
            t_trajectories = trajectory.clone()
            t_noise = self.add_trajectory_noise(trajectory=t_trajectories)
            if traversable_step is None:
                traversable_step = self.traversable_steps
            t_time_step = self.add_time_step_noise(trajectory=t_trajectories, traversable_steps=traversable_step)
            t_noisy_trajectory = self.noise_scheduler.add_noise(original_samples=t_trajectories, noise=t_noise,
                                                                timesteps=t_time_step)
            noise = torch.concat((noise, t_noise))
            time_step = torch.concat((time_step, t_time_step))
            noisy_trajectory = torch.concat((noisy_trajectory, t_noisy_trajectory), dim=0)
        return noisy_trajectory, noise, time_step

    def forward(self, observation, gt_path=None, traversable_step=None):
        h = self.encoder(observation)  # B x 512
        h_condition = self.trajectory_condition(h)
        output = {}

        noisy_trajectory, noise, time_step = self.add_trajectory_step_noise(trajectory=gt_path, traversable_step=traversable_step)
        if self.use_traversability:
            h_condition = torch.concat((h_condition, h_condition), dim=0)
        pred = self.diff_model(noisy_trajectory, time_step, local_cond=None, global_cond=h_condition)
        output.update({
            DataDict.prediction: pred,
            DataDict.noise: noise,
            DataDict.time_steps: time_step
        })
        return output
    
    def set_adaptive_inference_steps(self, steps):
        """
        디퓨전 추론 단계 수를 동적으로 설정하는 함수
        
        Args:
            steps (int): 설정할 단계 수
        """
        # 범위 제한
        steps = max(self.min_inference_steps, min(self.max_inference_steps, steps))
        self.inference_steps = steps
        self.last_used_steps = steps
        return steps
    
    def adjust_inference_steps_by_complexity(self, complexity_score):
        """
        환경 복잡성 점수에 기반하여 디퓨전 단계를 조정하는 함수
        
        Args:
            complexity_score (float): 0~1 사이의 환경 복잡성 점수
            
        Returns:
            int: 조정된 디퓨전 단계 수
        """
        # 복잡성 점수 기록
        self.last_complexity_score = complexity_score
        self.complexity_history.append(complexity_score)
        if len(self.complexity_history) > 100:  # 최대 100개까지만 기록
            self.complexity_history.pop(0)
        
        # 시그모이드 함수를 사용한 매핑
        import math
        normalized_steps = 1.0 / (1.0 + math.exp(-10.0 * (complexity_score - 0.5)))
        steps_range = self.max_inference_steps - self.min_inference_steps
        adjusted_steps = int(self.min_inference_steps + normalized_steps * steps_range)
        
        # 단계 설정
        return self.set_adaptive_inference_steps(adjusted_steps)
    
    def get_complexity_stats(self):
        """
        복잡성 점수 통계를 반환하는 함수
        
        Returns:
            dict: 복잡성 통계 정보
        """
        if not self.complexity_history:
            return {"current": None, "mean": None, "min": None, "max": None}
        
        import numpy as np
        history = np.array(self.complexity_history)
        return {
            "current": self.last_complexity_score,
            "mean": float(np.mean(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
            "recent_mean": float(np.mean(history[-10:])) if len(history) >= 10 else float(np.mean(history))
        }

    @torch.no_grad()
    def sample(self, observation, custom_inference_steps=None):
        h = self.encoder(observation)  # B x 512
        h_condition = self.trajectory_condition(h)

        B, C = h_condition.shape
        trajectory = torch.randn(size=(h_condition.shape[0], self.waypoints_num, self.waypoint_dim),
                                 dtype=h_condition.dtype, device=h_condition.device, generator=None)
        all_trajectories = []
        
        # 사용할 단계 수 결정 (custom_inference_steps가 주어진 경우 우선 사용)
        inference_steps = custom_inference_steps if custom_inference_steps is not None else self.inference_steps
        
        # DDIM 샘플링 방식으로 변경
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(inference_steps)  # 계산된 단계 수 적용
        
        # 적응형 단계 사용 시 성능 모니터링을 위한 정보 추가
        sampling_info = {
            "used_steps": inference_steps,
            "start_time": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            "end_time": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        if sampling_info["start_time"] is not None:
            sampling_info["start_time"].record()
        
        for t in scheduler.timesteps:
            # DDIM에서는 sample_times의 의미가 달라지므로 조건 수정 가능
            if (self.sample_times >= 0) and (len(scheduler.timesteps) - scheduler.timesteps.tolist().index(t) > self.sample_times):
                break
                
            t = t.to(h_condition.device)
            model_output = self.diff_model(trajectory, t.unsqueeze(0).repeat(B, ), local_cond=None,
                                           global_cond=h_condition)
            
            # DDIM step: eta 파라미터를 통해 stochasticity 조절 가능
            trajectory = scheduler.step(model_output, t, trajectory, eta=self.eta, generator=None).prev_sample.contiguous()
            
            if self.use_all_paths:
                all_trajectories.append(model_output.clone().detach().cpu().numpy())
        
        if sampling_info["end_time"] is not None:
            sampling_info["end_time"].record()
            torch.cuda.synchronize()
            sampling_info["elapsed_time"] = sampling_info["start_time"].elapsed_time(sampling_info["end_time"])
        
        output = {
            DataDict.prediction: trajectory,
            DataDict.all_trajectories: all_trajectories,
            "sampling_info": sampling_info  # 샘플링 성능 정보 포함
        }
        return output