import copy
import math
import os
import pickle
import shutil
from os.path import join, exists
from heapq import heappush, heappop
from typing import Tuple, Optional, Dict, List

import cv2
import imageio
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from src.utils.configs import LossConfig
from src.models.diff_hausdorf import HausdorffLoss
from src.utils.configs import GeneratorType, DataDict, Hausdorff, LossNames


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        with open(join(cfg.root, "data.pkl"), "rb") as input_file:
            data = pickle.load(input_file)
            self.network = data[DataDict.network]

        self.generator_type = cfg.generator_type
        # print(f"generator_type: {self.generator_type}")
        self.use_traversability = cfg.use_traversability
        self.collision_distance = 0.5

        self.target_dis = nn.MSELoss(reduction="mean")
        self.distance = HausdorffLoss(mode=cfg.distance_type)
        self.train_poses = cfg.train_poses
        self.distance_type = cfg.distance_type
        self.scale_waypoints = cfg.scale_waypoints
        self.last_ratio = cfg.last_ratio
        self.distance_ratio = cfg.distance_ratio
        self.vae_kld_ratio = cfg.vae_kld_ratio
        self.traversability_ratio = cfg.traversability_ratio

        self.map_resolution = cfg.map_resolution
        self.map_range = cfg.map_range
        self.output_dir = cfg.lossoutput_dir
        if self.output_dir:
            if not exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_accumulating = False
        self.metric_accumulator = {}
        
    def start_accumulation(self):
        """Initializes metric accumulator. Call this before the evaluation loop."""
        print("Starting metric accumulation...")
        self.is_accumulating = True
        self.metric_accumulator = {
            'Whole_evaluate_last_dis': [],
            'Whole_evaluate_path_dis': [],
            'Whole_dtg_traversability_rate': [],
            'Whole_true_dtg_traversability_rate': [],
        }

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Computes mean and std for all accumulated metrics.
        Call this after the evaluation loop.
        """
        print("Calculating final metrics from accumulated data...")
        self.is_accumulating = False
        final_results = {}
        for key, values in self.metric_accumulator.items():
            if not values:
                print(f"Warning: Metric '{key}' has no values.")
                mean, std = 0.0, 0.0
            else:
                mean = np.mean(values)
                std = np.std(values)
            
            final_results[f'{key}_mean'] = mean
            final_results[f'{key}_std'] = std
            
        print("Final metrics calculated.")
        return final_results
    
    def _cropped_distance(self, path, single_map):
        N, Cp = path.shape
        M, Cs = single_map.shape
        assert Cs == Cp, "dimension should be the same, but get {}, {}".format(Cs, Cp)
        single_map = single_map.view(M, 1, Cs).to(torch.float)  # Mx1xC
        path = path.view(1, N, Cs)  # 1xNxC
        d = torch.min(torch.norm(single_map - path, dim=-1), dim=0)[0] * self.map_resolution  # N

        traversability = torch.clamp(d, 0.0001, self.collision_distance)
        values = traversability[torch.where(traversability < self.collision_distance)]
        if len(values) < 1:
            return (torch.tensor(0, device=traversability.device, dtype=torch.float),
                    torch.tensor(1, device=traversability.device, dtype=torch.float))
        else:
            torch.cuda.empty_cache()
            loss = torch.arctanh((self.collision_distance - values) / self.collision_distance)
            return loss.mean(), values.mean()

    def _local_collision(self, yhat, local_map):
        assert len(yhat.shape) == 3, "the shape should be B,N,2"
        By, N, C = yhat.shape
        Bl, W, H = local_map.shape
        assert Bl == By, "the batch shape {} and {} should be the same".format(By, Bl)
        assert W == H, "the local map width {} not equals to height {}".format(W, H)
        pixel_yhat = yhat / self.map_resolution + self.map_range
        pixel_yhat = pixel_yhat.to(torch.int)
        all_losses = []
        traversability_values = []
        for i in range(By):
            map_indices = torch.stack(torch.where(local_map[i] > 0), dim=1)
            loss, traversability = self._cropped_distance(pixel_yhat[i], map_indices)
            all_losses.append(loss)
            traversability_values.append(traversability)
        return torch.stack(all_losses), torch.stack(traversability_values)

    def forward_cvae(self, input_dict):
        mu = input_dict[DataDict.zmu]
        logvar = input_dict[DataDict.zvar]
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]
        y_last = ygt[:, -1, :]

        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        path_dis = self.distance(ygt, y_hat_poses).mean()
        last_pose_dis = self.target_dis(y_last, y_hat_poses[:, -1, :])
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_last.shape[0]
        all_loss = self.distance_ratio * path_dis + self.last_ratio * last_pose_dis + self.vae_kld_ratio * kld_loss
        output = {
            LossNames.kld: kld_loss,
            LossNames.last_dis: last_pose_dis,
            LossNames.path_dis: path_dis,
        }

        if self.use_traversability:
            local_map = input_dict[DataDict.local_map]
            traversability_loss, traversability_values = self._local_collision(yhat=y_hat_poses, local_map=local_map)
            traversability_loss_mean = traversability_loss.mean()
            all_loss += self.traversability_ratio * traversability_loss_mean
            output.update({LossNames.traversability: traversability_loss_mean})

        output.update({LossNames.loss: all_loss})
        return output

    def forward_diffusion(self, input_dict):
        ygt = input_dict[DataDict.path]                 # (B_gt, T, C)
        y_hat = input_dict[DataDict.prediction]         # (B_pred, T, C)  # trav 사용 시 보통 2*B_gt
        output = {}

        # 1) 복원
        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        B_pred = y_hat_poses.size(0)
        B_gt   = ygt.size(0)

        # 2) 배치 정렬: pred만 2배인 경우 처리
        if self.use_traversability:
            if B_pred == 2 * B_gt:
                half = B_pred // 2
                # 전반부: 거리 손실용, 후반부: 통행성 손실용
                y_hat_poses_main       = y_hat_poses[:half]   # (B_gt, T, C)
                traversability_hat_pos = y_hat_poses[half:]   # (B_gt, T, C)
                ygt_main               = ygt                  # (B_gt, T, C)
                local_map_trav         = input_dict[DataDict.local_map]  # (B_gt, W, H)
            else:
                # 예외 케이스 방어: 크기 불일치 시 최소치로 맞춤
                minB = min(B_pred, B_gt)
                y_hat_poses_main       = y_hat_poses[:minB]
                traversability_hat_pos = y_hat_poses[:minB]
                ygt_main               = ygt[:minB]
                local_map_trav         = input_dict[DataDict.local_map][:minB]
        else:
            y_hat_poses_main = y_hat_poses
            ygt_main         = ygt

        # 3) 거리 손실 (스칼라)
        path_dis_mean  = self.distance(ygt_main, y_hat_poses_main).mean()
        last_pose_mean = self.target_dis(ygt_main[:, -1, :], y_hat_poses_main[:, -1, :]).mean()
        distance_loss  = self.distance_ratio * path_dis_mean + self.last_ratio * last_pose_mean

        output.update({
            LossNames.path_dis: path_dis_mean.detach(),
            LossNames.last_dis: last_pose_mean.detach(),
        })

        # 4) 통행성 손실 (스칼라)
        trav_mean = torch.tensor(0.0, device=y_hat.device)
        if self.use_traversability:
            trav_loss, trav_vals = self._local_collision(
                yhat=traversability_hat_pos, local_map=local_map_trav
            )   # trav_loss: (B_gt,)
            trav_mean = trav_loss.mean()
            output.update({LossNames.traversability: trav_mean.detach()})

        # 6) 최종 스칼라 손실
        all_loss = distance_loss + (self.traversability_ratio * trav_mean)
        output.update({LossNames.loss: all_loss})
        return output
    
    def forward(self, input_dict):
        if self.generator_type == GeneratorType.cvae:
            return self.forward_cvae(input_dict=input_dict)
        elif self.generator_type == GeneratorType.diffusion:
            return self.forward_diffusion(input_dict=input_dict)

    def convert_path_pixel(self, trajectory):
        return np.clip(np.around(trajectory / self.map_resolution)[:, :2] + self.map_range, 0, np.inf)

    def show_path_local_map(self, trajectory, gt_path, local_map, idx=0, indices=0, flag="gt"):
        filename = f"local_map_trajectory_{flag}_{indices}_{idx}.png"
        file_path = join(self.output_dir, filename)
        return write_png(local_map=local_map, center=np.array([local_map.shape[0] / 2, local_map.shape[1] / 2]),
                         file=file_path,
                         paths=[self.convert_path_pixel(trajectory=trajectory)],
                         others=self.convert_path_pixel(trajectory=gt_path)
                         )
    
    def _trajectory_to_pixel(self, trajectory: np.ndarray) -> np.ndarray:
        """궤적을 픽셀 좌표로 변환"""
        return np.round(trajectory / self.map_resolution + self.map_range).astype(int)

    def _filter_valid_points(self, pixel_trajectory: np.ndarray, map_shape: Tuple[int, int]) -> np.ndarray:
        """맵 범위 내의 유효한 점들만 필터링"""
        valid_x = (pixel_trajectory[:, 0] >= 0) & (pixel_trajectory[:, 0] < map_shape[0])
        valid_y = (pixel_trajectory[:, 1] >= 0) & (pixel_trajectory[:, 1] < map_shape[1])
        return np.where(valid_x & valid_y)[0]

    def calculate_traversability_metric(self, 
                                        trajectory: np.ndarray, 
                                        local_map: np.ndarray) -> Tuple[float, float]:

        # --- 1. 궤적을 픽셀 좌표로 변환 ---  
        pixel_trajectory = self._trajectory_to_pixel(trajectory)

        # --- 2. 맵 범위 내에 있는 유효한 웨이포인트만 필터링 ---
        valid_indices = self._filter_valid_points(pixel_trajectory, local_map.shape)

        # 궤적의 모든 점이 맵 밖으로 벗어난 경우, 통행 가능성은 0%
        if len(valid_indices) == 0:
            return 0.0, 0.0

        valid_pixel_trajectory = pixel_trajectory[valid_indices]

        # --- 3. 유효한 웨이포인트 위치의 맵 값을 직접 조회 ---
        waypoint_ys = valid_pixel_trajectory[:, 1]
        waypoint_xs = valid_pixel_trajectory[:, 0]

        # 각 웨이포인트 위치에 해당하는 local_map의 값을 가져옴
        map_values_at_waypoints = local_map[waypoint_xs, waypoint_ys]

        # 통행 가능한 포인트는 맵의 값이 0 이하인 경우
        traversable_points_count = np.sum(map_values_at_waypoints <= 0)

        # --- 4. 통행 가능 비율 계산 ---
        # 통행 가능한 포인트 수를 '전체' 웨이포인트 수로 나.
        traversability_rate = traversable_points_count / len(trajectory)

        hit_obstacle = np.any(map_values_at_waypoints > 0)
        true_dtg_traversability = 0.0 if hit_obstacle else 1.0

        return traversability_rate, true_dtg_traversability
    
    def evaluate_dtg_metrics(self, 
                             trajectories: np.ndarray,
                             local_maps: np.ndarray, # 이 이름은 traversability metric에 사용되는 맵을 의미
                             traversability_maps: Optional[np.ndarray] = None) -> Dict: # distance ratio용 맵

        batch_size = len(trajectories)
        traversability_rates = []
        true_traversability_rates = []
        
        for i in range(batch_size):
            # Traversability 계산
            # local_maps[i]는 traversability 지표 계산용 맵
            trav_rate, true_rate  = self.calculate_traversability_metric(
                trajectories[i], local_maps[i] 
            )
            traversability_rates.append(trav_rate)
            true_traversability_rates.append(true_rate)

        return traversability_rates, true_traversability_rates
    
    @torch.no_grad()
    def evaluate(self, input_dict, indices=0):
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]
        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

            path_dis = self.distance(ygt, y_hat_poses).mean()
            last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
            output = {
                LossNames.evaluate_last_dis: last_pose_dis,
                LossNames.evaluate_path_dis: path_dis,
            }
            dtg_traversability_rates = []
            true_dtg_traversability_rates = []

            if self.use_traversability:
                local_map = input_dict[DataDict.local_map]
                traversability_loss, traversability_values = self._local_collision(yhat=y_hat_poses, local_map=local_map)
                traversability_loss_mean = traversability_loss.mean()
                output.update({LossNames.evaluate_traversability: traversability_loss_mean})

                trajectories_np = y_hat_poses.detach().cpu().numpy()
                local_maps_np = local_map.detach().cpu().numpy()
                dtg_traversability_rates, true_dtg_traversability_rates  = self.evaluate_dtg_metrics(
                trajectories=trajectories_np,
                local_maps=local_maps_np
                )
                output.update({
                'dtg_traversability_mean': np.mean(dtg_traversability_rates) if dtg_traversability_rates else 0.0,
                'dtg_traversability_std': np.std(dtg_traversability_rates) if dtg_traversability_rates else 0.0,
                'true_dtg_traversability_mean': np.mean(true_dtg_traversability_rates) if true_dtg_traversability_rates else 0.0,
                'true_dtg_traversability_std':  np.std(true_dtg_traversability_rates)  if true_dtg_traversability_rates else 0.0,
                })
            if self.is_accumulating:
                self.metric_accumulator['Whole_evaluate_path_dis'].extend(
                torch.atleast_1d(path_dis).cpu().numpy())
                self.metric_accumulator['Whole_evaluate_last_dis'].extend(
                torch.atleast_1d(last_pose_dis).cpu().numpy())
                if self.use_traversability:
                    self.metric_accumulator['Whole_dtg_traversability_rate'].extend(dtg_traversability_rates)
                    self.metric_accumulator['Whole_true_dtg_traversability_rate'].extend(true_dtg_traversability_rates)
                # # 전체 샘플 수를 기준으로 indices 값을 업데이트
            
            # indices += len(y_hat_poses)       
            # if self.output_dir is not None:
            #         all_trajectories = input_dict[DataDict.all_trajectories]
            #         local_map = input_dict[DataDict.local_map]
            #         for idx in range(len(y_hat_poses)):
            #             self.show_path_local_map(trajectory=y_hat_poses[idx].detach().cpu().numpy(),
            #                                     gt_path=ygt[idx].detach().cpu().numpy(),
            #                                     local_map=local_map[idx].detach().cpu().numpy(), idx=idx, indices=indices)
            #             if self.train_poses:
            #                 temp_all_trajectories = [t_hat[idx] * self.scale_waypoints for t_hat in all_trajectories]
            #             else:
            #                 temp_all_trajectories = [np.cumsum(t_hat[idx], axis=0) * self.scale_waypoints for t_hat in all_trajectories]
            #             # for t_idx in range(len(temp_all_trajectories)):
            #             #     self.show_path_local_map(trajectory=temp_all_trajectories[t_idx], gt_path=ygt[idx].detach().cpu().numpy(),
            #             #                             local_map=local_map[idx].detach().cpu().numpy(), idx=t_idx, indices=indices, flag="syn")
            #             indices += 1
            return output, indices


def write_png(local_map=None, rgb_local_map=None, center=None, targets=None, paths=None, paths_color=None, path=None,
              crop_edge=None, others=None, file=None):
    dis = 2
    x_range = [local_map.shape[0], 0]
    y_range = [local_map.shape[1], 0]
    if rgb_local_map is not None:
        local_map_fig = rgb_local_map
    else:
        local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
    if center is not None:
        assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(center + np.array([x, y]))
        all_points = np.stack(all_points).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if targets is not None and len(targets) > 0:
        xs, ys = targets[:, 0], targets[:, 1]
        xs = np.clip(xs, dis, local_map_fig.shape[0] - dis)
        ys = np.clip(ys, dis, local_map_fig.shape[1] - dis)
        clipped_targets = np.stack((xs, ys), axis=-1)

        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(clipped_targets + np.array([x, y]))
        if len(clipped_targets.shape) == 2:
            all_points = np.concatenate(all_points, axis=0).astype(int)
        else:
            all_points = np.stack(all_points, axis=0).astype(int)

        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if others is not None:
        assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(others + np.array([x, y]))
        all_points = np.concatenate(all_points, axis=0).astype(int)

        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 255
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 0

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if path is not None:
        assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
        all_pts = path
        all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                  all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                  all_pts), axis=0)
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 0
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 255

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if paths is not None:
        for p_idx in range(len(paths)):
            path = paths[p_idx]
            if len(path) == 1 or np.any(path[0] == np.inf):
                continue
            path = np.asarray(path, dtype=int)
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            xs, ys = all_pts[:, 0], all_pts[:, 1]
            xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
            ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
            if paths_color is not None:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 0
                local_map_fig[xs, ys, 2] = paths_color[p_idx]
            else:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 255
                local_map_fig[xs, ys, 2] = 255

            if x_range[0] > min(all_pts[:, 0]):
                x_range[0] = min(all_pts[:, 0])
            if x_range[1] < max(all_pts[:, 0]):
                x_range[1] = max(all_pts[:, 0])
            if y_range[0] > min(all_pts[:, 1]):
                y_range[0] = min(all_pts[:, 1])
            if y_range[1] < max(all_pts[:, 1]):
                y_range[1] = max(all_pts[:, 1])
    if crop_edge:
        local_map_fig = local_map_fig[
                        max(0, x_range[0] - crop_edge):min(x_range[1] + crop_edge, local_map_fig.shape[0]),
                        max(0, y_range[0] - crop_edge):min(y_range[1] + crop_edge, local_map_fig.shape[1])]
    if file is not None:
        cv2.imwrite(file, local_map_fig)
    return local_map_fig