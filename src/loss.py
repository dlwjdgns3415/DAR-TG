import copy
import math
import os
import pickle
import shutil
from os.path import join, exists

import cv2
import imageio
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from src.models.diff_hausdorf import HausdorffLoss
from src.utils.configs import GeneratorType, DataDict, Hausdorff, LossNames


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        # with open(join(cfg.root, "data.pkl"), "rb") as input_file:
        #     data = pickle.load(input_file)
        #     self.network = data[DataDict.network]

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
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]

        output = {}

        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints
        if self.use_traversability:
            B, _, _ = y_hat.shape
            traversability_hat_poses = y_hat_poses[int(B / 2):]
            y_hat_poses = y_hat_poses[:int(B / 2)]

        # 기존 손실 계산 (거리 기반 손실 L_d)
        path_dis = self.distance(ygt, y_hat_poses).mean()
        last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
        distance_loss = self.distance_ratio * path_dis + self.last_ratio * last_pose_dis
        
        output.update({
            LossNames.last_dis: last_pose_dis,
            LossNames.path_dis: path_dis,
        })

        # Traversability 손실 계산 (L_t)
        traversability_loss_mean = torch.tensor(0.0, device=y_hat.device)
        
        if self.use_traversability:
            local_map = input_dict[DataDict.local_map]
            traversability_loss, traversability_values = self._local_collision(yhat=traversability_hat_poses, local_map=local_map)
            traversability_loss_mean = traversability_loss.to(float).mean()
            output.update({LossNames.traversability: traversability_loss_mean})
        
        # Path Risk Score를 가져와 손실 가중치 β 계산
        beta = torch.tensor(0.0, device=y_hat.device)
        
        if "path_risk_score" in input_dict:
            path_risk = input_dict["path_risk_score"]

            if isinstance(path_risk, np.ndarray) or isinstance(path_risk, list):
                path_risk = torch.tensor(path_risk, dtype=torch.float32)

            if not isinstance(path_risk, torch.Tensor):
                path_risk = torch.tensor(path_risk, dtype=torch.float32)

            path_risk = path_risk.to(y_hat.device)
            
            # 시그모이드 함수를 사용한 β 계산 (0과 1 사이의 값)
            # 부드럽고 비선형적인 가중치 조정을 위해 시그모이드 함수 사용
            # 논문 방식: β = σ(P_risk)
            beta = torch.sigmoid(path_risk)
            
            # 로깅을 위해 path_risk 값 저장
            output.update({LossNames.path_risk: path_risk})
        
        # 적응적 가중치를 이용한 최종 손실 계산
        # L = (1 - β) * L_d + β * L_t
        traversability_term = self.traversability_ratio * traversability_loss_mean
        all_loss = (1 - beta) * distance_loss + beta * traversability_term
        
        output.update({LossNames.loss: all_loss})
        return output

    def forward(self, input_dict):
        if self.generator_type == GeneratorType.cvae:
            return self.forward_cvae(input_dict=input_dict)
        elif self.generator_type == GeneratorType.diffusion:
            return self.forward_diffusion(input_dict=input_dict)

    def convert_path_pixel(self, trajectory):
        return np.clip(np.around(trajectory / self.map_resolution)[:, :2] + self.map_range, 0, np.inf)

    def show_path_local_map(self, trajectory, gt_path, local_map, idx=0, indices=0):
        return write_png(local_map=local_map, center=np.array([local_map.shape[0] / 2, local_map.shape[1] / 2]),
                         file=join(self.output_dir, "local_map_trajectory_{}.png".format(indices + idx)),
                         paths=[self.convert_path_pixel(trajectory=trajectory)],
                         others=self.convert_path_pixel(trajectory=gt_path)
                         )

    @torch.no_grad()
    def evaluate(self, input_dict, data_dict=None, indices=0):
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]
        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        if self.output_dir is not None:
            all_trajectories = input_dict[DataDict.all_trajectories]
            local_map = input_dict[DataDict.local_map]
            for idx in range(len(y_hat_poses)):
                self.show_path_local_map(trajectory=y_hat_poses[idx].detach().cpu().numpy(),
                                        gt_path=ygt[idx].detach().cpu().numpy(),
                                        local_map=local_map[idx].detach().cpu().numpy(), idx=idx, indices=indices)
                if self.train_poses:
                    temp_all_trajectories = [t_hat[idx] * self.scale_waypoints for t_hat in all_trajectories]
                else:
                    temp_all_trajectories = [np.cumsum(t_hat[idx], axis=0) * self.scale_waypoints for t_hat in all_trajectories]
                for t_idx in range(len(temp_all_trajectories)):
                    self.show_path_local_map(trajectory=temp_all_trajectories[t_idx], gt_path=ygt[idx].detach().cpu().numpy(),
                                            local_map=local_map[idx].detach().cpu().numpy(), idx=t_idx, indices=indices)

            path_dis = self.distance(ygt, y_hat_poses).mean()
            last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
            output = {
                LossNames.evaluate_last_dis: last_pose_dis,
                LossNames.evaluate_path_dis: path_dis,
            }

            if self.use_traversability:
                # 기존 traversability 손실 
                local_map = input_dict[DataDict.local_map]
                traversability_loss, traversability_values = self._local_collision(yhat=y_hat_poses, local_map=local_map)
                traversability_loss_mean = traversability_loss.mean()
                output.update({LossNames.evaluate_traversability: traversability_loss_mean})
                
                # ✅ Distance ratio 계산 (논문 수식 기반)
                # hr(τ̂) = 1 - |ht - hc|/(2|τ̂|)
                
                # 로봇의 현재 위치와 목표 위치 가져오기는 data_dict에서 수행
                if data_dict is not None and DataDict.pose in data_dict:
                    # 로봇의 현재 위치를 pose 정보에서 추출
                    robot_pose = data_dict[DataDict.pose]
                    # 변환 행렬에서 위치 부분(x, y)만 추출 - 마지막 열(인덱스 3)의 첫 두 행(인덱스 0,1)
                    robot_pos = robot_pose[:, :2, 3].to(y_hat_poses.device)  # 결과 shape: [batch_size, 2]
                else:
                    # pose 정보가 없는 경우 기본값 사용 (원점)
                    robot_pos = torch.zeros((y_hat_poses.shape[0], 2), device=y_hat_poses.device)
                
                # 목표 위치 가져오기
                if data_dict is not None and DataDict.target in data_dict:
                    # targets 형식: [coordinate_array, distance_value]
                    targets_data = data_dict[DataDict.target]
                    if isinstance(targets_data, list) and len(targets_data) > 0:
                        # 첫 번째 타겟에서 x, y 좌표만 추출
                        target_coord = targets_data[0][0][:2]  # 첫 번째 항목의 첫 번째 요소(좌표 배열)에서 x, y 좌표
                        goal_pos = torch.tensor([target_coord], device=y_hat_poses.device)
                    else:
                        goal_pos = ygt[:, -1, :2].to(y_hat_poses.device)
                else:
                    goal_pos = ygt[:, -1, :2].to(y_hat_poses.device)
                
                # 궤적 길이 계산 (|τ̂|)
                diffs = y_hat_poses[:, 1:] - y_hat_poses[:, :-1]
                segment_lengths = torch.norm(diffs, dim=2)
                trajectory_lengths = torch.sum(segment_lengths, dim=1)  # [B]
                
                # 로봇 위치에서 목표까지의 직선 거리 (hc)
                hc = torch.norm(goal_pos - robot_pos, dim=1)  # [B]
                
                # 궤적의 마지막 지점에서 목표까지의 직선 거리 (ht)
                last_waypoint = y_hat_poses[:, -1, :2]  # [B, 2]
                ht = torch.norm(goal_pos - last_waypoint, dim=1)  # [B]
                
                # 거리 비율 계산
                distance_ratio = 1.0 - torch.abs(ht - hc) / (2.0 * trajectory_lengths + 1e-6)
                output.update({"evaluate_distance_ratio": distance_ratio.mean()})
                
                # Traversability 점수 계산
                traversability_score = traversability_values.mean()
                output.update({"evaluate_traversability_score": traversability_score})
                
            return output
        
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