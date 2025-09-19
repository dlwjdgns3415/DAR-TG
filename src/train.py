import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess
import numpy as np

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
        self.global_evaluation_step_counter = 0

        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
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
            self._force_move_model_to_device()
            self._assert_model_on_device()

        self.current_rank = 0
        if self.device.type == "cpu":
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
        wandb.define_metric("evaluation/global_eval_step")
        wandb.define_metric("evaluation/*", step_metric="evaluation/global_eval_step", summary="mean")

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
    def _assert_model_on_device(self):
        """모델의 모든 파라미터/버퍼가 self.device에 있는지 점검하고, 아니면 상세히 알려줍니다."""
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        wrong = []
        for name, p in self.model.named_parameters(recurse=True):
            if p.device.type != dev.type:
                wrong.append(("param", name, str(p.device)))
        for name, b in self.model.named_buffers(recurse=True):
            if b.device.type != dev.type:
                wrong.append(("buffer", name, str(b.device)))
        if wrong:
            msg = "\n".join([f"[{k}] {n}: {d}" for k, n, d in wrong[:20]])
            raise RuntimeError(
                f"Some tensors are NOT on {dev}.\n{msg}\n"
                f"(showing up to 20; move them or call self._force_move_model_to_device())"
            )
    def _force_move_model_to_device(self):
        """모델 전체(파라미터+버퍼)를 self.device로 강제로 이동"""
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        self.model.to(dev)

    def _set_model_gpus(self, cfg_gpus):
        if self.distributed:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")

            if not dist.is_initialized():
                 dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            
            self.current_rank = dist.get_rank()
            if self.current_rank == 0:
                print(f'DDP Mode: Process {self.current_rank} of {dist.get_world_size()} on GPU {local_rank}.')
            
            self.model = self.model.to(local_rank)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank,
                             broadcast_buffers=not cfg_gpus.get('no_ddp_bb', True), 
                             find_unused_parameters=cfg_gpus.get('find_unused_parameters', False))
            
            if cfg_gpus.get('sync_bn', False):
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                if self.current_rank == 0: print('Converted model to use Synchronized BatchNorm.')

            else:
                if isinstance(self.device, str):
                    self.device = torch.device(self.device)
                if self.device.type == 'cuda':
                    torch.cuda.set_device(self.device)
                self.model = self.model.to(self.device)    
        if cfg_gpus.get('channels_last', False) and getattr(self.device, "type", "cpu") == 'cuda':     
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                if not self.distributed or self.current_rank == 0: print("Model converted to channels_last memory format.")
            except Exception as e:
                if not self.distributed or self.current_rank == 0: print(f"Failed to convert model to channels_last: {e}")
        self._force_move_model_to_device()
        self._assert_model_on_device()

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        map_dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        state_dict = torch.load(snapshot, map_location=map_dev)

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
        if 'global_evaluation_step_counter' in state_dict: # 변수명 일치
            self.global_evaluation_step_counter = state_dict['global_evaluation_step_counter']
            if not self.distributed or self.current_rank == 0:
                print(f'Global evaluation step counter has been loaded: {self.global_evaluation_step_counter}.')
        
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

    def step(self, data_dict, indices=0, train=True) -> Tuple[dict, dict]:
        """
        모델의 한 스텝을 실행하는 메서드. 경로 위험도가 높을 경우 재시도.
        
        Args:
            data_dict: 입력 데이터 딕셔너리
            train: 훈련 모드 여부 (기본값: True)
            
        Returns:
            output_dict: 모델 출력 및 손실 딕셔너리
        """
        # 데이터를 현재 디바이스로 이동
        data_dict = to_device(data_dict, device=self.device)
        
        self._assert_model_on_device()

        if DataDict.lidar in data_dict:
            x_dev = data_dict[DataDict.lidar].device.type
            m_dev = next(self.model.parameters()).device.type
            if x_dev != m_dev:
                raise RuntimeError(f"Input on {x_dev} but model on {m_dev}")
        if train:
            # 훈련 모드에서는 일반적인 처리 수행
            output_dict = self.model(data_dict, sample=False)
            torch.cuda.empty_cache()
            # 손실 계산
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)
            torch.cuda.empty_cache()

            return output_dict
        else:
            # 평가 모드에서는 안전한 경로 예측 수행
            inference_start_time = time.time()
            output_dict = self.model(data_dict, sample=True)
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                torch.cuda.synchronize
            current_inference_time = time.time() - inference_start_time
            torch.cuda.empty_cache()
            # 평가 메트릭 계산
            output_dict.update({'inference_time': current_inference_time})
            eval_dict, updated_indices = self.loss_func.evaluate(output_dict, indices=indices)
            output_dict.update(eval_dict)
        
            return output_dict, updated_indices

    def update_log(self, results, timestep=None, log_name=None, custom_step=None):
        if not self.wandb_run: return

        log_payload = {}
        current_prefix = ""
        if log_name:
            current_prefix = log_name + "/"

        if "sampling_info" in results and isinstance(results["sampling_info"], dict):
            sampling_info = results.get("sampling_info")
            if "used_steps" in sampling_info: log_payload[current_prefix + "diffusion_loop_used_steps"] = sampling_info["used_steps"]
            if "waypoints_num" in sampling_info: log_payload[current_prefix + "diffusion_loop_waypoints_num"] = sampling_info["waypoints_num"]

        serializable_results = self.filter_json_serializable(results)
        
        for key_obj, value in serializable_results.items():
            key_str = str(key_obj.name) if hasattr(key_obj, 'name') and isinstance(key_obj.name, str) else str(key_obj)

            if key_str in ["sampling_info", "per_sample_complexity_scores", "identifier_fn", "identifier_idx_in_pkl"]:
                continue
            
            final_log_key = current_prefix + key_str
            log_payload[final_log_key] = value

        if log_name == LogTypes.train:
            lr_key = LogNames.lr.name if hasattr(LogNames.lr, 'name') else str(LogNames.lr)
            if (current_prefix + lr_key) not in log_payload:
                lr_value = self.scheduler.get_last_lr()
                log_payload[current_prefix + lr_key] = lr_value[-1]

        if timestep is not None:
            step_time_key = LogNames.step_time.name if hasattr(LogNames.step_time, 'name') else str(LogNames.step_time)
            log_payload[current_prefix + step_time_key] = timestep
        
        if not log_payload: return

        if custom_step is not None:
            step_payload = {"evaluation/global_eval_step": custom_step} if log_name != LogTypes.train else {"train/iteration_step": custom_step}
            self.wandb_run.log({**log_payload, **step_payload})
        else:
            self.wandb_run.log(log_payload)


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
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):
            self.loss_func.start_accumulation()
            global_image_indices  = 0
        # if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0):
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                # if iteration % self.max_evaluation_iteration_per_epoch == 0 and iteration != 0:
                #     break
                current_wandb_eval_step = self.global_evaluation_step_counter + iteration + 1

                start_time = time.time()
                output_dict,updated_image_indices = self.step(data_dict, train=False, indices=global_image_indices)
                global_image_indices = updated_image_indices

                if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                    torch.cuda.synchronize
                step_time = time.time()
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(results=output_dict, timestep=step_time - start_time, 
                                log_name=LogTypes.others, custom_step = current_wandb_eval_step)
            
            final_metrics_report = self.loss_func.get_final_metrics()
                
            print(f"\n--- [ Full Evaluation Summary | Epoch: {self.epoch} ] ---")
            for key, value in final_metrics_report.items():
                print(f"{key:40s}: {value:.4f}")
            print("="*60)
            self.update_log(
                    results=final_metrics_report,
                    log_name="evaluation", # "evaluation/..." 이름으로 wandb에 기록
                    custom_step=self.global_evaluation_step_counter 
                )
            # 리스트에 점수가 수집되었는지 확인
        # 에폭 종료 후 누적 평가 스텝 수 업데이트
        self.global_evaluation_step_counter += len(self.evaluation_data_loader)
                                                   
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

    def filter_json_serializable(self, obj):
        import threading 
        import json
        # import torch # 가정: Trainer 상단에 이미 임포트됨
        # import numpy as np # 가정: Trainer 상단에 이미 임포트됨

        if obj is None or isinstance(obj, (bool, int, float, str)): return obj
        
        if isinstance(obj, torch.cuda.Event):
            try:
                return f"cuda_event_queried_{obj.query()}" 
            except Exception: 
                return "cuda_event_unqueried"
        
        if isinstance(obj, threading.Event): # threading.Event도 처리
            return obj.is_set()

        if isinstance(obj, list): return [self.filter_json_serializable(item) for item in obj]
        
        if isinstance(obj, dict):
            result = {}
            for k, v_obj in obj.items(): 
                result[k] = self.filter_json_serializable(v_obj) # 값에 대해 재귀 호출
            return result
        
        # torch.Tensor 또는 np.ndarray 처리 부분 강화
        is_tensor = isinstance(obj, torch.Tensor)
        is_numpy = isinstance(obj, np.ndarray)

        if is_tensor or is_numpy:
            if obj.ndim == 0: # 0차원 텐서/배열 (스칼라와 유사)
                return self.filter_json_serializable(obj.item()) # item() 호출 안전
            elif obj.size == 1: # 요소가 하나만 있는 경우 (예: shape [1], [1,1] 등)
                return self.filter_json_serializable(obj.item()) # item() 호출 안전
            elif obj.ndim == 1 and obj.shape[0] <= 20: # 1차원이고 길이가 적당하면 리스트로 변환
                 return [self.filter_json_serializable(el.item() if hasattr(el, 'item') else el) for el in obj.tolist()]
            else: # 그 외 다차원 텐서/배열 또는 긴 1차원 배열
                return f"{'tensor' if is_tensor else 'numpy_array'}_shape_{list(obj.shape)}"

        if hasattr(obj, 'tolist') and callable(obj.tolist): # 일반 객체의 tolist (예: Pandas Series 등)
            try:
                list_repr = obj.tolist()
                # 너무 큰 리스트는 로깅하지 않도록 제한 (위의 Tensor/NumPy 처리와 유사하게)
                if len(list_repr) > 10 and (isinstance(list_repr[0], (list, np.ndarray, torch.Tensor)) or len(list_repr) > 20):
                    return f"general_list_too_large_len_{len(list_repr)}_type_{type(obj)}"
                return self.filter_json_serializable(list_repr)
            except Exception:
                return f"general_object_tolist_unserializable_type_{type(obj)}"
        
        # 최후의 수단: JSON 직렬화 시도 또는 문자열 변환
        try: 
            json.dumps(obj)
            return obj # JSON 직렬화 가능하면 원본 반환 (단, 내부 요소는 이미 처리되었어야 함)
        except (TypeError, OverflowError, RuntimeError):
            try: 
                return str(obj)
            except Exception: 
                return f"unserializable_object_type_{type(obj)}"