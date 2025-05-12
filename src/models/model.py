import torch
from torch import nn
from src.models.perception import Perception, LidarImageModel
from src.models.vae import CVAE
from src.models.diffusion import Diffusion

from src.utils.configs import DataDict, GeneratorType

class HNav(nn.Module):
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device
        
        self.generator_type = config.generator_type
        
        # 다양한 모델 컴포넌트 초기화
        if self.generator_type == GeneratorType.cvae:
            self.perception = Perception(self.config.perception)
            self.generator = CVAE(self.config.cvae)
            self.diffusion = None  # CVAE 모드에서는 diffusion 없음
        elif self.generator_type == GeneratorType.diffusion:
            self.perception = Perception(self.config.perception)
            self.generator = Diffusion(self.config.diffusion)
            # diffusion 속성을 generator와 동일하게 설정
            self.diffusion = self.generator
        else:
            raise ValueError("the generator type is not defined")
    
    def forward(self, input_dict, sample=False):
        if sample:
            return self.sample(input_dict=input_dict)
        else:
            output = {DataDict.path: input_dict[DataDict.path],
                      DataDict.heuristic: input_dict[DataDict.heuristic],
                      DataDict.local_map: input_dict[DataDict.local_map]}
            
            observation = self.perception(lidar=input_dict[DataDict.lidar], vel=input_dict[DataDict.vel],
                                          target=input_dict[DataDict.target])
            
            # traversable_step이 input_dict에 있는 경우에만 전달
            traversable_step = input_dict.get(DataDict.traversable_step, None)
            
            generator_output = self.generator(observation=observation, 
                                              gt_path=input_dict[DataDict.path],
                                              traversable_step=traversable_step)
            output.update(generator_output)
            return output
    
    def sample(self, input_dict):
        output = {}
        if DataDict.path in input_dict.keys():
            output.update({DataDict.path: input_dict[DataDict.path]})
        if DataDict.heuristic in input_dict.keys():
            output.update({DataDict.heuristic: input_dict[DataDict.heuristic]})
        if DataDict.local_map in input_dict.keys():
            output.update({DataDict.local_map: input_dict[DataDict.local_map]})
        observation = self.perception(lidar=input_dict[DataDict.lidar], vel=input_dict[DataDict.vel],
                                      target=input_dict[DataDict.target])
        generator_output = self.generator.sample(observation=observation)
        output.update(generator_output)
        return output
    
    # diffusion 객체에 대한 안전한 접근을 위한 getter 메서드
    def get_diffusion_component(self):
        """
        모델의 diffusion 컴포넌트를 안전하게 반환하는 메서드
        
        Returns:
            diffusion 컴포넌트 또는 None
        """
        return self.diffusion

def get_model(config, device):
    return HNav(config=config, device=device)