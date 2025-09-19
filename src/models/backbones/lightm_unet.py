# src/models/backbones/lightm_unet.py
import torch
from torch import nn
from typing import Union

# 프로젝트의 임베딩을 그대로 재사용해 호환성 유지
from src.models.backbones.unet import SinusoidalPosEmb

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


class AdaGN(nn.Module):
    def __init__(self, num_channels, n_groups, cond_dim, t_dim, film_mode: str = "full"):
        super().__init__()
        # film_mode: "full"|"t_only"|"g_only"|"none"
        assert film_mode in ("full", "t_only", "g_only", "none")
        self.film_mode = film_mode
        self.norm = nn.GroupNorm(n_groups, num_channels, eps=1e-6, affine=False)
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim + t_dim, num_channels * 2),  # scale, shift
        )

    def forward(self, x, t_emb, g_emb):
        if self.film_mode == "none":
            # When completely off, only perform normalization without FiLM (leave affine=False as is)
            return self.norm(x)
        h = torch.cat([t_emb, g_emb], dim=-1)
        ss = self.affine(h).unsqueeze(-1)  # (B, 2C, 1)
        scale, shift = ss.chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


class MambaOrConv(nn.Module):
    def __init__(self, channels, d_state=16, expand=2, use_mamba: bool = None):
        super().__init__()
        # None follows the environment (HAS_MAMBA) and can be forced to True/False
        self.use_mamba = HAS_MAMBA if use_mamba is None else use_mamba
        if self.use_mamba:
            self.core = Mamba(d_model=channels, d_state=d_state, expand=expand)
        else:
            self.core = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv1d(channels, channels, 3, padding=1),
            )

    def forward(self, x):  # x: (B, C, T)
        if self.use_mamba:
            return self.core(x.transpose(1, 2)).transpose(1, 2)
        return self.core(x)

class PlainGN(nn.Module):
    """FiLM 소거 시 사용하는 표준 GroupNorm(affine=True)."""
    def __init__(self, num_channels, n_groups):
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, num_channels, eps=1e-6, affine=True)
    def forward(self, x, *_, **__):
        return self.norm(x)

class ResBlock(nn.Module):
    def __init__(self, channels, n_groups, cond_dim, t_dim, mamba_cfg, film_mode="full", use_mamba=None):
        super().__init__()
        # film_mode=="none"이면 PlainGN(affine=True) 사용
        Norm1 = PlainGN if film_mode == "none" else AdaGN
        self.adagn1 = Norm1(channels, n_groups) if film_mode == "none" \
                      else AdaGN(channels, n_groups, cond_dim, t_dim, film_mode)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.mamba = MambaOrConv(channels, use_mamba=use_mamba,**mamba_cfg)
        Norm2 = PlainGN if film_mode == "none" else AdaGN
        self.adagn2 = Norm2(channels, n_groups) if film_mode == "none" \
                      else AdaGN(channels, n_groups, cond_dim, t_dim, film_mode)        
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb, g_emb):
        h = self.conv1(self.act1(self.adagn1(x, t_emb, g_emb)))
        h = self.mamba(h)
        h = self.conv2(self.act2(self.adagn2(h, t_emb, g_emb)))
        return h + x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_groups, cond_dim, t_dim, mamba_cfg, do_down=True, film_mode="full", use_mamba=None):
        super().__init__()
        # 항상 in→out 채널 정규화 (스킵/다운 모두 out_ch 기준으로 동작)
        self.proj = nn.Conv1d(in_ch, out_ch, 1)
        self.res  = ResBlock(out_ch, n_groups, cond_dim, t_dim, mamba_cfg, film_mode=film_mode, use_mamba=use_mamba)
        self.down = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1) if do_down else nn.Identity()

    def forward(self, x, t_emb, g_emb):
        x = self.proj(x)
        x = self.res(x, t_emb, g_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_groups, cond_dim, t_dim, mamba_cfg, film_mode="full", use_mamba=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv1d(in_ch, out_ch, 1)
        self.fuse = nn.Conv1d(out_ch + skip_ch, out_ch, 1)
        self.res = ResBlock(out_ch, n_groups, cond_dim, t_dim, mamba_cfg, film_mode=film_mode, use_mamba=use_mamba)

    def forward(self, x, skip, t_emb, g_emb):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):  # odd/even 길이 보정
            diff = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, (0, diff))
        x = self.reduce(x)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.res(x, t_emb, g_emb)
        return x


class ConditionalLightMUNet1D(nn.Module):
    """
    Drop-in 교체:
      - forward(x, timestep, local_cond=None, global_cond=None)
      - x in:  (B, T, C) 또는 (B, C, T)
      - x out: (B, T, C)  # prediction_type="sample"
    """
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int,
        down_dims=(64, 128, 256),
        kernel_size=3,
        n_groups=8,
        mamba_d_state=16,
        mamba_expand=2,
        cond_predict_scale=False,
        output_threshold=None,
        film_mode: str = "full", # NEW: "full" | "t_only" | "g_only" | "none"
        use_mamba: bool = None, # NEW: None=auto(HAS_MAMBA), True/False => Forced
    ):
        super().__init__()
        assert len(down_dims) >= 2
        self.in_dim = input_dim
        self.gc_dim = global_cond_dim
        self.t_dim = diffusion_step_embed_dim
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        self.output_threshold = output_threshold
        self.film_mode = film_mode
        self.use_mamba = use_mamba

        mcfg = dict(d_state=mamba_d_state, expand=mamba_expand)

        # in/out
        self.proj_in = nn.Conv1d(input_dim, down_dims[0], kernel_size, padding=kernel_size // 2)
        self.proj_out = nn.Sequential(
            nn.GroupNorm(n_groups, down_dims[0]),
            nn.SiLU(),
            nn.Conv1d(down_dims[0], input_dim, kernel_size, padding=kernel_size // 2),
        )
        if self.cond_predict_scale:
            self.scale_head = nn.Sequential(
                nn.GroupNorm(n_groups, down_dims[0]),
                nn.SiLU(),
                nn.Conv1d(down_dims[0], input_dim, 1),
            )

        # t-embedding: 프로젝트와 동일 파이프라인
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(self.t_dim),
            nn.Linear(self.t_dim, self.t_dim * 4), nn.Mish(),
            nn.Linear(self.t_dim * 4, self.t_dim),
        )

        # down path: 각 스테이지마다 채널 증가
        self.downs = nn.ModuleList([
            DownBlock(down_dims[i], down_dims[i+1], n_groups, global_cond_dim, self.t_dim, mcfg, do_down=True,
                      film_mode=self.film_mode, use_mamba=self.use_mamba)
            for i in range(len(down_dims) - 1)
        ])

        # bottom: 이미 last dim으로 내려왔으므로 바로 mid
        self.mid = ResBlock(down_dims[-1], n_groups, global_cond_dim, self.t_dim, mcfg, 
                            film_mode=self.film_mode, use_mamba=self.use_mamba)
        # up path: bottom에서 시작 (cur=down_dims[-1]), skip도 같은 레벨 채널
        self.ups = nn.ModuleList([
            UpBlock(in_ch=down_dims[i+1], skip_ch=down_dims[i+1], out_ch=down_dims[i],
                    n_groups=n_groups, cond_dim=global_cond_dim, t_dim=self.t_dim, mamba_cfg=mcfg, 
                    film_mode=self.film_mode, use_mamba=self.use_mamba)
            for i in reversed(range(len(down_dims) - 1))
        ])

    @staticmethod
    def _canon_timestep(timestep: Union[torch.Tensor, float, int], batch: int, device):
        # unet.py와 동일한 브로드캐스팅 규칙
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif torch.is_tensor(timestep) and timestep.dim() == 0:
            timestep = timestep[None].to(device)
        return timestep.expand(batch)

    def forward(self, x, timestep: Union[torch.Tensor, float, int], local_cond=None, global_cond=None):
        # 입력 정규화: (B, T, C) 또는 (B, C, T) 모두 허용
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape}")
        # 더 안전한 판정: 마지막 축이 feature 채널(C)이면 (B,T,C)
        if x.shape[-1] == self.in_dim:        # (B, T, C)
            x = x.transpose(1, 2)             # -> (B, C, T)
        elif x.shape[1] == self.in_dim:       # 이미 (B, C, T)
            pass
        else:
            raise ValueError(
                f"Unexpected shape {x.shape}; expected (B,T,{self.in_dim}) or (B,{self.in_dim},T)"
            )
        B, _, _ = x.shape
        device, dtype = x.device, x.dtype

        # timestep을 (B,)로 정규화
        timestep = self._canon_timestep(timestep, B, device)

        # 임베딩
        t_emb = self.t_embed(timestep)                 # (B, t_dim)
        g_emb = global_cond if global_cond is not None else torch.zeros(B, self.gc_dim, device=device, dtype=dtype)

        # 다운 경로
        h = self.proj_in(x)
        skips = []
        for down in self.downs:
            h, s = down(h, t_emb, g_emb)
            skips.append(s)

        # 보텀
        h = self.mid(h, t_emb, g_emb)

        # 업 경로
        for up in self.ups:
            s = skips.pop()
            h = up(h, s, t_emb, g_emb)

        # 출력
        out_feat = h
        x0 = self.proj_out(out_feat)
        if self.cond_predict_scale:
            scale = torch.tanh(self.scale_head(out_feat))
            x0 = x0 * (1 + scale)

        if self.output_threshold is not None:
            x0 = torch.clamp(x0, -self.output_threshold, self.output_threshold)

        return x0.transpose(1, 2)  # (B, T, C)


####
# # src/models/backbones/lightm_unet.py
# import torch
# from torch import nn
# from typing import Union

# # 프로젝트의 임베딩을 그대로 재사용해 호환성 유지
# from src.models.backbones.unet import SinusoidalPosEmb

# try:
#     from mamba_ssm import Mamba
#     HAS_MAMBA = True
# except Exception:
#     HAS_MAMBA = False


# class AdaGN(nn.Module):
#     def __init__(self, num_channels, n_groups, cond_dim, t_dim):
#         super().__init__()
#         self.norm = nn.GroupNorm(n_groups, num_channels, eps=1e-6, affine=False)
#         self.affine = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(cond_dim + t_dim, num_channels * 2),  # scale, shift
#         )

#     def forward(self, x, t_emb, g_emb):
#         h = torch.cat([t_emb, g_emb], dim=-1)
#         ss = self.affine(h).unsqueeze(-1)  # (B, 2C, 1)
#         scale, shift = ss.chunk(2, dim=1)
#         return self.norm(x) * (1 + scale) + shift


# class MambaOrConv(nn.Module):
#     def __init__(self, channels, d_state=16, expand=2):
#         super().__init__()
#         self.use_mamba = HAS_MAMBA
#         if self.use_mamba:
#             self.core = Mamba(d_model=channels, d_state=d_state, expand=expand)
#         else:
#             self.core = nn.Sequential(
#                 nn.Conv1d(channels, channels, 3, padding=1),
#                 nn.SiLU(),
#                 nn.Conv1d(channels, channels, 3, padding=1),
#             )

#     def forward(self, x):  # x: (B, C, T)
#         if self.use_mamba:
#             return self.core(x.transpose(1, 2)).transpose(1, 2)
#         return self.core(x)


# class ResBlock(nn.Module):
#     def __init__(self, channels, n_groups, cond_dim, t_dim, mamba_cfg):
#         super().__init__()
#         self.adagn1 = AdaGN(channels, n_groups, cond_dim, t_dim)
#         self.act1 = nn.SiLU()
#         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
#         self.mamba = MambaOrConv(channels, **mamba_cfg)
#         self.adagn2 = AdaGN(channels, n_groups, cond_dim, t_dim)
#         self.act2 = nn.SiLU()
#         self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

#     def forward(self, x, t_emb, g_emb):
#         h = self.conv1(self.act1(self.adagn1(x, t_emb, g_emb)))
#         h = self.mamba(h)
#         h = self.conv2(self.act2(self.adagn2(h, t_emb, g_emb)))
#         return h + x


# class DownBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, n_groups, cond_dim, t_dim, mamba_cfg, do_down=True):
#         super().__init__()
#         # 항상 in→out 채널 정규화 (스킵/다운 모두 out_ch 기준으로 동작)
#         self.proj = nn.Conv1d(in_ch, out_ch, 1)
#         self.res  = ResBlock(out_ch, n_groups, cond_dim, t_dim, mamba_cfg)
#         self.down = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1) if do_down else nn.Identity()

#     def forward(self, x, t_emb, g_emb):
#         x = self.proj(x)
#         x = self.res(x, t_emb, g_emb)
#         skip = x
#         x = self.down(x)
#         return x, skip


# class UpBlock(nn.Module):
#     def __init__(self, in_ch, skip_ch, out_ch, n_groups, cond_dim, t_dim, mamba_cfg):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#         self.reduce = nn.Conv1d(in_ch, out_ch, 1)
#         self.fuse = nn.Conv1d(out_ch + skip_ch, out_ch, 1)
#         self.res = ResBlock(out_ch, n_groups, cond_dim, t_dim, mamba_cfg)

#     def forward(self, x, skip, t_emb, g_emb):
#         x = self.up(x)
#         if x.size(-1) != skip.size(-1):  # odd/even 길이 보정
#             diff = skip.size(-1) - x.size(-1)
#             x = nn.functional.pad(x, (0, diff))
#         x = self.reduce(x)
#         x = self.fuse(torch.cat([x, skip], dim=1))
#         x = self.res(x, t_emb, g_emb)
#         return x


# class ConditionalLightMUNet1D(nn.Module):
#     """
#     Drop-in 교체:
#       - forward(x, timestep, local_cond=None, global_cond=None)
#       - x in:  (B, T, C) 또는 (B, C, T)
#       - x out: (B, T, C)  # prediction_type="sample"
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         global_cond_dim: int,
#         diffusion_step_embed_dim: int,
#         down_dims=(64, 128, 256),
#         kernel_size=3,
#         n_groups=8,
#         mamba_d_state=16,
#         mamba_expand=2,
#         cond_predict_scale=False,
#         output_threshold=None,
#     ):
#         super().__init__()
#         assert len(down_dims) >= 2
#         self.in_dim = input_dim
#         self.gc_dim = global_cond_dim
#         self.t_dim = diffusion_step_embed_dim
#         self.n_groups = n_groups
#         self.cond_predict_scale = cond_predict_scale
#         self.output_threshold = output_threshold

#         mcfg = dict(d_state=mamba_d_state, expand=mamba_expand)

#         # in/out
#         self.proj_in = nn.Conv1d(input_dim, down_dims[0], kernel_size, padding=kernel_size // 2)
#         self.proj_out = nn.Sequential(
#             nn.GroupNorm(n_groups, down_dims[0]),
#             nn.SiLU(),
#             nn.Conv1d(down_dims[0], input_dim, kernel_size, padding=kernel_size // 2),
#         )
#         if self.cond_predict_scale:
#             self.scale_head = nn.Sequential(
#                 nn.GroupNorm(n_groups, down_dims[0]),
#                 nn.SiLU(),
#                 nn.Conv1d(down_dims[0], input_dim, 1),
#             )

#         # t-embedding: 프로젝트와 동일 파이프라인
#         self.t_embed = nn.Sequential(
#             SinusoidalPosEmb(self.t_dim),
#             nn.Linear(self.t_dim, self.t_dim * 4), nn.Mish(),
#             nn.Linear(self.t_dim * 4, self.t_dim),
#         )

#         # down path: 각 스테이지마다 채널 증가
#         self.downs = nn.ModuleList([
#             DownBlock(down_dims[i], down_dims[i+1], n_groups, global_cond_dim, self.t_dim, mcfg, do_down=True)
#             for i in range(len(down_dims) - 1)
#         ])

#         # bottom: 이미 last dim으로 내려왔으므로 바로 mid
#         self.mid = ResBlock(down_dims[-1], n_groups, global_cond_dim, self.t_dim, mcfg)
#         # up path: bottom에서 시작 (cur=down_dims[-1]), skip도 같은 레벨 채널
#         self.ups = nn.ModuleList([
#             UpBlock(in_ch=down_dims[i+1], skip_ch=down_dims[i+1], out_ch=down_dims[i],
#                     n_groups=n_groups, cond_dim=global_cond_dim, t_dim=self.t_dim, mamba_cfg=mcfg)
#             for i in reversed(range(len(down_dims) - 1))
#         ])

#     @staticmethod
#     def _canon_timestep(timestep: Union[torch.Tensor, float, int], batch: int, device):
#         # unet.py와 동일한 브로드캐스팅 규칙
#         if not torch.is_tensor(timestep):
#             timestep = torch.tensor([timestep], dtype=torch.long, device=device)
#         elif torch.is_tensor(timestep) and timestep.dim() == 0:
#             timestep = timestep[None].to(device)
#         return timestep.expand(batch)

#     def forward(self, x, timestep: Union[torch.Tensor, float, int], local_cond=None, global_cond=None):
#         # 입력 정규화: (B, T, C) 또는 (B, C, T) 모두 허용
#         if x.dim() != 3:
#             raise ValueError(f"Expected 3D tensor, got {x.shape}")
#         # 더 안전한 판정: 마지막 축이 feature 채널(C)이면 (B,T,C)
#         if x.shape[-1] == self.in_dim:        # (B, T, C)
#             x = x.transpose(1, 2)             # -> (B, C, T)
#         elif x.shape[1] == self.in_dim:       # 이미 (B, C, T)
#             pass
#         else:
#             raise ValueError(
#                 f"Unexpected shape {x.shape}; expected (B,T,{self.in_dim}) or (B,{self.in_dim},T)"
#             )
#         B, _, _ = x.shape
#         device, dtype = x.device, x.dtype

#         # timestep을 (B,)로 정규화
#         timestep = self._canon_timestep(timestep, B, device)

#         # 임베딩
#         t_emb = self.t_embed(timestep)                 # (B, t_dim)
#         g_emb = global_cond if global_cond is not None else torch.zeros(B, self.gc_dim, device=device, dtype=dtype)

#         # 다운 경로
#         h = self.proj_in(x)
#         skips = []
#         for down in self.downs:
#             h, s = down(h, t_emb, g_emb)
#             skips.append(s)

#         # 보텀
#         h = self.mid(h, t_emb, g_emb)

#         # 업 경로
#         for up in self.ups:
#             s = skips.pop()
#             h = up(h, s, t_emb, g_emb)

#         # 출력
#         out_feat = h
#         x0 = self.proj_out(out_feat)
#         if self.cond_predict_scale:
#             scale = torch.tanh(self.scale_head(out_feat))
#             x0 = x0 * (1 + scale)

#         if self.output_threshold is not None:
#             x0 = torch.clamp(x0, -self.output_threshold, self.output_threshold)

#         return x0.transpose(1, 2)  # (B, T, C)
