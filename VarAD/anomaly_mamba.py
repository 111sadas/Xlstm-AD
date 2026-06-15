# import numpy as np

# from .vssm import VSSM
# from .vision_tokenizer import VisionTokenizer
# from .dino_vision_tokenizer import VisionTokenizer as DINOVisionTokenizer
# from torch import nn
# from torch.nn import functional as F
# from scipy.ndimage import gaussian_filter
# import torch


# class AnomalyMamba(nn.Module):
#     def __init__(self,
#                 # tokenizer ----------------
#                 backbone,
#                 norm=False,
#                  hierarchies=[1,2,3],
#                 # VSSM --------------------
#                  depths=[2, 2],
#                  bos_length=4,
#                  valid_path=0,
#                  pred_next_n=8,
#                 # size ---------------------
#                  image_size=256,
#                  lambda_spatial=1.,

#                  learnable_bos=True,
#                  use_pe=True,
#                  is_proj=True,
#                  use_path_weights=True,
#                  cache_data=None,
#                  adapter=['linear']
#                  ):
#         super(AnomalyMamba, self).__init__()

#         self.image_size = image_size
#         self.norm = norm

#         if backbone.count('DINO') > 0 or backbone.count('dinov2') > 0:
#             self.tokenizer = DINOVisionTokenizer(backbone=backbone, hierarchies=hierarchies, norm=norm, adapter=adapter)
#         else:
#             self.tokenizer = VisionTokenizer(backbone=backbone, hierarchies=hierarchies,
#                                              norm=norm, is_proj=is_proj, image_size=image_size)

#         dims = self.tokenizer.target_dimension

#         x = self.tokenizer(cache_data)

#         self.vssm = nn.ModuleList()

#         n_blocks = 1
#         for indx, d in enumerate(dims):
#             vssm_temp = nn.Sequential(*[
#                 VSSM(depths=depths,
#                      dims=d,
#                      pred_next_n=pred_next_n // (2 ** indx),
#                      valid_path=valid_path,
#                     learnable_bos = learnable_bos,
#                     use_pe = use_pe,
#                      cache_data=x[indx],
#                      use_path_weights=use_path_weights,
#                      ) for _ in range(n_blocks)
#             ])
#             self.vssm.append(vssm_temp)

#         self.lambda_spatial = lambda_spatial


#     def train(self, mode: bool = True):
#         self.tokenizer.frozen_tokenizer.eval()
#         self.tokenizer.proj.train()
#         self.vssm.train()

#     def eval(self):
#         self.tokenizer.frozen_tokenizer.eval()
#         self.tokenizer.proj.eval()
#         self.vssm.eval()

#     @torch.cuda.amp.autocast()
#     def forward(self, x):
#         #patch_embed 是VIT出来的特征和经过MIP出来的特征残差连接的
#         patch_embed = self.tokenizer(x) # B, C, H, W
#         #predictive_embed 是经过mamba模型出来的特征
#         predictive_embed = []
#         for indx, patch in enumerate(patch_embed):
#             #把这里改成Xlstm
#             pred_embed = self.vssm[indx](patch)

#             if self.norm:
#                 pred_embed = F.normalize(pred_embed, dim=1)

#             predictive_embed.append(pred_embed)

#         return patch_embed, predictive_embed

#     # @torch.cuda.amp.autocast()
#     def _cal_loss(self, a, b):
#         # channel wise distillation
#         dist_channel = F.cosine_similarity(a, b, dim=1)

#         # spatial wise distillation
#         B, C, H, W = a.shape
#         spatial_a = a.view(B, C, H * W)
#         spatial_b = b.view(B, C, H * W)
#         dist_spatial = F.cosine_similarity(spatial_a, spatial_b, dim=2)

#         loss = (1 - dist_channel.mean()) + self.lambda_spatial * (1 - dist_spatial.mean()) # maximize similarities

#         loss = loss * a.shape[0]
#         return loss

#     def _cal_am(self, a, b):
#         dist = 1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)
#         anomaly_map = F.interpolate(dist, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
#         return anomaly_map

#     def cal_loss(self, gt, pred):
#         loss = 0
#         for a, b in zip(gt, pred):
#             loss += self._cal_loss(a, b)
#         # print(f'loss: {loss:.2f}')
#         return loss

#     @torch.no_grad()
#     def cal_am(self, gt, pred):

#         anomaly_map_list = []
#         for a, b in zip(gt, pred):
#             anomaly_map_list.append(self._cal_am(a, b))

#         anomaly_map = torch.mean(torch.cat(anomaly_map_list, dim=1), dim=1)
#         am_np = anomaly_map.squeeze(1).cpu().numpy()
#         am_np_list = []

#         for i in range(am_np.shape[0]):
#             am_np[i] = gaussian_filter(am_np[i], sigma=4)

#             if np.sum(np.isnan(am_np[i])) > 0:
#                 print(f'error: {np.sum(np.isnan(am_np[i]))} vaules are NaN')
#             am_np_list.append(am_np[i])

#         return am_np_list





import numpy as np

# from .vssm import VSSM
from .vision_tokenizer import VisionTokenizer
from .dino_vision_tokenizer import VisionTokenizer as DINOVisionTokenizer
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
import torch

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class AnomalyMamba(nn.Module):
    def __init__(self, backbone, norm=False, hierarchies=[1,2,3],
                 image_size=256, adapter=['linear'],lambda_spatial=1.0,  **kwargs):
        super().__init__()

        self.image_size = image_size
        self.norm = norm
        self.lambda_spatial = lambda_spatial

        # tokenizer
        if backbone.count('DINO') > 0 or backbone.count('dinov2') > 0:
            self.tokenizer = DINOVisionTokenizer(backbone=backbone, hierarchies=hierarchies, norm=norm, adapter=adapter)
        else:
            self.tokenizer = VisionTokenizer(backbone=backbone, hierarchies=hierarchies,
                                             norm=norm, is_proj=True, image_size=image_size)

        dims = self.tokenizer.target_dimension

        self.xlstm_stack = nn.ModuleList()
        for d in dims:  # dims = 每个尺度的 channel 数
            cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=8
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="cuda",
                        num_heads=8,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=1296,   # 先写死，后面 forward 时再动态替换
                num_blocks=8,
                embedding_dim=d,      # 每个尺度的 C
                slstm_at=[1],
            )
            self.xlstm_stack.append(xLSTMBlockStack(cfg))

    @torch.cuda.amp.autocast()
    def forward(self, x):
        patch_embed = self.tokenizer(x)   # list of feature maps (B,C,H,W)
        predictive_embed = []

        for indx, patch in enumerate(patch_embed):
            B, C, H, W = patch.shape
            # (B, C, H, W) → (B, L=H*W, D=C)
            patch_seq = patch.flatten(2).transpose(1, 2)

            pred_seq = self.xlstm_stack[indx](patch_seq)   # (B, L, D)
            
            

            # (B, L, D) → (B, C, H, W)
            pred_embed = pred_seq.transpose(1, 2).view(B, C, H, W)

            if self.norm:
                pred_embed = F.normalize(pred_embed, dim=1)

            predictive_embed.append(pred_embed)

        return patch_embed, predictive_embed


    # @torch.cuda.amp.autocast()
    def _cal_loss(self, a, b):
        # channel wise distillation
        dist_channel = F.cosine_similarity(a, b, dim=1)

        # spatial wise distillation
        B, C, H, W = a.shape
        spatial_a = a.view(B, C, H * W)
        spatial_b = b.view(B, C, H * W)
        dist_spatial = F.cosine_similarity(spatial_a, spatial_b, dim=2)

        loss = (1 - dist_channel.mean()) + self.lambda_spatial * (1 - dist_spatial.mean()) # maximize similarities

        loss = loss * a.shape[0]
        return loss

    def _cal_am(self, a, b):
        dist = 1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)
        anomaly_map = F.interpolate(dist, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        return anomaly_map

    def cal_loss(self, gt, pred):
        loss = 0
        for a, b in zip(gt, pred):
            loss += self._cal_loss(a, b)
        # print(f'loss: {loss:.2f}')
        return loss

    @torch.no_grad()
    def cal_am(self, gt, pred):

        anomaly_map_list = []
        for a, b in zip(gt, pred):
            anomaly_map_list.append(self._cal_am(a, b))

        anomaly_map = torch.mean(torch.cat(anomaly_map_list, dim=1), dim=1)
        am_np = anomaly_map.squeeze(1).cpu().numpy()
        am_np_list = []

        for i in range(am_np.shape[0]):
            am_np[i] = gaussian_filter(am_np[i], sigma=4)

            if np.sum(np.isnan(am_np[i])) > 0:
                print(f'error: {np.sum(np.isnan(am_np[i]))} vaules are NaN')
            am_np_list.append(am_np[i])

        return am_np_list

