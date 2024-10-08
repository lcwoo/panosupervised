import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from vidar.arch.networks.layers.panodepth.depth_sweeping import MultiDepthTransform
from vidar.arch.networks.layers.panodepth.confidence import AggWeightNetVolume


class CylindricalPad(nn.Module):
    #원통형 패딩을 적용하는 모듈
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')

def compute_volume_variance(grouped_panofeats, num_views):
    volume_sum = 0
    volume_sq_sum = 0
    
    # 각 뷰의 피처를 합산하여 volume_sum과 volume_sq_sum 계산
    for cam_feats in grouped_panofeats:
        panofeat_stack = torch.stack(cam_feats, dim=1)  # (B, num_cameras, C, H, W)
        panofeat_stack_transposed = panofeat_stack.transpose(1, 2)  # (B, C, num_cameras, H, W)
        
        # 피처 합산 및 제곱 값 계산
        volume_sum += panofeat_stack_transposed.sum(dim=2)  # (B, C, H, W)
        volume_sq_sum += (panofeat_stack_transposed ** 2).sum(dim=2)  # (B, C, H, W)
    
    # variance 계산
    mean = volume_sum / num_views
    mean_sq = volume_sq_sum / num_views
    variance = mean_sq - mean ** 2  # (B, C, H, W)

    # Variance 값을 클리핑 (너무 큰 값으로 인한 NaN 방지)
    variance_clamped = torch.clamp(variance, min=1e-8, max=10.0)

    return variance_clamped


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling_factor = d_model ** -0.5

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attention, _ = ScaledDotProductAttention(self.d_k)(query, key, value, mask)

        # Concatenate and apply the final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(attention)
        return output
        
# https://github.com/tatp22/multidim-positional-encoding
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=0)
        return torch.flatten(emb, 0, 1)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cache is not None and self.cache.shape == tensor.shape:
            return self.cache

        self.cache = None
        # batch_size, x, y, orig_ch = tensor.shape
        batch_size, orig_ch, y, x = tensor.shape
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(2).repeat(1, 1, x)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).repeat(1, y, 1)

        emb = torch.cat([emb_y, emb_x], axis=0)[None]
        self.cache = emb.repeat(tensor.shape[0], 1, 1, 1)
        return self.cache


class MultiDepthSweepFusion(nn.Module):
    """
    Cylindrical depth sweeping based fusion module using multiple depth hypotheses
    Project features for all views (per-camera + per-depth), apply view-attention, then apply a ResNet block

    Parameters
    ----------
    scale_and_shapes: Dict of {String: Tuple(in_scale, in_shape, out_scale, out_shape)}
        input and output shapes of this module per camera
    view_attention: bool
        turn on/off view attention
    positional_encoding: int
        the number of channels for positional encoding
    depth_hypotheses: List of float
        Hypothesized scalar depth values
    """
    def __init__(self, scale_and_shapes, view_attention=False, positional_encoding=0, depth_hypotheses=[3, 5, 10, 30],  neighbors = False):
        super().__init__()
        self.per_camera_transforms = nn.ModuleDict({
            camera: MultiDepthTransform(camera, *shapes, given_depths=depth_hypotheses)
                for camera, shapes in scale_and_shapes.items()})

        # TODO(soonminh): Assume all camera features have the same shape
        _, in_shape, _, out_shape = list(scale_and_shapes.values())[0]
        num_cameras = len(scale_and_shapes.keys())
        num_depths = len(depth_hypotheses)
        self.num_cameras = num_cameras
        self.num_depths = num_depths
        self.prepare_att = None
        if view_attention:
            self.prepare_att = nn.Conv2d(in_shape[0] * num_cameras * num_depths, num_cameras * num_depths, kernel_size=3, padding=1)
        
        self.cylindrical_pad = CylindricalPad(padding = 2)
        self.conv = BasicBlock(in_shape[0], out_shape[0], downsample=None)

        self.get_pos_enc = None
        if positional_encoding > 0:
            self.get_pos_enc = PositionalEncoding2D(positional_encoding)
        
        self.weight_net = nn.ModuleList([AggWeightNetVolume(in_shape[0]) for _ in range(self.num_depths)])

        self.neighbors = neighbors
        channels = in_shape[0]
        self.attention = ScaledDotProductAttention(channels)

    def forward(self, feats, meta, return_logs=False):
        # 각 깊이별로 카메라의 피처들을 저장할 리스트
        grouped_panofeats = [[] for _ in range(self.num_depths)]

        # 각 카메라의 피처를 깊이별로 변환 후 그룹화
        for cam, feat in feats.items():
            # 각 카메라의 피처들을 변환하여 깊이별로 나눔
            panofeat_list = self.per_camera_transforms[cam](feat, meta)
            for depth_idx in range(self.num_depths):
                grouped_panofeats[depth_idx].append(panofeat_list[depth_idx])

        volume_adapt = 0
        num_views = None  # 각 위치에서의 뷰 수를 저장할 변수

        # 각 깊이 가설에 대해 volume_variance 계산
        volume_variance = compute_volume_variance(grouped_panofeats, self.num_cameras)

        # 각 깊이 가설별로 처리
        for depth_idx, cam_feats in enumerate(grouped_panofeats):
            panofeat_stack = torch.stack(cam_feats, dim=1)
            panofeat_stack_transposed = panofeat_stack.transpose(1, 2)
            panofeat_stack_reshaped = panofeat_stack_transposed.view(
                panofeat_stack_transposed.size(0),
                panofeat_stack_transposed.size(1),
                -1,
                panofeat_stack_transposed.size(3),
                panofeat_stack_transposed.size(4)
            )
            # import ipdb; ipdb.set_trace()
            # weight_net에서 volume_variance를 활용한 신뢰도 계산
            weight = self.weight_net[depth_idx](panofeat_stack_reshaped, volume_variance)  # (B, 1, D, H, W)

            weighted_feat = (weight + 1) * panofeat_stack_reshaped
            depth_feat = weighted_feat.sum(dim=2)
            volume_adapt += depth_feat

            view_mask = (panofeat_stack.sum(dim=2) != 0).float()
            if num_views is None:
                num_views = view_mask
            else:
                num_views += view_mask

        num_views = num_views.sum(dim=1, keepdim=True).clamp(min=1.0)
        volume_adapt = volume_adapt / num_views

        padded_feat = self.cylindrical_pad(volume_adapt)
        out = self.conv(padded_feat)
        out = out[:, :, :, 1:-3]

        if self.get_pos_enc is not None:
            pos_enc = self.get_pos_enc(out)
            out = torch.cat([out, pos_enc], axis=1)

        return out


        # 1. Cylinderical sweeping for all views (per-camera + per-depth)
        # panofeats = []
        # for cam, feat in feats.items():
        #     panofeat = self.per_camera_transforms[cam](feat, meta)
        #     # panofeats.append(panofeat)
        #     panofeats.extend(panofeat)
            

        #     # for nbr in getattr(self.neighbors, cam, []):
        #     #     nbr_feat = feats[nbr]

        #     #     # query = feat_reshaped # [batch_size, height*width, channels]
        #     #     query = nbr_feat
        #     #     key = value = feat
        #     #     attended_feat = self.attention(query, key, value)[0]  # Apply attention
        #     #     processed_feat = self.per_camera_transforms[cam](attended_feat, meta)
        #     #     panofeats.extend(processed_feat)
        
        # grouped_panofeats = [[] for _ in range(self.num_depths)]  # 깊이별로 그룹화할 리스트 생성

        # for i in range(len(panofeats)):
        #     depth_idx = i % self.num_depths  # 깊이 인덱스 계산
        #     grouped_panofeats[depth_idx].append(panofeats[i])  

        # volume_adapt = None
        # for cam_feats in grouped_panofeats:
        #     for i, panofeat in enumerate(cam_feats):
        #         weight = torch.sigmoid(self.weight_net[i](panofeat))
        #         if volume_adapt is None:
        #             volume_adapt = (weight + 1) * panofeat
        #         else:
        #             volume_adapt = volume_adapt + (weight + 1) * panofeat
        # aggregated_out = volume_adapt.sum(dim=2) 
        # import ipdb;ipdb.set_tracde
        # """
        # out 사이즈 조절하고 패딩하는 부분 수정
        # """
        
        # # # 2. Aggregate multicam cylindrical features
        # # num_views = torch.concat([panofeat.detach().sum(axis=1, keepdim=True) != 0.0 for panofeat in panofeats], axis=1)
        # # num_views = num_views.sum(axis=1, keepdim=True).clamp(min=1.0)

        # if self.prepare_att is None:
        #     # aggregate_feat = torch.stack(panofeats, axis=1).sum(axis=1)
        #     # import ipdb; ipdb.set_trace()
        #     # padded_num_views = self.cylindrical_pad(num_views)
        #     padded_feat = self.cylindrical_pad(aggregated_out)
        #     # padded_feat = self.cylindrical_pad(aggregate_feat)
        #     # out = padded_feat / padded_num_views
        #     out = self.conv(padded_feat)
        #     out = out[:, :, :, 1:-3]
    
        # # else:
        # #     multicam_panofeat = torch.stack(panofeats, axis=1)

        # #     # 3. View-attention
        # #     # Get {num_cams x num_depths}-dimensional attention mask and expand it (by broadcasting)
        # #     # TODO(soonminh): Check if .detach() is required for attn
        # #     attn = torch.sigmoid(self.prepare_att(torch.cat(panofeats, axis=1))).unsqueeze(2)
        # #     multicam_panofeat = (multicam_panofeat * attn).sum(axis=1) / num_views
        # #     out = self.conv(multicam_panofeat)

        # # # 3. TODO(soonminh): cylindrical padding to improve boundaries
        
        # # 4. Positional encoding
        # if self.get_pos_enc is not None:
        #     pos_enc = self.get_pos_enc(out)
        #     out = torch.cat([out, pos_enc], axis=1)

        # return out
