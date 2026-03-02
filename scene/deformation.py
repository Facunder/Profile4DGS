import functools
import math
import os
import time
from tkinter import W
from time import time as get_time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        # time1 = get_time()
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        # time2 = get_time()
        # print("hex time",time2-time1)
        # torch.cuda.synchronize()
        # time3 = get_time()
        
        # hidden = self.feature_out(hidden)  
        
        # torch.cuda.synchronize() 
        # time4 = get_time()
        # print("linear time",time4-time3)
        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, frame_id=0, static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None, enable_rigid_cluster=False):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, frame_id, static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias, enable_rigid_cluster)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    # def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, frame_id=0, static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None):
    #     if frame_id % 3 == 0:
    #         key_frame_flag = True
    #     else:
    #         key_frame_flag = False
    #     torch.cuda.synchronize()
    #     time1 = get_time()
    #     dx = torch.zeros_like(rays_pts_emb[:,:3])
    #     ds = torch.zeros_like(scales_emb[:,:3])
    #     dr = torch.zeros_like(rotations_emb[:,:4])
    #     static_mask_view = static_mask.view(-1).bool()
    #     if key_frame_flag != True:
    #         dx = ref_pos_bias
    #         ds = ref_scale_bias
    #         dr = ref_rot_bias
    #         rays_pts_emb_dynamic = rays_pts_emb[static_mask_view==0]
    #         scales_emb_dynamic = scales_emb[static_mask_view==0]
    #         rotations_emb_dynamic = rotations_emb[static_mask_view==0]
    #         time_emb_dynamic = time_emb[static_mask_view==0]
    #     else:
    #         rays_pts_emb_dynamic = rays_pts_emb
    #         scales_emb_dynamic = scales_emb
    #         rotations_emb_dynamic = rotations_emb
    #         time_emb_dynamic = time_emb
    #     torch.cuda.synchronize()
    #     time1_1 = get_time()
    #     print("mask time: ",time1_1-time1)
        
    #     torch.cuda.synchronize()
    #     time2 = get_time()
    #     hidden = self.query_time(rays_pts_emb_dynamic, scales_emb_dynamic, rotations_emb_dynamic, time_feature, time_emb_dynamic)
    #     # print("hidden feature shape: ", hidden.shape)
    #     torch.cuda.synchronize()
    #     time2_1 = get_time()
    #     print("hexplane time: ",time2_1-time2)
        
    #     torch.cuda.synchronize()
    #     time3 = get_time()
    #     hidden = self.feature_out(hidden)  
    #     if self.args.static_mlp:
    #         mask = self.static_mlp(hidden)
    #     elif self.args.empty_voxel:
    #         mask = self.empty_voxel(rays_pts_emb[:,:3])
    #     else:
    #         mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
    #     # breakpoint()
    #     if self.args.no_dx:
    #         pts = rays_pts_emb[:,:3]
    #     else:
    #         if key_frame_flag != True:
    #             dx[static_mask_view==0] = self.pos_deform(hidden)
    #         else:
    #             dx = self.pos_deform(hidden)
    #         # pts = torch.zeros_like(rays_pts_emb[:,:3])
    #         pts = rays_pts_emb[:,:3]*mask + dx
        
    #     if self.args.no_ds :
    #         scales = scales_emb[:,:3]
    #     else:
    #         if key_frame_flag != True:
    #             ds[static_mask_view==0] = self.scales_deform(hidden)
    #         else:
    #             ds = self.scales_deform(hidden)
    #         # scales = torch.zeros_like(scales_emb[:,:3])
    #         scales = scales_emb[:,:3]*mask + ds
            
    #     if self.args.no_dr :
    #         rotations = rotations_emb[:,:4]
    #     else:
    #         if key_frame_flag != True:
    #             dr[static_mask_view==0] = self.rotations_deform(hidden)
    #         else:
    #             dr = self.rotations_deform(hidden)
    #         # rotations = torch.zeros_like(rotations_emb[:,:4])
    #         if self.args.apply_rotation:
    #             rotations = batch_quaternion_multiply(rotations_emb, dr)
    #         else:
    #             rotations = rotations_emb[:,:4] + dr

    #     if self.args.no_do :
    #         opacity = opacity_emb[:,:1] 
    #     else:
    #         do = self.opacity_deform(hidden) 
          
    #         opacity = torch.zeros_like(opacity_emb[:,:1])
    #         opacity = opacity_emb[:,:1]*mask + do
    #     if self.args.no_dshs:
    #         shs = shs_emb
    #     else:
    #         dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

    #         shs = torch.zeros_like(shs_emb)
    #         # breakpoint()
    #         shs = shs_emb*mask.unsqueeze(-1) + dshs
    #     torch.cuda.synchronize()
    #     time4 = get_time()
    #     print("mlp time",time4-time3)
        
    #     return pts, scales, rotations, opacity, shs, dx.cpu(), ds.cpu(), dr.cpu()


    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, frame_id=0, static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None, enable_rigid_cluster=False):
        if frame_id % 3 == 0:
            key_frame_flag = True
        else:
            key_frame_flag = False

        # key_frame_flag = True # for test
        if static_mask.sum() == 0:
            key_frame_flag = True

        torch.cuda.synchronize()
        time1 = get_time()
        dx = torch.zeros_like(rays_pts_emb[:,:3])
        ds = torch.zeros_like(scales_emb[:,:3])
        dr = torch.zeros_like(rotations_emb[:,:4])
        static_mask_view = static_mask.view(-1).bool()
        if key_frame_flag != True:
            dx = ref_pos_bias
            ds = ref_scale_bias
            dr = ref_rot_bias
            rays_pts_emb_dynamic = rays_pts_emb[static_mask_view==0]
            scales_emb_dynamic = scales_emb[static_mask_view==0]
            rotations_emb_dynamic = rotations_emb[static_mask_view==0]
            time_emb_dynamic = time_emb[static_mask_view==0]
        else:
            rays_pts_emb_dynamic = rays_pts_emb
            scales_emb_dynamic = scales_emb
            rotations_emb_dynamic = rotations_emb
            time_emb_dynamic = time_emb
        torch.cuda.synchronize()
        time1_1 = get_time()
        print("mask time: ",time1_1-time1)
        
        torch.cuda.synchronize()
        time2 = get_time()
        hidden = self.query_time(rays_pts_emb_dynamic, scales_emb_dynamic, rotations_emb_dynamic, time_feature, time_emb_dynamic)
        # print("hidden feature shape: ", hidden.shape)
        torch.cuda.synchronize()
        time2_1 = get_time()
        print("hexplane time: ",time2_1-time2)
        
        torch.cuda.synchronize()
        time3 = get_time()
        hidden = self.feature_out(hidden)  
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            pos_deform = self.pos_deform(hidden)
            if key_frame_flag != True:
                dx[static_mask_view==0] = pos_deform
            else:
                dx = pos_deform
            
            # ========== 新增的量化和聚类逻辑 ==========
            # 给定的3维scaler张量 (您需要提供这个张量，这里假设为示例值)
            # scaler = torch.tensor([1024.0, 1024.0, 1024.0], device=dx.device, dtype=dx.dtype)
            # *************************************************************************************************
            # if key_frame_flag:
            #     quant_bit = 14 # Orig
            # else:
            #     quant_bit = 14 # 14 Orig
            # scaler = self.get_aabb[1] - self.get_aabb[0]
            # scaler = (2**quant_bit) / scaler
            # # 对dx的每一列进行12bit量化
            # dx_scaled = pos_deform * scaler.unsqueeze(0)  # [N, 3]
            # dx_quantized = torch.clamp(torch.round(dx_scaled), 0, (2**quant_bit)-1).long()  # 12bit: [0, 4095]
            
            # # 将三列合并为一个36bit的唯一标识符
            # # 使用位操作将三个12bit值合并为一个30bit值
            # dx_combined = (dx_quantized[:, 0] << (2*quant_bit)) + (dx_quantized[:, 1] << quant_bit) + dx_quantized[:, 2]
            
            # # 聚类：找到所有唯一的36bit值和对应的首次出现索引
            # unique_values, inverse_indices, counts = torch.unique(dx_combined, return_inverse=True, return_counts=True)
            
            # # # 为每个唯一值找到第一个出现的索引
            # # first_indices = torch.zeros(len(unique_values), dtype=torch.long, device=dx.device)
            # # for i, val in enumerate(unique_values):
            # #     first_indices[i] = (dx_combined == val).nonzero(as_tuple=True)[0][0]
            # # # 只选择唯一行对应的hidden进行后续计算

            # # 找到每个唯一值的首次出现位置
            # first_indices = torch.zeros(len(unique_values), dtype=torch.long, device=dx_combined.device)
            # first_indices[inverse_indices] = torch.arange(len(dx_combined), device=dx_combined.device)
            # first_indices = first_indices[torch.arange(len(unique_values), device=dx_combined.device)]

            # hidden_unique = hidden[first_indices]
            # print("hidden_unique shape: ",hidden_unique.shape, "total values shape: ",dx_combined.shape, "cluster by: ", dx_combined.device)
            # *************************************************************************************************
            if(enable_rigid_cluster == True):
                if key_frame_flag:
                    quant_bit = 14
                else:
                    quant_bit = 14

                scaler = self.get_aabb[1] - self.get_aabb[0]
                scaler = (2**quant_bit) / scaler

                # 缩放并量化
                dx_scaled = pos_deform * scaler.unsqueeze(0)
                dx_quantized = torch.clamp(torch.round(dx_scaled), 0, (2**quant_bit)-1).long()

                # 提速小技巧：对于位拼接，使用按位或 '|' 比加法 '+' 更快且更安全
                dx_combined = (dx_quantized[:, 0] << (2*quant_bit)) | (dx_quantized[:, 1] << quant_bit) | dx_quantized[:, 2]

                # ================= 核心优化部分 =================
                # 1. 进行稳定的排序 (保持首次出现的相对顺序)
                sorted_vals, sorted_indices = torch.sort(dx_combined, stable=True)

                # 2. 找到发生变化的位置（即每个唯一值第一次出现的位置）
                unique_mask = torch.empty_like(sorted_vals, dtype=torch.bool)
                unique_mask[0] = True  # 第一个元素肯定是新的唯一值
                unique_mask[1:] = sorted_vals[1:] != sorted_vals[:-1]

                # 3. 提取首次出现的原始索引
                # first_indices = sorted_indices[unique_mask]
                unique_values, inverse_indices = torch.unique(dx_combined, return_inverse=True)

                # 创建一个索引序列
                indices = torch.arange(len(dx_combined), device=dx_combined.device)

                # 初始化 first_indices，默认值为一个极大的数（或长度），以确保 amin 能够覆盖它
                first_indices = torch.full((len(unique_values),), len(dx_combined), dtype=torch.long, device=dx_combined.device)

                # 使用 scatter_reduce 找到相同聚类中的最小索引（即首次出现的索引）
                first_indices.scatter_reduce_(
                    dim=0, 
                    index=inverse_indices, 
                    src=indices, 
                    reduce="amin", 
                    include_self=False
                )

                # (可选) 如果你需要 unique_values
                # unique_values = sorted_vals[unique_mask] 

                # 4. 获取对应的 hidden
                hidden_unique = hidden[first_indices]
                
                print("hidden_unique shape: ", hidden_unique.shape, "cluster by: ", dx_combined.device)

            # pts计算保持不变
            pts = rays_pts_emb[:,:3]*mask + dx
        
        # ========== 修改scales计算 ==========
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            if key_frame_flag != True:
                # 只对唯一行计算ds
                if not self.args.no_dx:  # 只有在进行了聚类的情况下才优化
                    if(enable_rigid_cluster == True):
                        ds_unique = self.scales_deform(hidden_unique)
                        # 将结果复用到所有对应的行
                        ds_full = ds_unique[inverse_indices]
                        ds[static_mask_view==0] = ds_full
                    else: # vanilla
                        ds[static_mask_view==0] = self.scales_deform(hidden)
                else:
                    ds[static_mask_view==0] = self.scales_deform(hidden)
            else:
                if not self.args.no_dx:  # 只有在进行了聚类的情况下才优化    
                    if(enable_rigid_cluster == True):
                        ds_unique = self.scales_deform(hidden_unique)
                        # 将结果复用到所有对应的行
                        ds = ds_unique[inverse_indices]
                    else:
                        ds[static_mask_view==0] = self.scales_deform(hidden)
                else:
                    ds = self.scales_deform(hidden)
            # scales = torch.zeros_like(scales_emb[:,:3])

            scales = scales_emb[:,:3]*mask + ds
        
        # ========== 修改rotations计算 ==========        
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            if key_frame_flag != True:
                # 只对唯一行计算dr
                if not self.args.no_dx:  # 只有在进行了聚类的情况下才优化
                    if(enable_rigid_cluster == True):
                        dr_unique = self.rotations_deform(hidden_unique)
                        # 将结果复用到所有对应的行
                        dr_full = dr_unique[inverse_indices]
                        dr[static_mask_view==0] = dr_full
                    else:
                        dr[static_mask_view==0] = self.rotations_deform(hidden)
                else:
                    dr[static_mask_view==0] = self.rotations_deform(hidden)
            else:
                if not self.args.no_dx:  # 只有在进行了聚类的情况下才优化
                    if(enable_rigid_cluster == True):
                        dr_unique = self.rotations_deform(hidden_unique)
                        # 将结果复用到所有对应的行
                        dr = dr_unique[inverse_indices]
                    else:
                        dr[static_mask_view==0] = self.rotations_deform(hidden)
                else:
                    dr = self.rotations_deform(hidden)
            # rotations = torch.zeros_like(rotations_emb[:,:4])

            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        # opacity和shs的计算保持不变
        if self.args.no_do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden) 
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            # shs_emb[static_mask_view==0] = torch.zeros_like(shs_emb[static_mask_view==0])
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs
        torch.cuda.synchronize()
        time4 = get_time()
        print("mlp time",time4-time3)
        
        return pts, scales, rotations, opacity, shs, dx.cpu(), ds.cpu(), dr.cpu()

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, frame_id=0, static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None, enable_rigid_cluster=False):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, frame_id, static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias, enable_rigid_cluster)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, frame_id=0, static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None, enable_rigid_cluster=False):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs, cur_dx, cur_ds, cur_dr = self.deformation_net( point_emb,
                                                scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel, 
                                                frame_id, static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias, enable_rigid_cluster)
        return means3D, scales, rotations, opacity, shs, cur_dx, cur_ds, cur_dr
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb