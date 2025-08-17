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
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, frame_id=0, group_static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, frame_id, group_static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb, frame_id=0, group_static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None):
        if frame_id % 3 == 0:
            key_frame_flag = True
        else:
            key_frame_flag = False
        point_nums = rays_pts_emb.shape[0]
        group_nums = point_nums // 4
        dynamic_group_nums = int(group_nums - group_static_mask.sum().item())
        print(f"Dynamic Cluster: {dynamic_group_nums}/{group_nums}")
        torch.cuda.synchronize()
        time1 = get_time()
        dx = torch.zeros_like(rays_pts_emb[:,:3])
        ds = torch.zeros_like(scales_emb[:,:3])
        dr = torch.zeros_like(rotations_emb[:,:4])
        group_static_mask_view = group_static_mask.view(-1).bool()
        if key_frame_flag != True:
            dx = ref_pos_bias
            ds = ref_scale_bias
            dr = ref_rot_bias
            rays_pts_emb_dynamic = rays_pts_emb[:group_nums*4].reshape(group_nums, 4, -1)[group_static_mask_view==0].reshape(-1, rays_pts_emb.shape[-1])
            rays_pts_emb_dynamic = torch.cat([rays_pts_emb_dynamic, rays_pts_emb[group_nums*4:]], dim=0)
            scales_emb_dynamic = scales_emb[:group_nums*4].reshape(group_nums, 4, -1)[group_static_mask_view==0].reshape(-1, scales_emb.shape[-1])
            scales_emb_dynamic = torch.cat([scales_emb_dynamic, scales_emb[group_nums*4:]], dim=0)
            rotations_emb_dynamic = rotations_emb[:group_nums*4].reshape(group_nums, 4, -1)[group_static_mask_view==0].reshape(-1, rotations_emb.shape[-1])
            rotations_emb_dynamic = torch.cat([rotations_emb_dynamic, rotations_emb[group_nums*4:]], dim=0)
            time_emb_dynamic = time_emb[:group_nums*4].reshape(group_nums, 4, -1)[group_static_mask_view==0].reshape(-1, time_emb.shape[-1])
            time_emb_dynamic = torch.cat([time_emb_dynamic, time_emb[group_nums*4:]], dim=0)
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
        
        dynamic_hidden_grouped = hidden[:dynamic_group_nums*4].view(dynamic_group_nums, 4, hidden.shape[1])
        first_row = dynamic_hidden_grouped[:, 0:1, :]  # (dynamic_group_nums, 1, 64)
        others = dynamic_hidden_grouped[:, 1:, :]      # (dynamic_group_nums, 3, 64)
        mse = ((others - first_row) ** 2).mean(dim=2)  # (dynamic_group_nums, 3)
        rigidity_mask = (mse < 0.01).all(dim=1)  # (dynamic_group_nums,)
        rigidity_cluster_num = rigidity_mask.sum()
        non_rigidity_cluster_num = dynamic_group_nums - rigidity_cluster_num
        print(f"Rigidity Dynamic Cluster: {rigidity_cluster_num}/{dynamic_group_nums}")
        ancher_dynamic_hidden = dynamic_hidden_grouped[rigidity_mask, 0, :]
        common_dynamic_hidden = dynamic_hidden_grouped[~rigidity_mask].reshape(-1, hidden.shape[1])
        common_dynamic_hidden = torch.cat([common_dynamic_hidden, hidden[dynamic_group_nums*4:]], dim=0)

        # in all clusters
        group_indices = torch.arange(group_nums).unsqueeze(1) * 4 + torch.arange(4)   # [group_nums, 4]
        group_indices = group_indices.to(group_static_mask_view.device)
        dynamic_point_indices = group_indices[~group_static_mask_view].reshape(-1)  # [num_common_points]
        # in dynamic clusters
        dynamic_dx = torch.zeros_like(rays_pts_emb_dynamic[:,:3])
        dynamic_ds = torch.zeros_like(scales_emb_dynamic[:,:3])
        dynamic_dr = torch.zeros_like(rotations_emb_dynamic[:,:4])
        dynamic_group_indices = torch.arange(dynamic_group_nums).unsqueeze(1) * 4 + torch.arange(4)   # [num_dynamic_groups, 4]
        dynamic_group_indices = dynamic_group_indices.to(rigidity_mask.device)
        common_dynamic_point_indices = dynamic_group_indices[~rigidity_mask].reshape(-1)  # [num_common_points]
        rigidity_dynamic_point_indices = dynamic_group_indices[rigidity_mask].reshape(-1)  # [num_rigidity_points]
        
        torch.cuda.synchronize()
        time3 = get_time()
        # hidden = self.feature_out(hidden)  
        ancher_dynamic_hidden = self.feature_out(ancher_dynamic_hidden)
        common_dynamic_hidden = self.feature_out(common_dynamic_hidden)
        
        # No Use
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
            ancher_dynamic_dx = self.pos_deform(ancher_dynamic_hidden)
            common_dynamic_dx = self.pos_deform(common_dynamic_hidden)
            dynamic_dx[rigidity_dynamic_point_indices] = ancher_dynamic_dx.repeat_interleave(4, dim=0)
            dynamic_dx[common_dynamic_point_indices] = common_dynamic_dx[:non_rigidity_cluster_num*4]
            dynamic_dx[dynamic_group_nums * 4:] = common_dynamic_dx[non_rigidity_cluster_num * 4:]
            if key_frame_flag != True:
                dx[dynamic_point_indices] = dynamic_dx[:dynamic_group_nums * 4]
                dx[group_nums*4:] = dynamic_dx[dynamic_group_nums * 4:]
            else:
                # dx = self.pos_deform(hidden)
                dx = dynamic_dx
            # pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
            
        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ancher_dynamic_ds = self.scales_deform(ancher_dynamic_hidden)
            common_dynamic_ds = self.scales_deform(common_dynamic_hidden)
            dynamic_ds[rigidity_dynamic_point_indices] = ancher_dynamic_ds.repeat_interleave(4, dim=0)
            dynamic_ds[common_dynamic_point_indices] = common_dynamic_ds[:non_rigidity_cluster_num*4]
            dynamic_ds[dynamic_group_nums * 4:] = common_dynamic_ds[non_rigidity_cluster_num * 4:]
            if key_frame_flag != True:
                # ds[static_mask_view==0] = self.scales_deform(hidden)
                # dynamic_ds = self.scales_deform(hidden)
                ds[dynamic_point_indices] = dynamic_ds[:dynamic_group_nums * 4]
                ds[group_nums*4:] = dynamic_ds[dynamic_group_nums * 4:]
            else:
                # ds = self.scales_deform(hidden)
                ds = dynamic_ds
            # scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            ancher_dynamic_dr = self.rotations_deform(ancher_dynamic_hidden)
            common_dynamic_dr = self.rotations_deform(common_dynamic_hidden)
            dynamic_dr[rigidity_dynamic_point_indices] = ancher_dynamic_dr.repeat_interleave(4, dim=0)
            dynamic_dr[common_dynamic_point_indices] = common_dynamic_dr[:non_rigidity_cluster_num*4]
            dynamic_dr[dynamic_group_nums * 4:] = common_dynamic_dr[non_rigidity_cluster_num * 4:]
            if key_frame_flag != True:
                # dr[static_mask_view==0] = self.rotations_deform(hidden)
                # dynamic_dr = self.rotations_deform(hidden)
                dr[dynamic_point_indices] = dynamic_dr[:dynamic_group_nums * 4]
                dr[group_nums*4:] = dynamic_dr[dynamic_group_nums * 4:]
            else:
                # dr = self.rotations_deform(hidden)
                dr = dynamic_dr
            # rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
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

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, frame_id=0, group_static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, frame_id, group_static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, frame_id=0, group_static_mask=None, ref_pos_bias=None, ref_scale_bias=None, ref_rot_bias=None):
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
                                                frame_id, group_static_mask, ref_pos_bias, ref_scale_bias, ref_rot_bias)
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