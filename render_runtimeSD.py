#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
from errno import EEXIST
from os import makedirs, path
import numpy as np
import torch
from scene import Scene
import os
# import shutil
# import glob
# import re
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
import copy

def insert_interpolated_views(views, N: int):
    """
    在相邻的两个 CameraInfo 之间插入 N 个新元素。
    - 新元素的 time 为线性插值。
    - 其它属性与右侧原始 CameraInfo 完全相同（复用引用）。
    - 保留原列表中的所有元素与顺序。
    """
    if N < 0:
        raise ValueError("N 必须为非负整数")
    # if len(views) <= 1 or N == 0:
    #     return list(views)

    out = []
    for i in range(len(views)-1): # for jetson
        left = views[i]
        right = views[i + 1]
        out.append(left)  # 先放入左端原始元素

        dt = right.time - left.time
        # 在 (left, right) 之间插入 N 个
        for k in range(1, N + 1):
            t = left.time + dt * (k / (N + 1))
            interp_camera = copy.deepcopy(right)
            interp_camera.time = t
            out.append(interp_camera)
    out.append(views[-1])  # 放入最后一个原始元素
    return out

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def multithread_write(image_list, path):
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=None) # Orig
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, dataset_name):
    if name == "test":
        if dataset_name == "dnerf":
            interp_frame_num = 19
        else:
            interp_frame_num = 3
        views = insert_interpolated_views(views, interp_frame_num)
    
    in_cluster_gauss_nums = 4
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    point_nums = gaussians._xyz.shape[0]
    aabb_bounding = gaussians._deformation.get_aabb
    bounding_scaler = torch.norm((aabb_bounding[0] - aabb_bounding[1]),p=2)
    print("bounding_scaler: ", bounding_scaler)
    group_nums = point_nums // in_cluster_gauss_nums # ignore last
    print("point nums:", point_nums)
    total_time = 0
    cur_time = 0.0 # time_stampe
    pre_pos_bias = torch.zeros(point_nums, 3)
    cur_pos_bias = torch.zeros(point_nums, 3, device="cpu")
    pre_scale_bias = torch.zeros(point_nums, 3)
    pre_rot_bias = torch.zeros(point_nums, 4)
    grid_pos_bias = torch.zeros(point_nums, 3, device="cpu")
    cur_scale_bias = torch.zeros(point_nums, 3, device="cpu")
    cur_rot_bias = torch.zeros(point_nums, 4, device="cpu")
    static_mask = torch.zeros(point_nums, 1, device="cpu") # 1 for static, 0 for non-static
    group_static_mask = torch.zeros(group_nums, 1, device="cpu") # 1 for static, 0 for non-static
    ref_scale = gaussians._scaling.to("cpu")
    ref_rot = gaussians._rotation.to("cpu")
    print("ref_scale:", ref_scale.shape)
    print("ref_rot:", ref_rot.shape)
    mean_relative_pos_bias = torch.zeros(point_nums, 1, device="cpu")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx == 0:
        #     time1=time()
        ### log out original/deformated gaussian parameters
         
        # if idx <= 150: # for jetson
            # continue
        if idx == 150: # for jetson
            break
         
        if (idx%3 == 0):
            ### full dynamic
            static_mask = torch.zeros(point_nums, 1, device="cpu")
        group_static_mask = static_mask[:group_nums*in_cluster_gauss_nums].view(group_nums, in_cluster_gauss_nums).min(dim=1).values
        # else:
            # # interp
            # time_next = view.time.to("cpu")
            # cur_pos_bias_interp = cur_pos_bias + grid_pos_bias * (time_next - cur_time)
            # cur_pos_bias = torch.where((0.005 <= mean_relative_pos_bias) & (mean_relative_pos_bias < 0.02), cur_pos_bias_interp, cur_pos_bias)
        print(f"[INFO] >>>>>>Begin frame-{idx} rendering, timeStamp: {view.time}")
        torch.cuda.synchronize()
        time_sub_1 = time()
        render_pkg = render(view, gaussians, pipeline, background,cam_type=cam_type, frame_id=idx, group_static_mask=group_static_mask, ref_pos_bias=cur_pos_bias.cuda(), ref_scale_bias=cur_scale_bias.cuda(), ref_rot_bias=cur_rot_bias.cuda(), in_cluster_gauss_nums=in_cluster_gauss_nums)
        torch.cuda.synchronize()
        time_sub_2 = time()
        print(f"[INFO] >>>>>> End frame-{idx} rendering, render time:{time_sub_2-time_sub_1}")
        rendering = render_pkg['render']
        total_time += (time_sub_2-time_sub_1)
        render_images.append(to8b(rendering).transpose(1,2,0))
        
        # inertia
        torch.cuda.synchronize()
        time_sub_3 = time()
        pos_bias = render_pkg["pos_bias"]
        scale_bias = render_pkg["scale_bias"]
        rot_bias = render_pkg["rot_bias"]
        time_stampe = render_pkg["time_stampe"].to("cpu")
        if ((idx % 3) == 0):
            pre_time = cur_time
            cur_time = time_stampe
            pre_pos_bias = cur_pos_bias
            cur_pos_bias = pos_bias
            pre_scale_bias = cur_scale_bias
            cur_scale_bias = scale_bias
            pre_rot = cur_rot_bias + ref_rot
            cur_rot = rot_bias + ref_rot
        if (idx != 0 and idx % 3 == 0):
            print("<<<Post Static Dynamic Spliting>>>")
            delta_time = cur_time - pre_time
            delta_pos_bias = cur_pos_bias - pre_pos_bias
            delta_scale_bias = cur_scale_bias - pre_scale_bias
            delta_rot_bias = cur_rot_bias - pre_rot_bias 
            abs_pre_pos_bias = torch.abs(pre_pos_bias)
            abs_pre_scale_bias = torch.abs(pre_scale_bias)
            abs_pre_rot_bias = torch.abs(pre_rot_bias)
            norm_pos_change = (torch.norm(delta_pos_bias, p=2, dim=1).unsqueeze(1)) / bounding_scaler.to("cpu")
            relative_scale_change = (torch.norm(delta_scale_bias/ref_scale, p=2, dim=1).unsqueeze(1))
            rot_qua_change = torch.sqrt(8*((1 - torch.sum(cur_rot * pre_rot, dim=1)/(torch.norm(cur_rot, p=2, dim=1)*torch.norm(pre_rot, p=2, dim=1))))).unsqueeze(1)
            
            static_mask = torch.where((norm_pos_change < 0.002) & (relative_scale_change < 0.04) & (rot_qua_change < 0.2), torch.ones_like(static_mask, device="cpu"), torch.zeros_like(static_mask, device="cpu"))
            pos_static = torch.where(norm_pos_change < 0.002, torch.ones_like(static_mask, device="cpu"), torch.zeros_like(static_mask, device="cpu"))
            scale_static = torch.where(relative_scale_change < 0.04, torch.ones_like(static_mask, device="cpu"), torch.zeros_like(static_mask, device="cpu"))
            rot_static = torch.where(rot_qua_change < 0.2, torch.ones_like(static_mask, device="cpu"), torch.zeros_like(static_mask, device="cpu"))
            # print(static_mask.shape)
            static_count = static_mask.sum().item()
            pos_static_count = pos_static.sum().item()
            scale_static_count = scale_static.sum().item()
            rot_static_count = rot_static.sum().item()
            print("pos static count:", pos_static_count)
            print("scale static count:", scale_static_count)
            print("rot static count:", rot_static_count)
            print("static count:", static_count)
        torch.cuda.synchronize()
        time_sub_4 = time()
        print(f"[INFO] >>>>>> End frame-{idx} inertia, inertia time:{time_sub_4-time_sub_3}")
            
        # # for metrics only
        # if dataset_name == "dnerf" and idx % (interp_frame_num + 1) != 0:
        #     continue
        # elif (dataset_name == "hypernerf" or dataset_name == "dynerf") and idx % (interp_frame_num + 1) != 0:
        #     continue
        # else:    
        #     if name in ["train", "test"]:
        #         if cam_type != "PanopticSports":
        #             gt = view.original_image[0:3, :, :]
        #         else:
        #             gt  = view['image'].cuda()
        #         gt_list.append(gt)
        #     render_list.append(rendering)

            
        # if idx == 12 or idx == 13: ## gen GauPRE-sim data
        #     buffer = render_pkg['buffer']
        #     print("\n[ITER {}] Saving Frame".format(idx))
        #     sim_input_path = os.path.join(args.model_path, "sim_pts/frame_{}".format(idx), "raw_data.pt")
        #     mkdir_p(os.path.dirname(sim_input_path))
        #     torch.save(buffer, sim_input_path)
        
        
    # time2=time()
    # print("FPS:",(len(views)-1)/(time2-time1))
    print("FPS:",(len(views)-1)/total_time)

    # # # speed up test 
    # multithread_write(gt_list, gts_path)
    # multithread_write(render_list, render_path)    
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=90) # 30 Orig
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, dataset_name: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[1.0], load_coarse=False, rigidity_ply=True)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type, dataset_name)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type, dataset_name)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--dataset_name", type=str, default="default")
    args = get_combined_args(parser)
    inference_quant = False
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # if active inference hexplane quant
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.dataset_name)