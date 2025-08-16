import numpy as np
import struct
from sklearn.cluster import KMeans
import argparse
import os

def read_ply_file(file_path):
    """读取PLY文件并返回点云数据"""
    with open(file_path, 'rb') as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break
        
        # 解析头部信息获取点数量
        num_points = 0
        for line in header:
            if line.startswith('element vertex'):
                num_points = int(line.split()[2])
                break
        
        # 定义数据结构（根据您提供的属性列表）
        properties = [
            'x', 'y', 'z', 'nx', 'ny', 'nz',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4',
            'f_rest_5', 'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9',
            'f_rest_10', 'f_rest_11', 'f_rest_12', 'f_rest_13', 'f_rest_14',
            'f_rest_15', 'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19',
            'f_rest_20', 'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24',
            'f_rest_25', 'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29',
            'f_rest_30', 'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34',
            'f_rest_35', 'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39',
            'f_rest_40', 'f_rest_41', 'f_rest_42', 'f_rest_43', 'f_rest_44',
            'opacity', 'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]
        
        # 读取二进制数据
        data_format = '<' + 'f' * len(properties)  # 小端序，所有属性都是float
        point_size = struct.calcsize(data_format)
        
        points_data = []
        for i in range(num_points):
            point_bytes = f.read(point_size)
            point_values = struct.unpack(data_format, point_bytes)
            points_data.append(point_values)
        
        return np.array(points_data), properties

def cluster_and_reorder_points(points_data, group_size=4):
    """
    基于XYZ坐标将点聚类并重新排序，不改变点的数量
    返回重排后的索引和聚类标签
    """
    # 提取XYZ坐标
    xyz_coords = points_data[:, :3]  # 前3列是x, y, z
    
    total_points = len(points_data)
    num_groups = total_points // group_size
    
    print(f"总点数: {total_points}")
    print(f"分组数: {num_groups}")
    print(f"每组点数: {group_size}")
    print(f"剩余点数: {total_points % group_size}")
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(xyz_coords)
    
    # 重新排列点：同一聚类的点放在一起
    reordered_indices = []
    cluster_groups = []  # 存储每个聚类的点索引
    cluster_centers = kmeans.cluster_centers_  # 获取聚类中心
    
    # 处理每个聚类
    for cluster_id in range(num_groups):
        # 找到属于当前聚类的所有点的索引
        cluster_point_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_point_indices) > 0:
            # 计算每个点到聚类中心的距离
            distances = np.linalg.norm(xyz_coords[cluster_point_indices] - cluster_centers[cluster_id], axis=1)
            # 按照距离排序，最近的在前
            sorted_indices = cluster_point_indices[np.argsort(distances)]
            
            # 如果聚类中点数超过group_size，选择前group_size个点
            if len(sorted_indices) > group_size:
                sorted_indices = sorted_indices[:group_size]
            
            # 添加到结果中
            reordered_indices.extend(sorted_indices)
            cluster_groups.append(sorted_indices)
    
    # 处理剩余的点（没有被分配到任何聚类或者被聚类排除的点）
    used_indices = set(reordered_indices)
    remaining_indices = [i for i in range(total_points) if i not in used_indices]
    
    # 将剩余的点添加到结果中
    reordered_indices.extend(remaining_indices)
    if remaining_indices:
        cluster_groups.append(np.array(remaining_indices))
    
    print(f"重排后索引数量: {len(reordered_indices)}")
    
    return np.array(reordered_indices), cluster_groups

def write_cluster_indices(file_path, cluster_groups):
    """写入聚类索引文件，每行一个点索引，按聚类分组顺序存储
       每组中最靠近聚类中心的点索引放在第一个位置"""
    with open(file_path, 'w') as f:
        for group in cluster_groups:
            # 确保组内点索引的顺序已经按照距离排序（最近的在第一个位置）
            for index in group:
                f.write(f"{index}\n")

def write_ply_file(file_path, points_data, properties):
    """写入PLY文件"""
    num_points = len(points_data)
    
    with open(file_path, 'wb') as f:
        # 写入头部
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {num_points}\n'.encode())
        
        # 写入属性定义
        for prop in properties:
            f.write(f'property float {prop}\n'.encode())
        
        f.write(b'end_header\n')
        
        # 写入点数据
        for point in points_data:
            for value in point:
                f.write(struct.pack('<f', float(value)))

def main():
    parser = argparse.ArgumentParser(description='点云聚类重排工具')
    parser.add_argument('input_file', help='输入的PLY文件路径')
    parser.add_argument('output_file', help='输出的PLY文件路径')
    parser.add_argument('--group_size', type=int, default=4, help='每组的点数量 (默认: 4)')
    parser.add_argument('--indices_file', help='输出的点索引文件路径(可选)', default=None)
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return
    
    print(f"正在读取文件: {args.input_file}")
    
    # 读取PLY文件
    try:
        points_data, properties = read_ply_file(args.input_file)
        print(f"成功读取 {len(points_data)} 个点")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 进行聚类重排
    print("开始聚类重排...")
    try:
        reordered_indices, cluster_groups = cluster_and_reorder_points(points_data, args.group_size)
        reordered_points = points_data[reordered_indices]
        print("聚类重排完成")
    except Exception as e:
        print(f"聚类重排时出错: {e}")
        return
    
    # 验证点数量没有改变
    assert len(reordered_points) == len(points_data), f"点数量发生了变化: {len(reordered_points)} != {len(points_data)}"
    
    # 写入新的PLY文件
    print(f"正在写入文件: {args.output_file}")
    try:
        write_ply_file(args.output_file, reordered_points, properties)
        print("文件写入完成")
    except Exception as e:
        print(f"写入文件时出错: {e}")
        return
    
    # 如果需要，写入索引文件
    if args.indices_file:
        print(f"正在写入索引文件: {args.indices_file}")
        try:
            write_cluster_indices(args.indices_file, cluster_groups)
            print("索引文件写入完成")
        except Exception as e:
            print(f"写入索引文件时出错: {e}")
    
    print(f"聚类重排完成！")
    print(f"原始点数: {len(points_data)}")
    print(f"重排后点数: {len(reordered_points)}")
    print(f"点数量保持不变: {len(reordered_points) == len(points_data)}")

if __name__ == "__main__":
    main()
# ilaunch q makkapakka python ply_cluster.py /lamport/makkapakka/linyuzheng/workspace_DL/Profile_Deformable_4DGS/output/dnerf/hook_testRigidity_ff/point_cloud/iteration_20000/point_cloud.ply clustered.ply --group_size 4 --indices_file indices.txt
