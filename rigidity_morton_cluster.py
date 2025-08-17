import numpy as np
import struct
import argparse
import os

# ====== 新：依赖 PyTorch（GPU 优先）======
import torch

# ---------------- PLY 读写保持不变 ----------------
def read_ply_file(file_path):
    """读取PLY文件并返回点云数据"""
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break

        num_points = 0
        for line in header:
            if line.startswith('element vertex'):
                num_points = int(line.split()[2])
                break

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

        data_format = '<' + 'f' * len(properties)
        point_size = struct.calcsize(data_format)

        points_data = []
        for _ in range(num_points):
            point_bytes = f.read(point_size)
            point_values = struct.unpack(data_format, point_bytes)
            points_data.append(point_values)

        return np.array(points_data), properties


def write_ply_file(file_path, points_data, properties):
    """写入PLY文件"""
    num_points = len(points_data)
    with open(file_path, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {num_points}\n'.encode())
        for prop in properties:
            f.write(f'property float {prop}\n'.encode())
        f.write(b'end_header\n')
        for point in points_data:
            for value in point:
                f.write(struct.pack('<f', float(value)))


def write_cluster_indices(file_path, cluster_groups):
    """写入索引文件：按组顺序逐行输出索引。这里每组即 Morton 排序后的连续 group_size 个点。"""
    with open(file_path, 'w') as f:
        for group in cluster_groups:
            for index in group:
                f.write(f"{int(index)}\n")

# ---------------- Morton 排序（GPU 加速）----------------
def _expand_bits_21(v: torch.Tensor) -> torch.Tensor:
    """
    把 21-bit 整数的 bit 展开成 0b00x00x... 形式，用于 3D Morton 交织。
    使用 torch.long(int64) 做位运算即可；常量掩码也要用 long。
    """
    v = v.to(torch.long)
    v = (v | (v << 32)) & torch.tensor(0x1f00000000ffff, dtype=torch.long, device=v.device)
    v = (v | (v << 16)) & torch.tensor(0x1f0000ff0000ff, dtype=torch.long, device=v.device)
    v = (v | (v << 8))  & torch.tensor(0x100f00f00f00f00f, dtype=torch.long, device=v.device)
    v = (v | (v << 4))  & torch.tensor(0x10c30c30c30c30c3, dtype=torch.long, device=v.device)
    v = (v | (v << 2))  & torch.tensor(0x1249249249249249, dtype=torch.long, device=v.device)
    return v

@torch.no_grad()
def morton_order_indices_torch(points_xyz: torch.Tensor, bits: int = 21) -> torch.Tensor:
    """
    输入: (N,3) 的 float32/float64
    输出: Morton(Z-order) 排序后的索引 (N,)
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    device = points_xyz.device
    eps = torch.finfo(points_xyz.dtype).eps

    mins = torch.min(points_xyz, dim=0).values
    maxs = torch.max(points_xyz, dim=0).values
    ranges = torch.clamp(maxs - mins, min=eps)
    normed = (points_xyz - mins) / ranges  # [0,1]

    max_q = (1 << bits) - 1
    # 注意：clamp 的上界给 float，然后再转 long，避免 dtype 不匹配
    q = torch.clamp((normed * max_q).floor(), 0.0, float(max_q)).to(torch.long)

    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    xx, yy, zz = _expand_bits_21(x), _expand_bits_21(y), _expand_bits_21(z)
    morton = (xx << 0) | (yy << 1) | (zz << 2)  # int64 位运算

    order = torch.argsort(morton)  # GPU/CPU 都支持
    return order

def reorder_points_by_morton(points_data: np.ndarray, group_size: int = 4, bits: int = 21):
    """
    用 Morton(Z-order) 对点进行空间排序；返回重排索引和按 group_size 划分的组。
    - 计算全量 Morton code（GPU 优先）
    - 按 code 升序排序
    - 每 group_size 个构成一组，组首为离组中心最近的点
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.as_tensor(points_data[:, :3], dtype=torch.float32, device=device)

    order = morton_order_indices_torch(xyz, bits=bits)  # (N,)
    sorted_xyz = xyz[order]

    total_points = len(order)
    cluster_groups = []

    for start in range(0, total_points, group_size):
        end = min(start + group_size, total_points)
        group_indices = order[start:end]

        # 计算该组中心（在 GPU 上）
        group_xyz = xyz[group_indices]
        center = group_xyz.mean(dim=0, keepdim=True)
        dists = torch.sum((group_xyz - center) ** 2, dim=1)
        min_idx = torch.argmin(dists).item()

        # 交换，把离中心最近的点放到第一个
        if min_idx != 0:
            group_indices = group_indices.clone()
            tmp = group_indices[0]
            group_indices[0] = group_indices[min_idx]
            group_indices[min_idx] = tmp

        cluster_groups.append(group_indices.cpu().numpy())

    # 拼成整体的重排索引
    reordered_indices = np.concatenate(cluster_groups)

    return reordered_indices, cluster_groups


# ---------------- 主流程：替换原 KMeans ----------------
def main():
    parser = argparse.ArgumentParser(description='点云 Morton(Z-order) 空间排序重排工具（GPU 加速）')
    parser.add_argument('input_file', help='输入的PLY文件路径')
    parser.add_argument('output_file', help='输出的PLY文件路径')
    parser.add_argument('--group_size', type=int, default=4, help='每组的点数量 (默认: 4)')
    parser.add_argument('--indices_file', help='输出的点索引文件路径(可选)', default=None)
    parser.add_argument('--bits', type=int, default=21, help='每轴量化比特数（<=21，默认21）')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return

    print(f"正在读取文件: {args.input_file}")
    try:
        points_data, properties = read_ply_file(args.input_file)
        print(f"成功读取 {len(points_data)} 个点")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # Morton 排序重排
    print("开始 Morton(Z-order) 空间排序重排...")
    try:
        reordered_indices, cluster_groups = reorder_points_by_morton(
            points_data, group_size=args.group_size, bits=args.bits
        )
        reordered_points = points_data[reordered_indices]
        print("空间排序重排完成")
    except Exception as e:
        print(f"空间排序重排时出错: {e}")
        return

    assert len(reordered_points) == len(points_data), f"点数量发生了变化: {len(reordered_points)} != {len(points_data)}"

    print(f"正在写入文件: {args.output_file}")
    try:
        write_ply_file(args.output_file, reordered_points, properties)
        print("文件写入完成")
    except Exception as e:
        print(f"写入文件时出错: {e}")
        return

    if args.indices_file:
        print(f"正在写入索引文件: {args.indices_file}")
        try:
            write_cluster_indices(args.indices_file, cluster_groups)
            print("索引文件写入完成")
        except Exception as e:
            print(f"写入索引文件时出错: {e}")

    print("重排完成！")
    print(f"原始点数: {len(points_data)}")
    print(f"重排后点数: {len(reordered_points)}（保持不变）")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()