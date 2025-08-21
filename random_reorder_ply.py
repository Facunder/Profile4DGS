#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuffle vertices in a PLY file and write out a new PLY.
- Supports ASCII or binary PLY via 'plyfile' library.
- If 'face' exists, remaps face vertex indices after shuffling.
Usage:
    python shuffle_ply.py input.ply output.ply [--seed 42] [--ascii]
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def shuffle_ply(in_path, out_path, seed=None, to_ascii=False):
    ply: PlyData = PlyData.read(in_path)

    if 'vertex' not in ply.elements:
        raise ValueError("输入文件不包含 'vertex' 元素，无法打乱点。")

    # 顶点（结构化数组）
    verts = ply['vertex'].data
    n = len(verts)
    if n == 0:
        raise ValueError("没有顶点可打乱。")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)             # 新顺序
    inv_perm = np.empty_like(perm)        # 旧索引 -> 新索引
    inv_perm[perm] = np.arange(n)

    # 重新排列顶点
    shuffled_verts = verts[perm]

    # 重新组装 vertex 元素（保持字段与类型）
    v_el = PlyElement.describe(shuffled_verts, 'vertex')

    new_elems = [v_el]

    # 如果有面，则重映射索引
    if 'face' in ply.elements:
        face_el = ply['face']
        face_data = face_el.data.copy()

        # 常见字段名：'vertex_indices'（list of int）
        # 也可能叫 'vertex_index' 等，这里尝试识别列表字段并逐项映射
        remapped_face_data = []
        list_field_names = []
        for name, prop in face_el.properties.items():
            # 找 list property
            if getattr(prop, 'is_list', False):
                list_field_names.append(name)

        for row in face_data:
            row_dict = {name: row[name] for name in face_data.dtype.names}
            for lf in list_field_names:
                # 将面顶点索引映射到新的顺序
                # row[lf] 可能是 list 或 numpy array
                idxs = np.asarray(row[lf], dtype=np.int64)
                row_dict[lf] = inv_perm[idxs].tolist()
            remapped_face_data.append(tuple(row_dict[name] for name in face_data.dtype.names))

        remapped_face_data = np.array(remapped_face_data, dtype=face_data.dtype)
        new_elems.append(PlyElement.describe(remapped_face_data, 'face'))

    # 保留其他非 vertex/face 元素（如 edge、自定义元素），不依赖索引的直接保留
    for el in ply.elements:
        if el.name in ('vertex', 'face'):
            continue
        new_elems.append(PlyElement.describe(el.data, el.name))

    # 写回（保持 ASCII 或二进制可选）
    # 若 to_ascii=True 则写 ASCII；否则沿用输入文件的 text 标志
    text_flag = True if to_ascii else ply.text
    PlyData(new_elems, text=text_flag).write(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入 .ply 文件路径")
    ap.add_argument("output", help="输出 .ply 文件路径")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    ap.add_argument("--ascii", action="store_true", help="将输出强制为 ASCII PLY")
    args = ap.parse_args()
    shuffle_ply(args.input, args.output, seed=args.seed, to_ascii=args.ascii)

if __name__ == "__main__":
    main()
