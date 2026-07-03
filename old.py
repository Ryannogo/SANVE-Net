import os
import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
from transfer.fold_trans import fold_transfer
from contour_prediction import predict_contour
from transfer.pre_ply import pkl_to_ply

import os
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

def load_ply_points(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)

def save_points_to_ply(points, filepath):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)

def pca_split(input_path, output_dir, block_size=2.0, stride=1.0):
    points = load_ply_points(input_path)
    pca = PCA(n_components=3)
    pca.fit(points)
    axes = pca.components_
    centered = points - np.mean(points, axis=0)
    rotated = centered @ axes.T

    min_bound = rotated.min(axis=0)
    max_bound = rotated.max(axis=0)
    step = np.ceil((max_bound - min_bound - block_size) / stride).astype(int) + 1

    os.makedirs(output_dir, exist_ok=True)
    meta = []
    idx = 0
    for i in range(step[0]):
        for j in range(step[1]):
            for k in range(step[2]):
                center = min_bound + np.array([i, j, k]) * stride + 0.5 * block_size
                bound_min = center - 0.5 * block_size
                bound_max = center + 0.5 * block_size
                mask = np.all((rotated >= bound_min) & (rotated <= bound_max), axis=1)
                block = rotated[mask]
                if len(block) < 64:
                    continue
                block = (block - center) / block_size  # 归一化到 [-0.5, 0.5]
                fname = os.path.join(output_dir, f"block_{idx:04d}.ply")
                save_points_to_ply(block, fname)
                meta.append((fname, center, block_size))
                idx += 1
    print(f"[完成] 分块共保存 {idx} 个块")
    return axes, meta


def convert_all_predicted_pkl_to_ply(meta, contour_pkl_dir, contour_ply_dir):
    os.makedirs(contour_ply_dir, exist_ok=True)
    count = 0
    for f, _, _ in meta:
        block_id = os.path.splitext(os.path.basename(f))[0]
        input_pkl = os.path.join(contour_pkl_dir, block_id, "pred_64_10_pwl.pkl")
        output_ply = os.path.join(contour_ply_dir, f"{block_id}.ply")
        if os.path.exists(input_pkl):
            try:
                pkl_to_ply(input_pkl, output_ply)
                count += 1
            except Exception as e:
                print(f"[错误] 转换失败 {input_pkl}: {e}")
        else:
            print(f"[缺失] {input_pkl}")
    print(f"[完成] 成功转换 {count} 个预测结果为 .ply")


def merge_predictions(meta, input_dir, axes, output_file):
    merged = []
    for f, center, scale in meta:
        ply_path = os.path.join(input_dir, os.path.basename(f))
        if not os.path.exists(ply_path):
            continue
        pcd = o3d.io.read_point_cloud(ply_path)
        pred = np.asarray(pcd.points)
        if pred.shape[0] == 0:
            continue
        pred = pred * scale + center
        pred = pred @ axes  # 坐标转回原始空间

        bound_min = center - 0.5 * scale
        bound_max = center + 0.5 * scale
        mask = np.all((pred >= bound_min) & (pred <= bound_max), axis=1)
        pred = pred[mask]
        merged.append(pred)

    if merged:
        merged = np.vstack(merged)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged)
        o3d.io.write_point_cloud(output_file, merged_pcd)
        print(f"[完成] 合并点云保存至: {output_file}")
    else:
        print("[警告] No valid predicted blocks to merge.")


    if merged:
        merged = np.vstack(merged)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged)
        o3d.io.write_point_cloud(output_file, merged_pcd)
        print(f"[完成] 合并点云已保存: {output_file}")
    else:
        print("[警告] No valid predicted blocks to merge.")


if __name__ == "__main__":
    input_path = "NerVE-main/bridge/50w.ply"
    split_dir = "blocks"
    predict_pkl_dir = "pkl"
    predict_ply_dir = "ply"
    output_path = "NerVE-main/bridge/contour/merged_output.ply"

    axes, meta = pca_split(input_path, split_dir)

    for f, _, _ in meta:
        block_id = os.path.splitext(os.path.basename(f))[0]
        output_pkl = os.path.join(predict_pkl_dir, block_id, "pred_pwl.pkl")
        predict_contour(f, output_pkl)

    convert_all_predicted_pkl_to_ply(meta, predict_pkl_dir, predict_ply_dir)
    merge_predictions(meta, predict_ply_dir, axes, output_path)
