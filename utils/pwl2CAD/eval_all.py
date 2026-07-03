import os
import pickle
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d  # 导入Open3D，核心修改

# --------------------------
# 核心重构：Open3D通用点云加载函数（支持PLY/PCD/XYZ/OBJ等，新增纯坐标TXT适配）
# --------------------------
def load_point_cloud(pc_path):
    """
    用Open3D读取点云文件，返回(N,3)的三维坐标数组，支持：
    - PLY/PCD/XYZ/OBJ等Open3D原生支持格式
    - 纯坐标TXT（每行3个浮点数，x y z）
    :param pc_path: 点云文件路径
    :return: (N,3) np.array(float32)，点云的x/y/z坐标
    """
    # 提取文件后缀（小写）
    file_suffix = os.path.splitext(pc_path)[-1].lower()
    
    try:
        # 专门处理纯坐标TXT文件（核心修改）
        if file_suffix == ".txt":
            # 读取纯坐标TXT：每行3个数值，x y z
            point_cloud = np.loadtxt(pc_path, dtype=np.float32)
            # 校验维度：必须是(N,3)
            if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
                raise ValueError(f"TXT文件格式错误：需每行3个坐标，当前形状为{point_cloud.shape}")
            return point_cloud
        
        # 其他格式用Open3D读取（原有逻辑不变）
        pcd = o3d.io.read_point_cloud(pc_path)
        # 检查点云是否为空（文件损坏/无顶点时触发）
        if not pcd.has_points():
            raise ValueError("点云文件无有效顶点数据")
        # 提取点云坐标，转换为(N,3)的float32数组
        point_cloud = np.asarray(pcd.points, dtype=np.float32)
        return point_cloud
    
    except Exception as e:
        raise ValueError(f"读取点云文件失败 {pc_path}：{str(e)}")

def load_cad_curve(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    edge_points = []
    for name, val in data.items():
        if name == 'endpoints':
            continue
        points = val['points']
        edge_points.append(points[1:-1])
    edge_points = np.concatenate(edge_points, axis=0)
    endpts = data['endpoints']
    points = np.concatenate([edge_points, endpts], axis=0)
    return points

def convert_cad_to_pwl(data_path, output_path):
    def get_edge(_v0, _v1, _vidx):
        e1 = np.r_[_v0, _vidx]
        e2 = np.r_[_vidx, _v1]
        e12 = np.vstack([e1, e2]).T.astype(int)
        return e12.tolist()

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    edges = []
    edge_points = []
    endpts = data['endpoints']
    tree_endpts = KDTree(endpts) if len(endpts) > 0 else None

    current_idx = len(endpts)
    for name, val in data.items():
        if name == 'endpoints':
            continue
        
        points = val['points']
        is_closed = val['closed']
        if is_closed:
            edge_points.append(points)
            pts_idx = list(range(current_idx, current_idx + points.shape[0]))
            next_idx = np.roll(pts_idx, -1)
            tmp_edges = np.vstack([pts_idx, next_idx]).T.astype(int)
            edges.extend(tmp_edges)
            current_idx += points.shape[0]
            continue
        
        if tree_endpts is None:
            raise ValueError('No endpoints in open curves')

        num_inpts = points.shape[0] - 2
        _, v0 = tree_endpts.query(points[0])
        _, v1 = tree_endpts.query(points[-1])
        if num_inpts == 0:
            edges.append([v0, v1])
            continue

        edge_points.append(points[1:-1])
        pts_idx = list(range(current_idx, current_idx + num_inpts))
        tmp_edges = get_edge(v0, v1, pts_idx)
        edges.extend(tmp_edges)
        current_idx += num_inpts

    edge_points = np.concatenate(edge_points, axis=0)
    points = np.concatenate([endpts, edge_points], axis=0) if len(endpts) > 0 else edge_points
    res = {'points': points, 'edges': edges}
    with open(output_path, 'wb') as f:
        pickle.dump(res, f)

def load_pwl_curve(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    pts, edges = data['points'], data['edges']
    midpts = np.mean(pts[np.asarray(edges)], axis=1)
    samples = np.concatenate([pts, midpts], axis=0)
    return samples

def load_step_curve(data_path, offset=None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    pts, edges = data['points'], data['edges']
    midpts = np.mean(pts[np.asarray(edges)], axis=1)
    samples = np.concatenate([pts, midpts], axis=0)
    if offset is not None:
        samples += offset
    return samples

def calc_metrics(pred, gt, max_HD=False, threshold=0.1):
    """
    计算曲线匹配的各项指标
    pred: (N,3) 预测点云坐标（Open3D读取的任意格式点云）
    gt: (M,3) 真实曲线的采样点
    threshold: 距离阈值，小于此值认为点匹配成功
    """
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)

    dist_pred2gt, _ = tree_gt.query(pred)
    dist_gt2pred, _ = tree_pred.query(gt)

    chamfer_dist = np.mean(dist_pred2gt**2) + np.mean(dist_gt2pred**2)
    bhaussdorf_dist = (max(dist_pred2gt.max(), dist_gt2pred.max()) 
                      if max_HD 
                      else (dist_pred2gt.max() + dist_gt2pred.max()) / 2)

    TP = np.sum(dist_pred2gt < threshold)
    FP = len(dist_pred2gt) - TP
    FN = np.sum(dist_gt2pred >= threshold)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    # 修复MCC除0错误，鲁棒性优化
    denominator = np.sqrt((TP + FP) * (TP + FN) * max(FP, 1e-8) * max(FN, 1e-8))
    mcc = (TP * 0 - FP * FN) / denominator if denominator != 0 else 0.0

    return {
        'CD': chamfer_dist, 'BHD': bhaussdorf_dist,
        'Precision': precision, 'Recall': recall,
        'IoU': iou, 'MCC': mcc
    }

if __name__ == "__main__":
    # --------------------------
    # 配置参数（仅需改这里，与你的项目路径匹配）
    # --------------------------
    # pred_root = "NerVE-main/predict_N64D"    # 预测点云根目录
    pred_root = "traditional method/results32"
    gt_root = "NerVE-main/NerVE64Dataset"    # 真实数据根目录
    result_file = "NerVE-main/EVAL/eval_TraMeth.txt_val32.txt"  # 结果输出文件
    input_list = "NerVE-main/NerVE64Dataset/val.txt"    # 测试集列表
    distance_threshold = 0.01                 # 距离阈值
    # PRED_POINT_CLOUD_NAME = "pred_DG_pwl.pkl"       # 你的预测点云文件名（核心！改这里即可兼容其他格式）
    PRED_POINT_CLOUD_NAME = "TraMeth.txt"

    # 读取目标文件夹列表
    with open(input_list, "r") as f:
        target_folders = [line.strip() for line in f if line.strip()]

    # 初始化结果文件
    with open(result_file, "w") as f:
        f.write("Folder_ID\tCD\tBHD\tPrecision\tRecall\tIoU\n")
        # f.write("Folder_ID\tPrecision\tRecall\tIoU\n")

    processed_count = 0

    # 批量处理每个文件夹
    for folder in target_folders:
        if not (folder.isdigit() and len(folder) == 8):
            print(f"跳过格式错误的文件夹: {folder}")
            continue

        pred_folder = os.path.join(pred_root, folder)
        gt_folder = os.path.join(gt_root, folder)
        if not (os.path.exists(pred_folder) and os.path.exists(gt_folder)):
            print(f"文件夹 {folder} 不存在，已跳过")
            continue

        # 构建文件路径（预测点云仅需改上面的文件名，无需改这里）
        pred_path = os.path.join(pred_folder, PRED_POINT_CLOUD_NAME)
        gt_curve_path = os.path.join(gt_folder, "nerve_reso64_curve.pkl")
        gt_step_path = os.path.join(gt_folder, "step_curve_no_offset.pkl")

        # 验证文件存在
        required_files = [pred_path, gt_curve_path, gt_step_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"文件夹 {folder} 缺少文件，已跳过")
            continue

        try:
            # 加载真实数据（原逻辑不变）
            with open(gt_curve_path, "rb") as f:
                offset = pickle.load(f)["stable_offset"]
            gt_samples = load_step_curve(gt_step_path, offset)

            # --------------------------
            # 核心加载逻辑：通用点云读取（支持多格式，无需修改）
            # --------------------------
            file_suffix = os.path.splitext(pred_path)[-1].lower()
            if file_suffix in [".ply", ".pcd", ".xyz", ".obj", ".txt"]:
                # Open3D支持的点云格式 + 纯坐标TXT，统一用load_point_cloud读取
                pred_samples = load_point_cloud(pred_path)
            elif file_suffix == ".pkl":
                # 保留原PKL格式支持，兼容旧预测数据
                pred_samples = load_pwl_curve(pred_path)
            else:
                raise ValueError(f"不支持的文件格式：{file_suffix}，支持ply/pcd/xyz/obj/txt/pkl")

            # 计算指标并写入结果（原逻辑不变）
            metrics = calc_metrics(pred_samples, gt_samples, threshold=distance_threshold)
            with open(result_file, "a") as f:
                f.write(
                    f"{folder}\t"
                    f"{metrics['CD']:.8e}\t"
                    f"{metrics['BHD']:.8e}\t"
                    f"{metrics['Precision']:.6f}\t"
                    f"{metrics['Recall']:.6f}\t"
                    f"{metrics['IoU']:.6f}\n"
                )

            processed_count += 1
            del offset, pred_samples, gt_samples, metrics  # 释放内存

        except Exception as e:
            print(f"处理 {folder} 时出错: {str(e)}")
            continue

    print(f"处理完成，有效处理数量: {processed_count}/{len(target_folders)}")