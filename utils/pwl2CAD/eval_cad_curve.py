import os,pickle
import numpy as np
from scipy.spatial import KDTree

def load_cad_curve(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    edge_points = []

    for name,val in data.items():
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
    if len(endpts) > 0:
        tree_endpts = KDTree(endpts)
    else:
        tree_endpts = None

    current_idx = len(endpts)
    for name, val in data.items():
        if name == 'endpoints':
            continue
        
        points = val['points']
        is_closed = val['closed']
        if is_closed:
            edge_points.append(points)
            pts_idx = list(range(current_idx, current_idx+points.shape[0]))
            next_idx = np.roll(pts_idx, -1)
            tmp_edges = np.vstack([pts_idx, next_idx]).T.astype(int)
            edges.extend(tmp_edges)
            current_idx += points.shape[0]
            continue
        
        if tree_endpts is None:
            raise ValueError('Not endpts in open curves')

        num_inpts = points.shape[0] - 2
        _,v0 = tree_endpts.query(points[0])
        _,v1 = tree_endpts.query(points[-1])
        if num_inpts == 0:
            edges.append([v0,v1])
            continue

        edge_points.append(points[1:-1])
        pts_idx = list(range(current_idx, current_idx+num_inpts))
        tmp_edges = get_edge(v0, v1, pts_idx)
        edges.extend(tmp_edges)
        current_idx += num_inpts

    edge_points = np.concatenate(edge_points, axis=0)
    if len(endpts) > 0:
        points = np.concatenate([endpts, edge_points], axis=0)
    else:
        points = edge_points
    res = {
        'points': points,
        'edges': edges
    }
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

def calc_loss(pred, gt, max_HD=False):
    """
    pred: (N, 3) samples of predicted curve
    gt: (M,3) samples of gt curve
    """
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)

    dist_pred2gt, _ = tree_gt.query(pred)
    dist_gt2pred, _ = tree_pred.query(gt)

    chamfer_dist = np.mean(dist_pred2gt**2) + np.mean(dist_gt2pred**2)
    if max_HD:
        bhaussdorf_dist = max(dist_pred2gt.max(), dist_gt2pred.max())
    else:
        bhaussdorf_dist = (dist_pred2gt.max() + dist_gt2pred.max()) / 2
    return {
        'CD': chamfer_dist,
        'BHD': bhaussdorf_dist
    }


# if __name__ == "__main__":
#     pred_curve_path = '/path/to/your/pred_nerve_pwl_curve.pkl'
#     # pred_cad_curve_path = '/path/to/your/cad_pwl_curve.pkl'
#     gt_nerve_curve_path = '/path/to/your/nerve_reso64_curve.pkl'
#     gt_step_path = '/path/to/your/step_curve_no_offset.pkl'

#     offset = pickle.load(open(gt_nerve_curve_path, 'rb'))['stable_offset']
#     pred_pwl_samples = load_pwl_curve(pred_curve_path)
#     # pred_pwl_samples = load_pwl_curve(pred_cad_curve_path)
#     gt_step_samples = load_step_curve(gt_step_path, offset)

#     err = calc_loss(pred_pwl_samples, gt_step_samples)


#批量处理（处理列表包含文件）
if __name__ == "__main__":
    pred_root = "NerVE-main/predict_N64D"
    gt_root = "NerVE-main/NerVE64Dataset"
    result_file = "NerVE-main/EVAL/eval_X_val.txt"
    input_list = "NerVE-main/NerVE64Dataset/val.txt"  # 包含需要处理文件夹名称的文本文件

    # 读取输入列表文件
    with open(input_list, "r") as f:
        # 过滤空行和去除首尾空格
        target_folders = [line.strip() for line in f if line.strip()]

    # 初始化结果文件
    with open(result_file, "w") as f:
        f.write("Folder_ID\tCD\tHD\n")

    # 处理指定文件夹
    for folder in target_folders:
        # 验证文件夹格式
        if not (folder.isdigit() and len(folder) == 8):
            print(f"跳过格式错误的文件夹名: {folder}")
            continue

        # 构建路径
        pred_folder = os.path.join(pred_root, folder)
        gt_folder = os.path.join(gt_root, folder)
        
        # 验证主文件夹存在
        if not (os.path.exists(pred_folder) and os.path.exists(gt_folder)):
            print(f"文件夹 {folder} 不存在于数据目录，已跳过")
            continue

        # 构建文件路径
        # NerVE-main/predict_N64D/00000003/pred_A_pwl.pkl
        # NerVE-main/NerVE64Dataset/00000003/nerve_reso64_curve.pkl
        pred_path = os.path.join(pred_folder, "pred_M_pwl.pkl")
        # pred_path = os.path.join(pred_folder, "cad_pwl_curve.pkl")
        gt_curve_path = os.path.join(gt_folder, "nerve_reso64_curve.pkl")#这个文件有stable_offset，还有points，和edges
        gt_step_path = os.path.join(gt_folder, "step_curve_no_offset.pkl")
        
        # 验证必需文件存在
        required_files = [pred_path, gt_curve_path, gt_step_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"文件夹 {folder} 缺少必需文件，已跳过")
            continue

        try:
            # 加载数据（使用with确保文件及时关闭）
            with open(gt_curve_path, "rb") as f:
                offset = pickle.load(f)["stable_offset"]
            pred_samples = load_pwl_curve(pred_path)
            gt_samples = load_step_curve(gt_step_path, offset)
            
            # 计算并写入结果
            metrics = calc_loss(pred_samples, gt_samples)
            with open(result_file, "a") as f:
                f.write(f"{folder}\t{metrics['CD']:.8e}\t{metrics['BHD']:.8e}\n")
            
            # 主动释放内存
            del offset, pred_samples, gt_samples, metrics
        except Exception as e:
            print(f"处理 {folder} 时发生错误: {str(e)}")
            continue

print(f"处理完成，有效处理数量: {len(target_folders)}")
    