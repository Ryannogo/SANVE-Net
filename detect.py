import os
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from tqdm import tqdm
import trimesh
import glob

from transfer.fold_trans import fold_transfer  # 修改为你的实际路径
from contour_prediction import predict_contour 
from transfer.pre_ply import pkl_to_ply
from transfer.center_scale import pc_normalize,network_input_normalize
from transfer.denorm import inverse_pc_normalize,batch_inverse_normalize

# ---------- 参数配置 ----------
RATIO = [18, 3, 1]     # 分块比例（按主方向、次主方向、第三方向）
SCALE = 1.1            # 分块时的缩放比例（带重叠）
CROP_SCALE = 1       # 最终裁剪contour时使用的比例（无重叠）
MIN_POINTS = 200      # 最小点数量阈值，少于此值不保存
RAW_DIR = "NerVE-main/Cut50/Relu/raw"
PKL_DIR = "NerVE-main/Cut50/Relu/pkl"
PLY_DIR = "NerVE-main/Cut50/Relu/ply"
MERGED_PLY = "NerVE-main/Cut50/Relu/final_merged.ply"
CONTOUR = "NerVE-main/Cut50/Relu/contour"
CROPPED_CONTOUR = "NerVE-main/Cut50/Relu/cropped_contour"  # 裁剪后的contour保存目录

for d in [RAW_DIR, PKL_DIR, PLY_DIR, CONTOUR, CROPPED_CONTOUR]:
    os.makedirs(d, exist_ok=True)

# ---------- 计算主方向并转换点云到主方向坐标系 ----------
def compute_main_axes_and_transform(pcd):
    """计算主方向并返回转换后的点云和相关参数"""
    pts = np.asarray(pcd.points)
    pca = PCA(n_components=3)
    pca.fit(pts)
    
    components = pca.components_  # 主方向向量（3x3矩阵，行向量）
    mean = pca.mean_             # 点云中心
    
    # 将点云转换到主方向坐标系（中心化后旋转）
    pts_centered = pts - mean
    pts_aligned = pts_centered @ components.T  # 转换到主方向坐标系
    
    sizes = np.sqrt(pca.explained_variance_)   # 各方向尺度
    
    return {
        "components": components,    # 主方向矩阵
        "mean": mean,                # 原始坐标系中的中心点
        "pts_aligned": pts_aligned,  # 主方向坐标系中的点
        "sizes": sizes,              # 各方向尺度
        "original_points": pts       # 原始坐标系中的点
    }

# ---------- 在主方向坐标系上分块 & 裁剪 ----------
def split_and_crop_in_pca_space(transform_info, ratio=RATIO, scale=SCALE):
    pts_aligned = transform_info["pts_aligned"]
    components = transform_info["components"]
    
    # 确定主方向（0：主方向，1：次主方向，2：第三方向）
    main_dir = 0
    min_bound = pts_aligned.min(axis=0)
    max_bound = pts_aligned.max(axis=0)
    total_sizes = max_bound - min_bound  # 各方向总长度
    
    # 输出主方向和分块参数信息
    print(f"\n主方向坐标系信息：")
    print(f"  各方向总长度: {total_sizes}")
    print(f"  主方向索引: {main_dir} (长度: {total_sizes[main_dir]})")
    print(f"  分块比例: {ratio}")
    print(f"  分块缩放比例: {scale}")
    print(f"  最终裁剪比例: {CROP_SCALE}")
    print(f"  最小保存点数量: {MIN_POINTS}")
    
    # 分块数量映射到各方向
    dims = np.argsort(total_sizes)[::-1]  # 按长度排序的方向索引
    blocks = np.array(ratio)[np.argsort(dims)]
    print(f"  各方向分块数: {blocks} (主方向分块数: {blocks[main_dir]})")
    
    # 计算步长
    steps = total_sizes / blocks
    print(f"  各方向步长: {steps} (主方向步长: {steps[main_dir]})")
    
    # 分块时使用的缩放比例（带重叠）
    coverage = steps * scale
    gaps = steps - coverage  # 正值为间隙，负值为重叠
    print(f"  分块时理论间隙/重叠: {gaps} (主方向: {gaps[main_dir]:.4f})")
    
    # 最终裁剪时使用的比例（无重叠）
    final_coverage = steps * CROP_SCALE
    final_gaps = steps - final_coverage
    print(f"  最终裁剪后间隙/重叠: {final_gaps} (主方向: {final_gaps[main_dir]:.4f})")
    
    # 存储两种裁剪范围：带重叠的（用于分块）和无重叠的（用于最终裁剪）
    crop_infos = []          # 带重叠的裁剪范围（用于分块）
    final_crop_infos = []    # 无重叠的裁剪范围（用于contour最终裁剪）
    centers_aligned = []     # 主方向坐标系下的中心
    
    for i in range(blocks[0]):
        for j in range(blocks[1]):
            for k in range(blocks[2]):
                # 计算主方向坐标系下的块中心
                center = np.array([
                    min_bound[0] + (i + 0.5) * steps[0],
                    min_bound[1] + (j + 0.5) * steps[1],
                    min_bound[2] + (k + 0.5) * steps[2]
                ])
                
                # 带重叠的裁剪范围（用于分块）
                half_box = steps * scale / 2
                min_box = center - half_box
                max_box = center + half_box
                crop_infos.append((min_box, max_box))
                
                # 无重叠的裁剪范围（用于最终裁剪contour）
                final_half_box = steps * CROP_SCALE / 2
                final_min_box = center - final_half_box
                final_max_box = center + final_half_box
                final_crop_infos.append((final_min_box, final_max_box))
                
                centers_aligned.append(center)
                
                # 输出主方向相邻块的关系
                if i > 0 and j == 0 and k == 0 and (i % 5 == 0 or i == blocks[0]-1):
                    # 分块时的重叠情况
                    prev_max = (min_bound[main_dir] + (i-1 + 0.5) * steps[main_dir]) + half_box[main_dir]
                    curr_min = center[main_dir] - half_box[main_dir]
                    actual_gap = curr_min - prev_max
                    print(f"  分块时主方向块 {i-1}与{i}: 实际{'间隙' if actual_gap>0 else '重叠'}={abs(actual_gap):.4f}")
                    
                    # 最终裁剪后的情况
                    final_prev_max = (min_bound[main_dir] + (i-1 + 0.5) * steps[main_dir]) + final_half_box[main_dir]
                    final_curr_min = center[main_dir] - final_half_box[main_dir]
                    final_actual_gap = final_curr_min - final_prev_max
                    print(f"  裁剪后主方向块 {i-1}与{i}: 实际{'间隙' if final_actual_gap>0 else '重叠'}={abs(final_actual_gap):.4f}")
    
    return crop_infos, final_crop_infos, centers_aligned, transform_info

# ---------- 在主方向坐标系裁剪并转换回原始坐标系保存 ----------
def crop_in_pca_space_and_save(crop_infos, transform_info, save_dir):
    """在主方向坐标系裁剪，然后转换回原始坐标系保存，筛选少于阈值点的分块"""
    saved_files = []
    empty_blocks = 0  # 统计空块数量
    pts_aligned = transform_info["pts_aligned"]
    original_points = transform_info["original_points"]
    components = transform_info["components"]
    mean = transform_info["mean"]
    
    print("\n📦 [1/4] 在主方向坐标系裁剪并保存...")
    print(f"  点数量筛选阈值: 少于{MIN_POINTS}点的分块将不保存")
    
    # 主方向点云分布范围（用于验证）
    main_dir = 0
    min_main = np.min(pts_aligned[:, main_dir])
    max_main = np.max(pts_aligned[:, main_dir])
    print(f"  主方向坐标系下点云分布: [{min_main:.4f}, {max_main:.4f}]")
    
    for idx, (min_box, max_box) in enumerate(tqdm(crop_infos, desc="裁剪 RAW")):
        # 在主方向坐标系下裁剪
        mask = np.all((pts_aligned >= min_box) & (pts_aligned <= max_box), axis=1)
        sub_pts_aligned = pts_aligned[mask]
        point_count = len(sub_pts_aligned)
        
        # 筛选点数量少于阈值的分块
        if point_count < MIN_POINTS:
            empty_blocks += 1
            # 每20个空块输出一次提示（避免过多输出）
            if empty_blocks % 20 == 0:
                print(f"  分块 {idx}: 点数量不足 ({point_count} < {MIN_POINTS})，不保存")
            continue
        
        # 将裁剪后的点转换回原始坐标系
        sub_pts_original = (sub_pts_aligned @ components) + mean
        
        # 创建点云并保存
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(sub_pts_original)
        
        fname = os.path.join(save_dir, f"{idx:08d}.ply")
        o3d.io.write_point_cloud(fname, sub_pcd)
        saved_files.append(fname)
        
        # 输出主方向覆盖信息（每10个分块）
        if idx % 10 == 0:
            sub_min_main = np.min(sub_pts_aligned[:, main_dir])
            sub_max_main = np.max(sub_pts_aligned[:, main_dir])
            print(f"  分块 {idx}: 主方向覆盖 [{sub_min_main:.4f}, {sub_max_main:.4f}], 点数: {point_count}")
    
    print(f"  处理完成: 共保存 {len(saved_files)} 个有效分块点云，过滤 {empty_blocks} 个点数量不足的分块")
    return saved_files


def run_processing(raw_dir):
    print("\n🛠️ [2/4] fold_transfer：生成 raw/*/pc_obj.ply ...")
    fold_transfer(raw_dir)

    print("\n🔍 [3/4] contour_prediction：开始预测边线曲线 ...")
    predict_contour(RAW_DIR, PKL_DIR)

    print("\n📄 [4/4] pkl_to_ply：转换预测结果为 PLY ...")
    all_txt_path = os.path.join(RAW_DIR, "all.txt")
    if os.path.exists(all_txt_path):
        with open(all_txt_path, 'r') as f:
            all_folders = [line.strip() for line in f.readlines()]
    for name in tqdm(all_folders, desc="PklToPly"):
        pkl_path = os.path.join(PKL_DIR, name, "pred_M_pwl.pkl")
        ply_path = os.path.join(PLY_DIR, f"{name}.ply")
        if os.path.exists(pkl_path):
            pkl_to_ply(pkl_path, ply_path)
        else:
            print(f"⚠️ 缺失: {pkl_path}")

# ---------- 计算逆归一化的参数 ----------
def center_scale(input_dir, output_dir, knn_size=8, grid_size=64, mode='cube_face'):
    """
    批量处理raw目录下的PLY文件，直接输出到指定文件夹
    输入结构: raw_dir/00000000/pc_obj.ply
    输出结构: output_dir/00000000.npz
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有符合模式的PLY文件
    ply_pattern = os.path.join(input_dir, "*", "pc_obj.ply")
    ply_files = glob.glob(ply_pattern)
    
    if not ply_files:
        print(f"警告: 在 {input_dir} 下未找到符合条件的PLY文件")
        return
    
    # 遍历所有找到的PLY文件
    for file_path in ply_files:
        try:
            # 解析路径结构，获取子文件夹名称作为输出文件名
            folder_name = os.path.basename(os.path.dirname(file_path))
            
            # 读取点云
            pointcloud = trimesh.load(file_path)
            pc = pointcloud.vertices
            
            # 执行归一化
            result = network_input_normalize(pc, knn_size=knn_size, grid_size=grid_size, mode=mode)
            
            # 生成输出文件路径
            output_path = os.path.join(output_dir, f"{folder_name}.npz")
            
            # 保存为NPZ文件
            np.savez(
                output_path,
                center=result['center'],
                scale=result['scale'],
                grid_size=result['grid_size'],
                pc_norm=result['pc_norm'],
                knn_pos=result['knn_pos'],
                knn_idx=result['knn_idx']
            )
            
            print(f"处理成功: {file_path} -> {output_path}")
            
        except Exception as e:
            print(f"处理失败 {file_path}: {str(e)}")
     

# ---------- 裁剪contour点云（去除重叠部分） ----------
def crop_contour_files(contour_dir, final_crop_infos, transform_info, output_dir):
    """
    使用无重叠的裁剪范围对contour点云进行精确裁剪，去除因SCALE>1.0产生的重叠部分
    
    :param contour_dir: 包含contour点云的目录
    :param final_crop_infos: 无重叠的裁剪范围
    :param transform_info: 点云转换信息（包含主方向等）
    :param output_dir: 裁剪后的点云保存目录
    """
    # 获取所有contour PLY文件
    contour_files = [f for f in os.listdir(contour_dir) if f.endswith('.ply')]
    if not contour_files:
        print(f"警告: 在 {contour_dir} 下未找到PLY文件")
        return
    
    components = transform_info["components"]
    mean = transform_info["mean"]
    
    print(f"\n✂️ 开始裁剪 {len(contour_files)} 个contour点云文件（去除重叠部分）...")
    
    for file in tqdm(contour_files, desc="裁剪Contour"):
        try:
            # 提取文件索引（假设文件名格式为00000000.ply）
            file_idx = int(os.path.splitext(file)[0])
            
            # 检查索引是否在有效范围内
            if file_idx < 0 or file_idx >= len(final_crop_infos):
                print(f"警告: 文件 {file} 的索引超出范围，跳过")
                continue
            
            # 获取无重叠的裁剪范围（关键修改点）
            min_box, max_box = final_crop_infos[file_idx]
            
            # 加载contour点云
            file_path = os.path.join(contour_dir, file)
            mesh = trimesh.load(file_path)
            pts = mesh.vertices
            
            # 将点转换到主方向坐标系
            pts_centered = pts - mean
            pts_aligned = pts_centered @ components.T  # 转换到主方向坐标系
            
            # 应用无重叠的裁剪范围
            mask = np.all((pts_aligned >= min_box) & (pts_aligned <= max_box), axis=1)
            cropped_pts = pts[mask]
            
            # 保存裁剪后的点云
            if len(cropped_pts) > 0:
                cropped_mesh = trimesh.points.PointCloud(cropped_pts)
                output_path = os.path.join(output_dir, file)
                cropped_mesh.export(output_path)
            else:
                print(f"警告: 文件 {file} 裁剪后没有剩余点")
                
        except Exception as e:
            print(f"处理 {file} 失败: {str(e)}")
    
    print(f"Contour裁剪完成（已去除重叠部分），结果保存在 {output_dir}")


# ---------- 拼接 ----------
def merge_ply_files(input_dir, output_file):
    """将目录中的所有PLY文件拼接为一个单一的点云文件"""
    # 获取所有PLY文件
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    if not ply_files:
        print(f"错误：输入目录中没有PLY文件！ {input_dir}")
        return

    print(f"发现 {len(ply_files)} 个PLY文件进行拼接")
    # 初始化点云列表
    all_points = []
    # 进度条处理
    for ply_file in tqdm(ply_files, desc="拼接点云"):
        try:
            # 加载点云
            mesh = trimesh.load(os.path.join(input_dir, ply_file))
            
            # 确保是点云
            if not isinstance(mesh, trimesh.PointCloud):
                # 尝试提取顶点
                if hasattr(mesh, 'vertices'):
                    all_points.append(mesh.vertices)
                else:
                    raise ValueError(f"文件 {ply_file} 不是有效的点云格式")
            else:
                all_points.append(mesh.vertices)
                
        except Exception as e:
            print(f"\n处理失败: {ply_file}")
            print(f"错误类型: {type(e).__name__}, 详情: {str(e)}")
    
    # 检查是否有点云数据
    if not all_points:
        print("错误：没有成功加载任何点云数据")
        return
    
    # 拼接所有点云
    merged_points = np.vstack(all_points)
    
    # 创建点云对象
    merged_pc = trimesh.points.PointCloud(merged_points)
    
    # 保存结果
    merged_pc.export(output_file)
    print(f"\n拼接完成！总点数: {len(merged_points):,}")
    print(f"输出文件: {output_file}")


# ---------- 主函数 ----------
def main(ply_path):
    
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"🔧 加载输入点云: {ply_path}")
    transform_info = compute_main_axes_and_transform(pcd)
    print("📐 计算主方向并分块 ...")
    # 获取两种裁剪范围：带重叠的和无重叠的
    crop_infos, final_crop_infos, centers_aligned, transform_info = split_and_crop_in_pca_space(transform_info)
    saved_files = crop_in_pca_space_and_save(crop_infos, transform_info, RAW_DIR)

    run_processing(RAW_DIR)

    center_scale(RAW_DIR, PLY_DIR, knn_size=8, grid_size=64, mode='cube_face')#生成逆归一化需要的参数
    batch_inverse_normalize(PLY_DIR, PLY_DIR, CONTOUR)#逆归一化
    
    # 使用无重叠的裁剪范围裁剪contour
    crop_contour_files(CONTOUR, final_crop_infos, transform_info, CROPPED_CONTOUR)

    # 使用裁剪后的contour点云进行拼接
    merge_ply_files(CROPPED_CONTOUR, MERGED_PLY)

if __name__ == "__main__":
    main("NerVE-main/Cut50/50w点桥梁.ply")  # 请将 input.ply 替换为你的输入文件路径
    