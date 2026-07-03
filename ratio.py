import numpy as np
import open3d as o3d
import sys

def calculate_principal_direction(pcd):
    """计算点云的主方向"""
    # 计算协方差矩阵
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    cov_matrix = np.dot(points_centered.T, points_centered) / len(points_centered)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 特征向量对应主方向，按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvectors

def calculate_dimensions_ratio(pcd):
    """计算点云在主方向上的长宽高比例"""
    # 计算主方向
    principal_directions = calculate_principal_direction(pcd)
    
    # 将点云转换到主方向坐标系
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    points_transformed = np.dot(points_centered, principal_directions)
    
    # 计算各方向上的范围
    x_range = np.max(points_transformed[:, 0]) - np.min(points_transformed[:, 0])
    y_range = np.max(points_transformed[:, 1]) - np.min(points_transformed[:, 1])
    z_range = np.max(points_transformed[:, 2]) - np.min(points_transformed[:, 2])
    
    # 按最大值归一化
    dimensions = np.array([x_range, y_range, z_range])
    max_dim = np.max(dimensions)
    normalized_dimensions = dimensions / max_dim
    
    # 找到最小非零值并缩放，使得该值为1
    non_zero_dims = normalized_dimensions[normalized_dimensions > 0]
    min_non_zero = np.min(non_zero_dims)
    ratio = normalized_dimensions / min_non_zero
    
    # 保留一位小数
    ratio = np.round(ratio, 1)
    
    return ratio

def main():
    # if len(sys.argv) != 2:
    #     print("用法: python predict/ratio.py NerVE-main/Cut50/50w点桥梁.ply")
    #     print("支持的格式: .ply, .pcd, .xyz, .xyzrgb, .xyzn, .pts")
    #     sys.exit(1)
    # #50w点桥梁 17.9:4.0:1.0,
    # file_path = sys.argv[1]
    file_path ="score-denoise-main/data/PUNet/pointclouds/train/50w_denoised.ply"  
    
    try:
        # 读取点云
        pcd = o3d.io.read_point_cloud(file_path)
        
        if pcd.is_empty():
            print(f"错误: 无法读取点云文件 {file_path} 或文件为空")
            sys.exit(1)
        
        # 1. 输出点云的数量
        point_count = len(pcd.points)
        print(f"点云数量: {point_count}")
        
        # 2. 计算并输出长宽高比例
        ratio = calculate_dimensions_ratio(pcd)
        print(f"点云长宽高比例: {ratio[0]}:{ratio[1]}:{ratio[2]}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()    