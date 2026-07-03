import os
import pickle
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import data
import utils
from time import time
def predict_contour(input_pc_path, output_ply_path):
    """
    整合点云预测和格式转换功能
    输入: 原始点云文件路径
    输出: 转换后的PLY文件路径
    """
    # ===== 1. 设置固定参数 =====
    ROOT_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(ROOT_path, 'results', 'train6410')
    output_path = os.path.join(ROOT_path, 'bridge')
    os.makedirs(output_path, exist_ok=True)
    device = torch.device('cuda')
    config = {
        'grid_size': 64,
        'cube_shift_mode': 'None'
    }

    # ===== 2. 提取文件名 =====
    file_name = os.path.basename(input_pc_path)
    name = os.path.splitext(file_name)[0]

    # ===== 3. 加载模型 =====
    model_cube = utils.load_model(os.path.join(model_path, 'cube'), device)
    model_face = utils.load_model(os.path.join(model_path, 'face'), device)
    model_geom = utils.load_model(os.path.join(model_path, 'geom'), device)

    # ===== 4. 创建数据集 =====
    single_config = config.copy()
    single_config['data_path'] = os.path.dirname(input_pc_path)
    single_config['file_list'] = None
    single_config['pc_file'] = file_name
    
    dataset = data.get_dataset('RawPCDataset')(single_config)

    # ===== 5. 执行预测 =====
    t0 = time()
    with torch.no_grad():
        model_input, info = dataset.get_data(0, normalize='cube_face')
        model_input = {key: val.cuda() for key, val in model_input.items()}
        model_input['info'] = info

        print(f'Processing: {name}')
        
        res = {}
        edge_cube, peid = model_cube.predict_curve(model_input)
        res.update(edge_cube)
        edge_face = model_face.predict_curve(model_input, peid)
        res.update(edge_face)

        model_input, info = dataset.get_data(0, normalize='geom')
        model_input = {key: val.cuda() for key, val in model_input.items()}        
        model_input['info'] = info
        edge_geom = model_geom.predict_curve(model_input, peid)
        res.update(edge_geom)

        # 创建输出目录
        curve_outputpath = os.path.join(output_path, name)
        os.makedirs(curve_outputpath, exist_ok=True)
        
        # 生成需要的pkl文件
        curve_outfile = os.path.join(curve_outputpath, 'pred_64_10_pwl_curve.pkl')
        utils.nerve2pwl(res, curve_outfile)

    # ===== 6. 转换PKL到PLY =====
    pkl_to_ply(curve_outfile, output_ply_path)
    print('Total time cost: ', time()-t0)

def pkl_to_ply(pkl_path, ply_path, verbose=True):
    """
    PKL转PLY格式 (内部函数)
    """
    try:
        # 读取pkl文件
        if verbose: 
            print(f"Reading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 提取点云数据
        if 'points' not in data:
            raise KeyError("'points' field not found in pkl file")
            
        points = np.asarray(data['points'], dtype=np.float32)
        if verbose:
            print(f"Loaded {points.shape[0]} points | Dimensions: {points.shape[1]}")

        # 验证坐标维度
        if points.shape[1] not in [3, 6]:
            print("Warning: Unusual coordinate dimensions (supports 3D or 6D with RGB)")

        # 创建PLY数据结构
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if points.shape[1] == 6:
            dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
        vertex = np.array([tuple(p) for p in points], dtype=dtype)
        vertex_element = PlyElement.describe(vertex, 'vertex')

        # 写入PLY文件
        if verbose:
            print(f"Generating {ply_path}...")
        PlyData([vertex_element], text=True).write(ply_path)
        
        if verbose:
            print(f"Conversion complete! Saved to {ply_path}")

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise

# 使用示例
if __name__ == "__main__":
    input_file = "NerVE-main/PCA_length_10/00000001/pc_obj.ply"  # 输入点云文件
    output_file = "bridge/output.ply"      # 输出PLY文件
    predict_contour(input_file, output_file)