import os, pickle
import torch
import data, utils

import os.path as op
from time import time

ROOT_path = os.path.dirname(os.path.abspath(__file__))
model_path = op.join(ROOT_path, 'results', 'train6410')  # 模型路径
output_path = op.join(ROOT_path, 'baybridge')  # 输出路径
os.makedirs(output_path, exist_ok=True)

device = torch.device('cuda')
config = {
    'grid_size': 64,
    'cube_shift_mode': 'None'
}

# 设置要处理的单个文件路径
single_file_path = "baybridge"  # 修改为实际的文件路径
file_name = os.path.basename(single_file_path)
name = os.path.splitext(file_name)[0]  # 获取文件名（不带扩展名）作为输出文件夹名

# 加载模型
model_cube = utils.load_model(op.join(model_path, 'cube'), device)
model_face = utils.load_model(op.join(model_path, 'face'), device)
model_geom = utils.load_model(op.join(model_path, 'geom'), device)

# 创建数据集配置
single_config = config.copy()
single_config['data_path'] = os.path.dirname(single_file_path)
single_config['file_list'] = None  # 不使用文件列表
single_config['pc_file'] = file_name

# 创建数据集
dataset = data.get_dataset('RawPCDataset')(single_config)

t0 = time()
with torch.no_grad():
    # 只处理第一个（也是唯一的）文件
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

    curve_outputpath = os.path.join(output_path, name)
    os.makedirs(curve_outputpath, exist_ok=True)
    curve_outfile = os.path.join(curve_outputpath, 'pred_00000_pwl_curve.pkl')
    nerve_outfile = os.path.join(curve_outputpath, 'pred_00000.pkl')  # 输出文件名
    utils.nerve2pwl(res, curve_outfile)
    with open(nerve_outfile, 'wb') as f:
        pickle.dump(res, f)

print('Done, time cost: ', time()-t0)