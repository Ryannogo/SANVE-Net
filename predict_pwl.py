import os, pickle
import torch
import data, utils

import os.path as op
from time import time
#line9,10,18,53,54
ROOT_path = os.path.dirname(os.path.abspath(__file__))
model_path = op.join(ROOT_path, 'results', 'train_dk12')#模型路径
output_path = op.join(ROOT_path, 'predict_N64D')#输出路径**************************
os.makedirs(output_path, exist_ok=True)

device = torch.device('cuda')
config = {#输入被预测的文件路径
    'data_path': op.join(ROOT_path,'NerVE64Dataset'),#*****************************
    'file_list': op.join(ROOT_path, 'NerVE64Dataset', 'val.txt'),#******************
    'pc_file': 'pc_obj.pkl',
    'grid_size': 64,
    'cube_shift_mode': 'None'
}
dataset = data.get_dataset('RawPCDataset')(config)

model_cube = utils.load_model(op.join(model_path, 'cube'), device)
model_face = utils.load_model(op.join(model_path, 'face'), device)
model_geom = utils.load_model(op.join(model_path, 'geom'), device)


t0 = time()
with torch.no_grad():
    for idx in range(len(dataset)):
        model_input, info = dataset.get_data(idx, normalize='cube_face')
        model_input = {key: val.cuda() for key,val in model_input.items()}
        model_input['info'] = info

        name = info['name']
        print(f'Processing {idx}: {name}')
        
        res = {}
        edge_cube, peid = model_cube.predict_curve(model_input)
        res.update(edge_cube)
        edge_face = model_face.predict_curve(model_input, peid)
        res.update(edge_face)


        model_input, info = dataset.get_data(idx, normalize='geom')
        model_input = {key: val.cuda() for key,val in model_input.items()}        
        model_input['info'] = info
        edge_geom = model_geom.predict_curve(model_input, peid)
        res.update(edge_geom)

        curve_outputpath = os.path.join(output_path, name)
        os.makedirs(curve_outputpath, exist_ok=True)
        curve_outfile = os.path.join(curve_outputpath, 'pred_dk120_pwl.pkl')##############
        nerve_outfile = os.path.join(curve_outputpath, 'pred_dk120.pkl')#输出文件名########
        utils.nerve2pwl(res, curve_outfile)
        with open(nerve_outfile, 'wb') as f:
            pickle.dump(res, f)

print('Done, time cost: ', time()-t0)
