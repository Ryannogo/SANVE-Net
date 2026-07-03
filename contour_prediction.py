import os, pickle
import torch
import data, utils
import numpy as np
from plyfile import PlyData, PlyElement

import os.path as op
from time import time

def predict_contour(data_path,output_path):
#line9,10,18
    ROOT_path = os.path.dirname(os.path.abspath(__file__))
    model_path = op.join(ROOT_path, 'results', 'train_relu')#模型路径

    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda')
    config = {
        'data_path': data_path,
        'file_list': op.join( data_path, 'all.txt'),#******************
        'pc_file': 'pc_obj.ply',
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
            curve_outfile = os.path.join(curve_outputpath, 'pred_M_pwl.pkl')
            nerve_outfile = os.path.join(curve_outputpath, 'pred_M.pkl')#输出文件名
            utils.nerve2pwl(res, curve_outfile)
            with open(nerve_outfile, 'wb') as f:
                pickle.dump(res, f)

    print('Done, time cost: ', time()-t0)

# ---------- 去噪方法集合 ----------

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    统计滤波 (适合孤立点去除)
    """
    import open3d as o3d
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def remove_radius_outliers(pcd, nb_points=16, radius=0.05):
    """
    半径滤波 (适合稀疏点去除)
    """
    import open3d as o3d
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points,
                                        radius=radius)
    return pcd.select_by_index(ind)


def wavelet_denoise(points, wavelet='db1', level=2):
    """
    小波去噪 (适合表面噪声平滑)
    points: numpy array (N, 3)
    """
    import pywt
    denoised = []
    for i in range(3):  # 分别处理 x, y, z
        coeffs = pywt.wavedec(points[:, i], wavelet, level=level)
        # 对高频系数做阈值处理
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(points)))
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        denoised.append(pywt.waverec(coeffs, wavelet))
    return np.stack(denoised, axis=1)




if __name__ == "__main__":
    data_path = "NerVE-main/PCA_length_5x"
    output_path = "NerVE-main/bridge"
    predict_contour(data_path,output_path)

