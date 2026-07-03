import os
import pickle
import numpy as np

# ========== 需要修改的路径 ==========
root_dir = "NerVE-main/NerVE64Dataset"          # 存放 all.txt 和子文件夹的根目录
output_dir = "score-denoise-main/data/ABC"  # 你希望保存 .xyz 的目标文件夹
# ===================================

os.makedirs(output_dir, exist_ok=True)

# 读取 all.txt 获取所有子文件夹名
all_txt_path = os.path.join(root_dir, "all.txt")
with open(all_txt_path, "r") as f:
    folder_names = [line.strip() for line in f if line.strip()]

for folder in folder_names:
    folder_path = os.path.join(root_dir, folder)
    pkl_path = os.path.join(folder_path, "step_curve_no_offset.pkl")
    output_path = os.path.join(output_dir, f"{folder}.xyz")

    if not os.path.exists(pkl_path):
        print(f"⚠️ 未找到 {pkl_path}，跳过。")
        continue

    try:
        # 加载 pkl 文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 提取 points
        points = np.array(data.get("points", []), dtype=float)

        if points.size == 0:
            print(f"⚠️ {folder} 中未找到有效的 'points' 数据，跳过。")
            continue

        # 保存为 .xyz 格式
        np.savetxt(output_path, points, fmt="%.10f")
        print(f"✅ 已保存 {folder}.xyz ({len(points)} 个点)")

    except Exception as e:
        print(f"❌ 处理 {folder} 时出错: {e}")
