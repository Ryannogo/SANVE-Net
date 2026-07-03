import os

def remove_folder_suffix(target_dir, suffixes=["_filtered"]):
    """
    批量去除指定目录下文件夹的后缀（兼容_filtered）
    :param target_dir: 要处理的根目录（里面存放00001234_filtered这类文件夹）
    :param suffixes: 要去除的后缀列表，默认处理_filtered和_flitered
    """
    # 校验目标目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目标目录不存在 → {target_dir}")
        return

    processed = 0  # 成功重命名数量
    skipped = 0    # 跳过数量
    # 遍历目标目录下的所有项
    for folder_name in os.listdir(target_dir):
        old_folder_path = os.path.join(target_dir, folder_name)
        # 只处理文件夹，跳过文件
        if not os.path.isdir(old_folder_path):
            skipped += 1
            continue

        # 初始化新文件夹名称为原名称
        new_folder_name = folder_name
        # 遍历后缀，找到匹配的则去除（优先匹配_filtered，再匹配_flitered）
        for suffix in suffixes:
            if folder_name.endswith(suffix):
                new_folder_name = folder_name[:-len(suffix)]
                break  # 找到匹配后缀，无需继续判断

        # 如果名称未变化（无目标后缀），跳过
        if new_folder_name == folder_name:
            skipped += 1
            continue

        # 构建新文件夹路径
        new_folder_path = os.path.join(target_dir, new_folder_name)
        # 容错：如果新名称文件夹已存在，跳过并提示
        if os.path.exists(new_folder_path):
            print(f"跳过：新文件夹已存在 → 原：{folder_name} | 新：{new_folder_name}")
            skipped += 1
            continue

        # 执行重命名
        os.rename(old_folder_path, new_folder_path)
        print(f"重命名成功 → 原：{folder_name} → 新：{new_folder_name}")
        processed += 1

    # 打印最终统计结果
    print("-" * 50)
    print(f"批量重命名完成！")
    print(f"目标目录：{target_dir}")
    print(f"成功重命名：{processed} 个文件夹")
    print(f"跳过：{skipped} 个（非文件夹/无目标后缀/新名称已存在）")

if __name__ == "__main__":
    # -------------------------- 仅需修改这里的路径 --------------------------
    # 填写存放00001234_filtered文件夹的**根目录**（绝对路径/相对路径都可以）
    TARGET_DIRECTORY = r"traditional method/EdgeFormer"
    # -----------------------------------------------------------------------

    # 执行重命名
    remove_folder_suffix(TARGET_DIRECTORY)