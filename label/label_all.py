import json
import datetime
from collections import defaultdict
import os


def merge_ad_datasets(dataset_paths, output_path="/home/jiangyuxin/CODE/Datasets/merged_ad_datasets_cover1_copped.json"):
# def merge_ad_datasets(dataset_paths,
#                           output_path="/home/ud202480265/AnomalyAny-main/anomaly_image/merged_ad_datasets.json"):
    """
    合并多个异常检测数据集的 JSON 文件到一个 JSON 文件，确保字段完整性

    :param dataset_paths: 数据集 JSON 文件路径字典 {
        'AITEX': 'path/to/aitex.json',
        'BTech': 'path/to/btech.json',
        ...
    }
    :param output_path: 输出 JSON 文件路径
    :return: 合并后的数据字典
    """
    merged_data = {
        "datasets": [],
        "statistics": defaultdict(int),
        "version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "data_sources": list(dataset_paths.keys())  # 记录所有数据源
    }

    for dataset_name, dataset_path in dataset_paths.items():
        print(f"Processing {dataset_name}...")
        try:
            # 从 JSON 文件中读取数据集数据
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)

            # 标准化图像条目
            standardized_images = []
            for img_entry in dataset_data.get("images", []):
                if not isinstance(img_entry, dict):
                    print(f"Warning: Non-dictionary entry found in 'images' list of {dataset_name}: {img_entry}")
                    continue

                # 处理不同数据集可能使用的字段别名
                defect_name = img_entry.get("defect_name",
                                            img_entry.get("defect_type",
                                                          img_entry.get("anomaly_type", "")))

                category = img_entry.get("category",
                                         img_entry.get("class",
                                                       img_entry.get("object_type", "")))

                # 在路径前添加数据集名字
                # image_path = os.path.join(dataset_name, img_entry.get("image_path", ""))
                # mask_path = os.path.join(dataset_name, img_entry.get("mask_path", ""))
                image_path = os.path.join(dataset_name, img_entry.get("cropped_image_path", ""))
                mask_path = os.path.join(dataset_name, img_entry.get("cropped_mask_path", ""))

                # 处理 analysis_files 路径，避免重复的 prompt
                analysis_files = img_entry.get("analysis_files", "")
                if analysis_files.startswith("prompt/prompt"):
                    analysis_files = analysis_files[7:]  # 移除重复的 prompt
                analysis_files = os.path.join(dataset_name, analysis_files)

                # 处理 analysis_files_short 路径，避免重复的 prompt_short
                analysis_files_short = img_entry.get("analysis_files_short", "")
                if analysis_files_short.startswith("prompt_short/prompt_short"):
                    analysis_files_short = analysis_files_short[len("prompt_short/"):]
                analysis_files_short = os.path.join(dataset_name, analysis_files_short)

                # 构建标准化条目
                std_entry = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "defect_code": img_entry.get("defect_code", ""),
                    "defect_name": defect_name,
                    "category": category,
                    "split": img_entry.get("split",
                                           img_entry.get("phase", "test")),  # 默认 test
                    "analysis_files": analysis_files,
                    "analysis_files_short": analysis_files_short,
                    "dataset": dataset_name  # 添加数据集标识
                }
                standardized_images.append(std_entry)

            # 处理缺陷类型
            standardized_defect_types = []
            for defect in dataset_data.get("defect_types", []):
                if isinstance(defect, dict):
                    # 处理字典形式的缺陷类型
                    standardized_defect_types.append({
                        "code": defect.get("code", ""),
                        "name": defect.get("name", "")
                    })
                elif isinstance(defect, str):
                    # 处理字符串形式的缺陷类型
                    standardized_defect_types.append({
                        "code": defect,
                        "name": defect
                    })
                else:
                    # 其他类型的处理
                    standardized_defect_types.append({
                        "code": str(defect),
                        "name": str(defect)
                    })

            # 构建标准化数据集结构
            standardized_dataset = {
                "name": dataset_name,
                "categories": dataset_data.get("categories", []),
                "defect_types": standardized_defect_types,
                "images": standardized_images
            }

            # 添加到合并数据
            merged_data["datasets"].append(standardized_dataset)

            # 更新统计信息
            merged_data["statistics"]["total_datasets"] += 1
            merged_data["statistics"]["total_categories"] += len(set(standardized_dataset["categories"]))  # 使用 set 去重
            merged_data["statistics"]["total_defect_types"] += len(standardized_defect_types)
            merged_data["statistics"]["total_images"] += len(standardized_images)

            # 按分割统计
            for img in standardized_images:
                split = img["split"].lower()  # 统一小写处理
                if split in ["train", "test", "val", "validation"]:
                    key = f"{split}_images"
                    merged_data["statistics"][key] = merged_data["statistics"].get(key, 0) + 1

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    # 计算全局唯一类别和缺陷类型
    all_categories = set()
    all_defect_types = set()

    for dataset in merged_data["datasets"]:
        all_categories.update(dataset["categories"])
        for defect in dataset["defect_types"]:
            # 这里确保defect已经是字典形式
            all_defect_types.add(f"{defect.get('code', '')}:{defect.get('name', '')}")

    merged_data["statistics"]["unique_categories"] = len(all_categories)
    merged_data["statistics"]["unique_defect_types"] = len(all_defect_types)

    # 保存合并后的数据
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully merged {len(merged_data['datasets'])} datasets. Saved to {output_path}")

    # 打印汇总统计
    print("\n合并统计摘要:")
    for stat, value in merged_data["statistics"].items():
        print(f"{stat.replace('_', ' ').title()}: {value}")

    return merged_data


if __name__ == "__main__":
    # 配置所有数据集的 JSON 文件路径
    dataset_paths = {
        # 'AITEX': '/home/jiangyuxin/CODE/Datasets/AITEX/dataset_info.json',
        # 'BTech': '/home/jiangyuxin/CODE/Datasets/BTech/dataset_info.json',
        # 'eyecandies_preprocessed': '/home/jiangyuxin/CODE/Datasets/eyecandies_preprocessed/dataset_info.json',
        # 'MPDD': '/home/jiangyuxin/CODE/Datasets/MPDD/dataset_info.json',
        # 'mvtec': '/home/jiangyuxin/CODE/Datasets/mvtec/dataset_info_1.json',
        # 'MTD': '/home/jiangyuxin/CODE/Datasets/MTD/dataset_info.json',
        # 'mvtec3d': '/home/jiangyuxin/CODE/Datasets/mvtec3d/dataset_info.json',
        # 'VisA_pytorch/1cls': '/home/jiangyuxin/CODE/Datasets/VisA_pytorch/1cls/dataset_info_1.json',
        # # # 'realAD': 'realAD/dataset_info.json',
        # 'KolektorSDD2': '/home/jiangyuxin/CODE/Datasets/KolektorSDD2/dataset_info_1.json',
        # 'MulSen_AD': '/home/jiangyuxin/CODE/Datasets/MulSen_AD/dataset_info.json',
        # 'mvtec_ad_2': '/home/jiangyuxin/CODE/Datasets/mvtec_ad_2/dataset_info.json',
        # 'DAGM_anomaly_detection': '/home/jiangyuxin/CODE/Datasets/DAGM_anomaly_detection/dataset_info.json',
        # 'MANTA_TINY_256': '/home/jiangyuxin/CODE/Datasets/MANTA_TINY_256/dataset_info.json',
        # 'VisA_reference': '/home/jiangyuxin/CODE/Datasets/VisA_reference/dataset_info_1.json',
        'CYK': '/home/jiangyuxin/CODE/Datasets/CYK/dataset_info_cropped.json',
    }

    # 检查每个路径是否存在
    for dataset_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Error: The JSON file for {dataset_name} at {path} does not exist.")
            exit(1)

    # 执行合并
    merged_data = merge_ad_datasets(dataset_paths)

    # 打印第一个数据集的第一个图像条目作为示例
    if merged_data["datasets"]:
        sample_dataset = merged_data["datasets"][0]
        print(f"\n示例数据集 '{sample_dataset['name']}':")
        print(f"类别: {sample_dataset['categories']}")
        print(f"缺陷类型: {[d['name'] for d in sample_dataset['defect_types'][:2]]}...")

        sample_image = sample_dataset["images"][0]
        print("\n示例图像条目:")
        print(json.dumps(sample_image, indent=2))
