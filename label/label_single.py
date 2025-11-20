import os
import cv2
import numpy as np
import shutil
import json
from tqdm import tqdm
from collections import defaultdict


# Supported image extensions (case insensitive)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMAGE_EXTENSIONS_CASE_INSENSITIVE = tuple(ext.lower() for ext in IMAGE_EXTENSIONS) + tuple(
    ext.upper() for ext in IMAGE_EXTENSIONS)

# AITEX dataset defect codes and names
aitex_defect_codes = {
    2: "Broken end",
    6: "Broken yarn",
    10: "Broken pick",
    16: "Weft curling",
    19: "Fuzzyball",
    22: "Cut selvage",
    23: "Crease",
    25: "Warp ball",
    27: "Knots",
    29: "Contamination",
    30: "Nep",
    36: "Weft crack"
}

# realAD dataset defect codes and names
realad_defect_codes = {
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing parts",
    "YW": "foreign objects",
    "ZW": "contamination"
}


def draw_defect_boxes(image, mask):
    """Draw bounding boxes around defects in the image based on the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image


def get_possible_mask_names(image_filename):
    """Generate all possible mask filename variations for a given image filename."""
    base_name = os.path.splitext(image_filename)[0]
    variations = []

    # Add all possible combinations of extensions and suffixes
    for ext in ['.png', '.bmp']:
        variations.extend([
            base_name + ext,  # image.jpg -> image.png
            base_name + '_mask' + ext,  # image.jpg -> image_mask.png
            base_name + '_gt' + ext,  # image.jpg -> image_gt.png
            base_name + '_label' + ext,  # image.jpg -> image_label.png
            'mask_' + base_name + ext  # image.jpg -> mask_image.png
        ])

    return variations


def create_dataset_info_json(dataset_root):
    """Create JSON files with dataset information for all supported datasets."""
    datasets = {
        # 'AITEX': process_aitex_metadata,
        # 'BTech': process_btech_metadata,
        # 'eyecandies_preprocessed': process_eyecandies_metadata,
        # 'MPDD': process_mpdd_mvtec_metadata,
        # 'mvtec': process_mpdd_mvtec_metadata,
        # 'MTD': process_mtd_metadata,
        # 'mvtec3d': process_mvte3d_metadata,
        # 'VisA_pytorch/1cls': process_mpdd_mvtec_metadata,
        # 'VisA_20220922': process_visa_metadata,
        # 'VisA_reference': process_visa_metadata,
        # 'realAD': process_realad_metadata,
        # 'KolektorSDD2': process_kolektor_metadata,
        # 'MulSen_AD': process_mulsen_ad_metadata,
        # 'mvtec_ad_2': process_mvtec_ad_2_metadata,
        # 'DAGM_anomaly_detection': process_dagm_anomaly_detection_metadata,
        # 'MANTA_TINY_256': process_manta_metadata,
        'CYK': process_over1_metadata,
    }

    for dataset_name, processor in datasets.items():
        dataset_path = os.path.join(dataset_root, dataset_name)
        if os.path.exists(dataset_path):
            print(f"Processing metadata for {dataset_name}...")
            metadata = processor(dataset_path)
            with open(os.path.join(dataset_path, 'dataset_info_1.json'), 'w') as f:
                json.dump(metadata, f, indent=4)


def process_aitex_metadata(dataset_path):
    """Create metadata for AITEX dataset."""
    metadata = {
        "name": "AITEX",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    # Add defect types
    for code, name in aitex_defect_codes.items():
        metadata["defect_types"].append({
            "code": code,
            "name": name
        })

    # Process defect images
    defect_images_dir = os.path.join(dataset_path, 'Defect_images')
    mask_images_dir = os.path.join(dataset_path, 'Mask_images')

    if os.path.exists(defect_images_dir) and os.path.exists(mask_images_dir):
        for filename in os.listdir(defect_images_dir):
            if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                parts = filename.split('_')
                if len(parts) >= 3:
                    defect_code = int(parts[1])
                    # Try all possible mask filename variations
                    for mask_filename in get_possible_mask_names(filename):
                        if os.path.exists(os.path.join(mask_images_dir, mask_filename)):
                            metadata["images"].append({
                                "image_path": os.path.join('Defect_images', filename),
                                "mask_path": os.path.join('Mask_images', mask_filename),
                                "defect_code": defect_code,
                                "defect_name": aitex_defect_codes.get(defect_code, "Unknown"),
                                "category": "fabric",
                                "split": "test"
                            })
                            # 统计物体和缺陷
                            object_name = "fabric"
                            defect_name = aitex_defect_codes.get(defect_code, "Unknown")
                            if object_name not in metadata["objects"]:
                                metadata["objects"][object_name] = set()
                            metadata["objects"][object_name].add(defect_name)
                            break
                    else:
                        print(f"Warning: Could not find mask for {filename} in {mask_images_dir}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


def process_btech_metadata(dataset_path):
    """Create metadata for BTech dataset."""
    metadata = {
        "name": "BTech",
        "categories": [],
        "defect_types": ["ko"],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path) and category.isdigit():
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)
            # Process test images with defects (ko)
            test_ko_dir = os.path.join(category_path, 'test', 'ko')
            gt_dir = os.path.join(category_path, 'ground_truth', 'ko')

            if os.path.exists(test_ko_dir) and os.path.exists(gt_dir):
                for filename in os.listdir(test_ko_dir):
                    if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                        # Try all possible mask filename patterns
                        for mask_filename in get_possible_mask_names(filename):
                            mask_path = os.path.join(gt_dir, mask_filename)
                            if os.path.exists(mask_path):
                                if category == "01":
                                    category_0 = "bottle"
                                elif category == "02":
                                    category_0 = "fabric"
                                else:
                                    category_0 = "plate"

                                metadata["images"].append({
                                    "image_path": os.path.join(category, 'test', 'ko', filename),
                                    "mask_path": os.path.join(category, 'ground_truth', 'ko', mask_filename),
                                    "defect_code": "",
                                    "defect_name": "defect",
                                    "category": category_0,
                                    "split": "test"
                                })
                                # 统计物体和缺陷
                                object_name = category_0
                                defect_name = "defect"
                                if object_name not in metadata["objects"]:
                                    metadata["objects"][object_name] = set()
                                metadata["objects"][object_name].add(defect_name)
                                break
                        else:
                            print(f"Warning: Could not find mask for {filename} in {gt_dir}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


def process_eyecandies_metadata(dataset_path):
    """Create metadata for eyecandies dataset."""
    metadata = {
        "name": "eyecandies",
        "categories": [],
        "defect_types": ["bad"],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)

            # Process test images with defects
            test_dir = os.path.join(category_path, 'test', 'bad')
            gt_dir = os.path.join(category_path, 'test', 'bad', 'gt')

            if os.path.exists(test_dir) and os.path.exists(gt_dir):
                rgb_dir = os.path.join(test_dir, 'rgb')
                if os.path.exists(rgb_dir):
                    for filename in os.listdir(rgb_dir):
                        if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                            # Try all possible mask filename patterns
                            for mask_filename in get_possible_mask_names(filename):
                                mask_path = os.path.join(gt_dir, mask_filename)
                                if os.path.exists(mask_path):
                                    metadata["images"].append({
                                        "image_path": os.path.join(category, 'test', 'bad', 'rgb', filename),
                                        "mask_path": os.path.join(category, 'test', 'bad', 'gt', mask_filename),
                                        "defect_code": "",
                                        "defect_name": "defect",
                                        "category": category,
                                        "split": "test"
                                    })
                                    # 统计物体和缺陷
                                    object_name = category
                                    defect_name = "defect"
                                    if object_name not in metadata["objects"]:
                                        metadata["objects"][object_name] = set()
                                    metadata["objects"][object_name].add(defect_name)
                                    break
                            else:
                                print(f"Warning: Could not find mask for {filename} in {gt_dir}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


def process_manta_metadata(dataset_path):
    """Create metadata for MPDD or MVTec dataset."""
    dataset_name = os.path.basename(dataset_path)
    metadata = {
        "name": dataset_name,
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},  # 统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        metadata["categories"].append(category)

        # 遍历子类别（如 test/train 或具体的子类别名称）
        for small_category in os.listdir(category_path):
            small_path = os.path.join(category_path, small_category)
            if not os.path.isdir(small_path):
                continue

            # 处理测试图像和掩码
            test_dir = os.path.join(small_path, "test")
            gt_base_dir = os.path.join(small_path, "ground_truth")

            if not os.path.exists(gt_base_dir):
                continue  # 如果没有 ground_truth 目录，跳过该子类别

            # 遍历缺陷类型
            for defect_type in os.listdir(test_dir):
                defect_test_path = os.path.join(test_dir, defect_type)
                defect_gt_path = os.path.join(gt_base_dir, defect_type)

                if not os.path.isdir(defect_test_path) or not os.path.isdir(defect_gt_path):
                    continue

                metadata["defect_types"].append({
                    "code": defect_type,
                    "name": defect_type
                })

                # 遍历缺陷类型下的所有图像
                for filename in os.listdir(defect_test_path):
                    if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                        # 尝试所有可能的掩码文件名变体
                        for mask_filename in get_possible_mask_names(filename):
                            mask_path = os.path.join(defect_gt_path, mask_filename)
                            if os.path.exists(mask_path):
                                # 构建相对路径
                                image_rel_path = os.path.join(
                                    category, small_category, "test", defect_type, filename
                                )
                                mask_rel_path = os.path.join(
                                    category, small_category, "ground_truth", defect_type, mask_filename
                                )

                                metadata["images"].append({
                                    "image_path": image_rel_path,
                                    "mask_path": mask_rel_path,
                                    "defect_code": defect_type,
                                    "defect_name": defect_type,
                                    "category": category,
                                    "split": small_category  # test 或 train
                                })

                                # 统计物体和缺陷
                                if category not in metadata["objects"]:
                                    metadata["objects"][category] = set()
                                metadata["objects"][category].add(defect_type)
                                break
                        else:
                            print(f"Warning: Mask not found for {filename} in {defect_gt_path}")

    # 将集合转换为列表
    for obj in metadata["objects"]:
        metadata["objects"][obj] = list(metadata["objects"][obj])

    return metadata

def process_mpdd_mvtec_metadata(dataset_path):
    """Create metadata for MPDD or MVTec dataset."""
    dataset_name = os.path.basename(dataset_path)
    metadata = {
        "name": dataset_name,
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)

            # Process test images with defects
            test_dir = os.path.join(category_path, 'test')
            gt_dir = os.path.join(category_path, 'ground_truth')

            if os.path.exists(test_dir) and os.path.exists(gt_dir):
                for defect_type in os.listdir(test_dir):
                    defect_path = os.path.join(test_dir, defect_type)
                    if os.path.isdir(defect_path) and defect_type != 'good':
                        metadata["defect_types"].append({
                            "code": defect_type,
                            "name": defect_type
                        })

                        for filename in os.listdir(defect_path):
                            if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                # Try all possible mask filename variations
                                for mask_filename in get_possible_mask_names(filename):
                                    mask_path = os.path.join(gt_dir, defect_type, mask_filename)
                                    if os.path.exists(mask_path):
                                        metadata["images"].append({
                                            "image_path": os.path.join(category, 'test', defect_type, filename),
                                            "mask_path": os.path.join(category, 'ground_truth', defect_type,
                                                                      mask_filename),
                                            "defect_code": "",
                                            "defect_name": defect_type,
                                            "category": category,
                                            "split": "test"
                                        })
                                        # 统计物体和缺陷
                                        object_name = category
                                        defect_name = defect_type
                                        if object_name not in metadata["objects"]:
                                            metadata["objects"][object_name] = set()
                                        metadata["objects"][object_name].add(defect_name)
                                        break
                                else:
                                    print(
                                        f"Warning: Could not find mask for {filename} in {os.path.join(gt_dir, defect_type)}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata

def process_over1_metadata(dataset_path):
    """Create metadata for MPDD or MVTec dataset."""
    dataset_name = os.path.basename(dataset_path)
    metadata = {
        "name": dataset_name,
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)

            # Process test images with defects
            test_dir = os.path.join(category_path, 'test')
            gt_dir = os.path.join(category_path, 'ground_truth')

            if os.path.exists(test_dir) and os.path.exists(gt_dir):
                for defect_type in os.listdir(test_dir):
                    defect_path = os.path.join(test_dir, defect_type)
                    if os.path.isdir(defect_path) and defect_type != 'good':
                        metadata["defect_types"].append({
                            "code": defect_type,
                            "name": defect_type
                        })

                        for filename in os.listdir(defect_path):
                            if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                # Try all possible mask filename variations
                                for mask_filename in get_possible_mask_names(filename):
                                    mask_path = os.path.join(gt_dir, defect_type, mask_filename)
                                    if os.path.exists(mask_path):
                                        metadata["images"].append({
                                            "image_path": os.path.join(category, 'test', defect_type, filename),
                                            "mask_path": os.path.join(category, 'ground_truth', defect_type,
                                                                      mask_filename),
                                            "defect_code": "",
                                            "defect_name": defect_type,
                                            "category": category,
                                            "split": "test"
                                        })
                                        # 统计物体和缺陷
                                        object_name = category
                                        defect_name = defect_type
                                        if object_name not in metadata["objects"]:
                                            metadata["objects"][object_name] = set()
                                        metadata["objects"][object_name].add(defect_name)
                                        break
                                else:
                                    print(
                                        f"Warning: Could not find mask for {filename} in {os.path.join(gt_dir, defect_type)}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


def process_mtd_metadata(dataset_path):
    """Create metadata for MTD dataset."""
    metadata = {
        "name": "MTD",
        "categories": ["fabric"],
        "defect_types": [],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }
    test_dir = os.path.join(dataset_path, 'test')
    if os.path.exists(test_dir):
        for defect_type in os.listdir(test_dir):
            defect_path = os.path.join(test_dir, defect_type)
            if os.path.isdir(defect_path):
                code = defect_type
                defect_name = defect_type.split('_')[-1]
                metadata["defect_types"].append({
                    "code": code,
                    "name": defect_name
                })
                img_dir = os.path.join(defect_path, 'Imgs')
                if os.path.exists(img_dir):
                    for filename in os.listdir(img_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            base_name, ext = os.path.splitext(filename)
                            mask_filename = base_name + '.png'
                            mask_path = os.path.join(img_dir, mask_filename)
                            if os.path.exists(mask_path):
                                metadata["images"].append({
                                    "image_path": os.path.join('test', defect_type, 'Imgs', filename),
                                    "mask_path": os.path.join('test', defect_type, 'Imgs', mask_filename),
                                    "defect_code": code,
                                    "defect_name": defect_name,
                                    "category": "fabric",
                                    "split": "test"
                                })
                                # 统计物体和缺陷
                                object_name = "fabric"
                                defect_name = defect_name
                                if object_name not in metadata["objects"]:
                                    metadata["objects"][object_name] = set()
                                metadata["objects"][object_name].add(defect_name)
                            else:
                                print(f"Warning: Could not find mask for {filename} in {defect_path}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


def process_mvte3d_metadata(dataset_path):
    """Create metadata for MVTec3D dataset."""
    metadata = {
        "name": "mvtec3d",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)

            # Process test images with defects
            test_dir = os.path.join(category_path, 'test')

            if os.path.exists(test_dir):
                for defect_type in os.listdir(test_dir):
                    defect_path = os.path.join(test_dir, defect_type)
                    if os.path.isdir(defect_path) and defect_type != 'good':
                        metadata["defect_types"].append({
                            "code": defect_type,
                            "name": defect_type
                        })

                        rgb_dir = os.path.join(defect_path, 'rgb')
                        gt_dir = os.path.join(defect_path, 'gt')

                        if os.path.exists(rgb_dir) and os.path.exists(gt_dir):
                            for filename in os.listdir(rgb_dir):
                                if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                    # Try all possible mask filename variations
                                    for mask_filename in get_possible_mask_names(filename):
                                        mask_path = os.path.join(gt_dir, mask_filename)
                                        if os.path.exists(mask_path):
                                            metadata["images"].append({
                                                "image_path": os.path.join(category, 'test', defect_type, 'rgb',
                                                                           filename),
                                                "mask_path": os.path.join(category, 'test', defect_type, 'gt',
                                                                          mask_filename),
                                                "defect_code": "",
                                                "defect_name": defect_type,
                                                "category": category,
                                                "split": "test"
                                            })
                                            # 统计物体和缺陷
                                            object_name = category
                                            defect_name = defect_type
                                            if object_name not in metadata["objects"]:
                                                metadata["objects"][object_name] = set()
                                            metadata["objects"][object_name].add(defect_name)
                                            break
                                    else:
                                        print(
                                            f"Warning: Could not find mask for {filename} in {os.path.join(gt_dir)}")

    # 将集合转换为列表
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata


# def process_visa_metadata(dataset_path):
#     """Create metadata for VisA_pytorch dataset."""
#     metadata = {
#         "name": "VisA_pytorch",
#         "categories": [],
#         "defect_types": [],
#         "images": [],
#         "objects": {},
#     }
#     for category in os.listdir(dataset_path):
#         category_path = os.path.join(dataset_path, category)
#         if os.path.isdir(category_path):
#             if category not in ["annotated", "prompt_short", "prompt"]:
#                 metadata["categories"].append(category)
#             test_dir = os.path.join(category_path, 'test')
#             mask_dir = os.path.join(category_path, 'ground_truth')
#             if os.path.exists(test_dir):
#                 for defect_type in os.listdir(test_dir):
#                     defect_path = os.path.join(test_dir, defect_type)
#                     mask_path = os.path.join(mask_dir, defect_type)
#                     if os.path.isdir(defect_path) and defect_type != 'good':
#                         metadata["defect_types"].append({
#                             "code": defect_type,
#                             "name": defect_type
#                         })
#                         for filename in os.listdir(defect_path):
#                             if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
#                                 base_name, ext = os.path.splitext(filename)
#                                 mask_filename = base_name + '.png'
#                                 mask_path = os.path.join(mask_path, mask_filename)
#                                 if os.path.exists(mask_path):
#                                     metadata["images"].append({
#                                         "image_path": os.path.join(category, 'test', defect_type, filename),
#                                         "mask_path": os.path.join(category, 'ground_truth', defect_type, mask_filename),
#                                         "defect_code": defect_type,
#                                         "defect_name": defect_type,
#                                         "category": category,
#                                         "split": "test"
#                                     })
#                                     object_name = category
#                                     defect_name = defect_type
#                                     if object_name not in metadata["objects"]:
#                                         metadata["objects"][object_name] = set()
#                                     metadata["objects"][object_name].add(defect_name)
#
#     for object_name in metadata["objects"]:
#         metadata["objects"][object_name] = list(metadata["objects"][object_name])
#     return metadata

def process_visa_metadata(dataset_path):
    """Create metadata for VisA_pytorch dataset."""
    metadata = {
        "name": "VisA_reference",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)
            test_dir = os.path.join(category_path,"Data", 'Images')
            mask_dir = os.path.join(category_path,"Data", 'Masks')
            if os.path.exists(test_dir):
                for defect_type in os.listdir(test_dir):
                    defect_path = os.path.join(test_dir, defect_type)
                    # Mask_path = os.path.join(mask_dir, defect_type)
                    Mask_path = os.path.join(mask_dir)
                    if os.path.isdir(defect_path) and defect_type != 'Normal':
                        metadata["defect_types"].append({
                            "code": defect_type,
                            "name": defect_type
                        })
                        print(defect_path)
                        for filename in os.listdir(defect_path):
                            if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                base_name, ext = os.path.splitext(filename)
                                mask_filename = base_name + '.png'
                                mask_path = os.path.join(Mask_path, mask_filename)
                                print(mask_path)
                                analysis_file, analysis_file_short = get_analysis_file_paths(dataset_path,
                                                                                             image_rel_path)
                                if os.path.exists(mask_path):
                                    metadata["images"].append({
                                        "image_path": os.path.join(category, "Data", 'Images', defect_type, filename),
                                        "mask_path": os.path.join(category, "Data", 'Masks', mask_filename),
                                        "defect_code": defect_type,
                                        "defect_name": defect_type,
                                        "category": category,
                                        "split": "test"
                                    })
                                    object_name = category
                                    defect_name = defect_type
                                    if object_name not in metadata["objects"]:
                                        metadata["objects"][object_name] = set()
                                    metadata["objects"][object_name].add(defect_name)

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata
def process_visa_metadata(dataset_path):
    """Create metadata for VisA_pytorch dataset."""
    metadata = {
        "name": "VisA_reference",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)
            test_dir = os.path.join(category_path,"Anomaly")
            mask_dir = os.path.join(category_path,"Anomaly_mask")
            if os.path.exists(test_dir):
                defect_path = os.path.join(test_dir)
                Mask_path = os.path.join(mask_dir)
                if os.path.isdir(defect_path):
                    metadata["defect_types"].append({
                        "code": "Anomaly",
                        "name": "Anomaly"
                    })
                    for filename in os.listdir(defect_path):
                        if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                            base_name, ext = os.path.splitext(filename)
                            mask_filename = base_name + '_mask.png'
                            mask_path = os.path.join(Mask_path, mask_filename)
                            image_rel_path = os.path.join(category, "Anomaly", filename)
                            analysis_file, analysis_file_short = get_analysis_file_paths(dataset_path,
                                                                                         image_rel_path)
                            if os.path.exists(mask_path):
                                metadata["images"].append({
                                    "image_path": os.path.join(category, "Anomaly", filename),
                                    "mask_path": os.path.join(category, "Anomaly_mask", mask_filename),
                                    "defect_code": "Anomaly",
                                    "defect_name": "Anomaly",
                                    "category": category,
                                    "split": "test",
                                    "analysis_files": analysis_file,
                                    "analysis_files_short": analysis_file_short
                                })
                                object_name = category
                                defect_name = "Anomaly"
                                if object_name not in metadata["objects"]:
                                    metadata["objects"][object_name] = set()
                                metadata["objects"][object_name].add(defect_name)

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata
# def process_visa_metadata(dataset_path):
#     """Create metadata for VisA_pytorch dataset."""
#     metadata = {
#         "name": "VisA_reference",
#         "categories": [],
#         "defect_types": [],
#         "images": [],
#         "objects": {},
#     }
#     for category in os.listdir(dataset_path):
#         category_path = os.path.join(dataset_path, category)
#         if os.path.isdir(category_path):
#             if category not in ["annotated", "prompt_short", "prompt"]:
#                 metadata["categories"].append(category)
#             test_dir = os.path.join(category_path,"Anomaly")
#             mask_dir = os.path.join(category_path,"Anomaly_mask")
#             if os.path.exists(test_dir):
#                 defect_path = os.path.join(test_dir)
#                 Mask_path = os.path.join(mask_dir)
#                 if os.path.isdir(defect_path):
#                     metadata["defect_types"].append({
#                         "code": "Anomaly",
#                         "name": "Anomaly"
#                     })
#                     for filename in os.listdir(defect_path):
#                         if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
#                             base_name, ext = os.path.splitext(filename)
#                             mask_filename = base_name + '_mask.png'
#                             mask_path = os.path.join(Mask_path, mask_filename)
#                             if os.path.exists(mask_path):
#                                 metadata["images"].append({
#                                     "image_path": os.path.join(category, "Anomaly", filename),
#                                     "mask_path": os.path.join(category, "Anomaly_mask", mask_filename),
#                                     "defect_code": "Anomaly",
#                                     "defect_name": "Anomaly",
#                                     "category": category,
#                                     "split": "test"
#                                 })
#                                 object_name = category
#                                 defect_name = "Anomaly"
#                                 if object_name not in metadata["objects"]:
#                                     metadata["objects"][object_name] = set()
#                                 metadata["objects"][object_name].add(defect_name)
#
#     for object_name in metadata["objects"]:
#         metadata["objects"][object_name] = list(metadata["objects"][object_name])
#     return metadata
def process_realad_metadata(dataset_path, output_dir="processed_dataset"):
    """处理RealAD数据集，生成带标注的图像"""
    metadata = {
        "name": "realAD",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }

    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    # 获取所有类别
    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))
                  and d not in ["annotated", "prompt_short", "prompt"]]

    metadata["categories"] = categories

    # 处理每个类别
    for category in tqdm(categories, desc="处理类别"):
        print(category)
        category_path = os.path.join(dataset_path, category)

        # 处理NG和OK两种质量类型
        for quality_type in ["NG"]:
            quality_path = os.path.join(category_path, quality_type)

            # 如果目录不存在则跳过
            if not os.path.exists(quality_path):
                continue

            # 创建输出目录
            annotated_dir = os.path.join(quality_path, 'annotated')
            if os.path.exists(annotated_dir):
                shutil.rmtree(annotated_dir)
            os.makedirs(annotated_dir, exist_ok=True)

            # 获取所有缺陷类型目录
            defect_types = [d for d in os.listdir(quality_path)
                            if os.path.isdir(os.path.join(quality_path, d))]

            # 处理每个缺陷类型
            for defect_type in defect_types:
                print(defect_type)
                defect_path = os.path.join(quality_path, defect_type)

                # 获取缺陷名称（使用提供的映射表）
                if defect_type != "annotated":
                    defect_name = realad_defect_codes.get(defect_type, "Unknown")

                    # 记录缺陷类型元数据
                    if {"code": defect_type, "name": defect_name} not in metadata["defect_types"]:
                        metadata["defect_types"].append({
                            "code": defect_type,
                            "name": defect_name
                        })

                # 获取所有子文件夹
                folders = [f for f in os.listdir(defect_path)
                           if os.path.isdir(os.path.join(defect_path, f))]

                # 处理每个子文件夹
                for folder in folders:
                    folder_path = os.path.join(defect_path, folder)

                    # 获取所有图像文件
                    image_files = [f for f in os.listdir(folder_path)
                                   if f.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE)]

                    # 处理每个图像
                    for filename in image_files:
                        image_path = os.path.join(folder_path, filename)

                        try:
                            # 读取图像
                            image = cv2.imread(image_path)
                            if image is None:
                                print(f"警告: 无法读取图像 {image_path}")
                                continue

                            # 处理NG图像（有缺陷）
                            if quality_type == 'NG':
                                # 尝试所有可能的掩码文件名变体
                                mask_found = False
                                for mask_filename in get_possible_mask_names(filename):
                                    mask_path = os.path.join(folder_path, mask_filename)
                                    if os.path.exists(mask_path):
                                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                        if mask is None:
                                            print(f"警告: 无法读取掩码 {mask_path}")
                                            result_image = image
                                        else:
                                            # 绘制缺陷边界框
                                            result_image = draw_defect_boxes(image, mask)

                                        # 记录图像元数据
                                        metadata["images"].append({
                                            "image_path": os.path.relpath(image_path, dataset_path),
                                            "mask_path": os.path.relpath(mask_path, dataset_path),
                                            "defect_code": defect_type,
                                            "defect_name": defect_name,
                                            "category": category,
                                            "quality_type": quality_type,
                                            "folder": folder
                                        })
                                        mask_found = True
                                        break

                                if not mask_found:
                                    # 没有找到掩码文件，使用原图
                                    result_image = image
                                    print(f"警告: 未找到掩码文件 for {filename} in {folder_path}")
                            else:
                                # 处理OK图像（无缺陷）
                                result_image = image

                            # 保存处理后的图像
                            output_path = os.path.join(annotated_dir, filename)
                            cv2.imwrite(output_path, result_image)

                        except Exception as e:
                            print(f"处理图像 {image_path} 时出错: {str(e)}")

    # 整理objects元数据
    for img in metadata["images"]:
        object_name = img["category"]
        defect_name = img["defect_name"]
        if object_name not in metadata["objects"]:
            metadata["objects"][object_name] = set()
        metadata["objects"][object_name].add(defect_name)

    # 将set转换为list以便于JSON序列化
    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])

    return metadata

def process_kolektor_metadata(dataset_path):
    """Create metadata for KolektorSDD2 dataset."""
    metadata = {
        "name": "KolektorSDD2",
        "categories": ["image"],
        "defect_types": [{
            "code": "defect",
            "name": "defect"
        }],
        "images": [],
        "objects": {},  # 新增：统计物体和缺陷
    }

    # 跳过train类别
    for split in ['test']:
        split_dir = os.path.join(dataset_path, split)
        if os.path.exists(split_dir):
            img_dir = os.path.join(split_dir)
            mask_dir = os.path.join(split_dir)
            for filename in os.listdir(img_dir):
                if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                    mask_filename = filename.split('.')[0] + '_GT.png'
                    mask_path = os.path.join(mask_dir, mask_filename)
                    if os.path.exists(mask_path):
                        # Read mask (must be grayscale)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Warning: Could not read mask {mask_path}")
                            continue
                        # Check if mask is all black
                        if np.all(mask == 0):
                            metadata["images"].append({
                                "image_path": os.path.join(split, filename),
                                "mask_path": os.path.join(split, mask_filename),
                                "defect_code": "defect",
                                "defect_name": "defect",
                                "category": "image",
                                "split": split
                            })
                            object_name = "image"
                            defect_name = "defect"
                            if object_name not in metadata["objects"]:
                                metadata["objects"][object_name] = set()
                            metadata["objects"][object_name].add(defect_name)

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata



def process_mulsen_ad_metadata(dataset_path):
    """Create metadata for MulSen_AD dataset."""
    metadata = {
        "name": "MulSen_AD",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(category)
            rgb_dir = os.path.join(category_path, 'RGB')
            # 跳过train类别
            for split in ['test']:
                split_rgb_dir = os.path.join(rgb_dir, split)
                if os.path.exists(split_rgb_dir):
                    for defect_type in os.listdir(split_rgb_dir):
                        defect_path = os.path.join(split_rgb_dir, defect_type)
                        if os.path.isdir(defect_path):
                            metadata["defect_types"].append({
                                "code": defect_type,
                                "name": defect_type
                            })
                            gt_dir = os.path.join(rgb_dir, 'GT', defect_type)
                            for filename in os.listdir(defect_path):
                                if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                    # Try all possible mask filename variations
                                    for mask_filename in get_possible_mask_names(filename):
                                        mask_path = os.path.join(gt_dir, mask_filename)
                                        if os.path.exists(mask_path):
                                            metadata["images"].append({
                                                "image_path": os.path.join(category, 'RGB', split, defect_type,
                                                                           filename),
                                                "mask_path": os.path.join(category, 'RGB', 'GT', defect_type,
                                                                          mask_filename),
                                                "defect_code": defect_type,
                                                "defect_name": defect_type,
                                                "category": category,
                                                "split": split
                                            })
                                        object_name = category
                                        defect_name = defect_type
                                        if object_name not in metadata["objects"]:
                                            metadata["objects"][object_name] = set()
                                        metadata["objects"][object_name].add(defect_name)

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata


def process_mvtec_ad_2_metadata(dataset_path):
    """Create metadata for mvtec_ad_2 dataset."""
    metadata = {
        "name": "mvtec_ad_2",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }
    for defect_category in os.listdir(dataset_path):
        defect_category_path = os.path.join(dataset_path, defect_category)
        if os.path.isdir(defect_category_path):
            if defect_category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(defect_category)
            test_public_dir = os.path.join(defect_category_path, 'test_public')
            if os.path.exists(test_public_dir):
                # 只处理 'bad' 子目录
                for sub_dir in ['bad']:
                    sub_path = os.path.join(test_public_dir, sub_dir)
                    if os.path.isdir(sub_path):
                        for filename in os.listdir(sub_path):
                            if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                                defect_type = sub_dir
                                metadata["defect_types"].append({
                                    "code": defect_type,
                                    "name": defect_type
                                })
                                # Try all possible mask filename variations
                                for mask_filename in get_possible_mask_names(filename):
                                    mask_path = os.path.join(test_public_dir, 'ground_truth', sub_dir, mask_filename)
                                    if os.path.exists(mask_path):
                                        metadata["images"].append({
                                            "image_path": os.path.join(defect_category, 'test_public', sub_dir,
                                                                       filename),
                                            "mask_path": os.path.join(defect_category, 'test_public',
                                                                      'ground_truth', sub_dir, mask_filename),
                                            "defect_code": defect_type,
                                            "defect_name": "defect",
                                            "category": defect_category,
                                            "split": "test"
                                        })
                                        object_name = defect_category
                                        defect_name = defect_type
                                        if object_name not in metadata["objects"]:
                                            metadata["objects"][object_name] = set()
                                        metadata["objects"][object_name].add(defect_name)
                                        break
                                else:
                                    print(
                                        f"Warning: Could not find mask for {filename} in {os.path.join(test_public_dir, 'ground_truth')}")

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata


def process_dagm_anomaly_detection_metadata(dataset_path):
    """Create metadata for DAGM_anomaly_detection dataset."""
    metadata = {
        "name": "DAGM_anomaly_detection",
        "categories": [],
        "defect_types": [],
        "images": [],
        "objects": {},
    }
    for object_category in os.listdir(dataset_path):
        object_category_path = os.path.join(dataset_path, object_category)
        if os.path.isdir(object_category_path):
            if object_category not in ["annotated", "prompt_short", "prompt"]:
                metadata["categories"].append(object_category)
            # 跳过Train类别
            for split in ['Test']:
                split_dir = os.path.join(object_category_path, split)
                if os.path.exists(split_dir):
                    label_dir = os.path.join(split_dir, 'Label')
                    for filename in os.listdir(split_dir):
                        if filename.lower().endswith(IMAGE_EXTENSIONS_CASE_INSENSITIVE):
                            base_name = os.path.splitext(filename)[0]
                            mask_filename = base_name + '_label.PNG'
                            mask_path = os.path.join(label_dir, mask_filename)
                            if os.path.exists(mask_path):
                                metadata["defect_types"].append({
                                    "code": "defect",
                                    "name": "defect"
                                })
                                metadata["images"].append({
                                    "image_path": os.path.join(object_category, split, filename),
                                    "mask_path": os.path.join(object_category, split, 'Label', mask_filename),
                                    "defect_code": "defect",
                                    "defect_name": "defect",
                                    "category": object_category,
                                    "split": split.lower()
                                })

                                object_name = object_category
                                defect_name = "defect"
                                if object_name not in metadata["objects"]:
                                    metadata["objects"][object_name] = set()
                                metadata["objects"][object_name].add(defect_name)

    for object_name in metadata["objects"]:
        metadata["objects"][object_name] = list(metadata["objects"][object_name])
    return metadata


def analyze_datasets_statistics(dataset_root):
    """Analyze and report statistics about objects and defects across all datasets"""
    statistics = {
        'datasets': {},
        'objects': defaultdict(lambda: {'defects': set(), 'datasets': set()}),
        'defects': defaultdict(lambda: {'objects': set(), 'datasets': set()})
    }

    # Walk through all dataset directories
    for dataset_name in os.listdir(dataset_root):
        dataset_path = os.path.join(dataset_root, dataset_name)
        json_path = os.path.join(dataset_path, 'dataset_info.json')

        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Initialize dataset entry
        dataset_stats = {
            'name': metadata.get('name', dataset_name),
            'objects': set(),
            'defects': set(),
            'object_defect_pairs': set()
        }

        # Process each image in the dataset
        for image_info in metadata.get('images', []):
            obj = image_info.get('category', 'unknown')
            defect = image_info.get('defect_name', 'unknown')

            # Update dataset statistics
            dataset_stats['objects'].add(obj)
            dataset_stats['defects'].add(defect)
            dataset_stats['object_defect_pairs'].add((obj, defect))

            # Update global object statistics
            statistics['objects'][obj]['defects'].add(defect)
            statistics['objects'][obj]['datasets'].add(dataset_name)

            # Update global defect statistics
            statistics['defects'][defect]['objects'].add(obj)
            statistics['defects'][defect]['datasets'].add(dataset_name)

        # Convert sets to lists for JSON serialization
        dataset_stats['objects'] = sorted(dataset_stats['objects'])
        dataset_stats['defects'] = sorted(dataset_stats['defects'])
        dataset_stats['object_defect_pairs'] = [
            {'object': pair[0], 'defect': pair[1]}
            for pair in sorted(dataset_stats['object_defect_pairs'])
        ]

        statistics['datasets'][dataset_name] = dataset_stats

    # Convert defaultdicts to regular dicts for JSON serialization
    statistics['objects'] = {
        obj: {
            'defects': sorted(data['defects']),
            'datasets': sorted(data['datasets'])
        }
        for obj, data in statistics['objects'].items()
    }

    statistics['defects'] = {
        defect: {
            'objects': sorted(data['objects']),
            'datasets': sorted(data['datasets'])
        }
        for defect, data in statistics['defects'].items()
    }

    return statistics


def save_statistics_report(statistics, output_dir):
    """Save the statistics report to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save full statistics
    with open(os.path.join(output_dir, 'full_statistics_report.json'), 'w') as f:
        json.dump(statistics, f, indent=4)

    # Save summary by objects
    objects_summary = []
    for obj, data in statistics['objects'].items():
        objects_summary.append({
            'object': obj,
            'defect_count': len(data['defects']),
            'dataset_count': len(data['datasets']),
            'defects': data['defects'],
            'datasets': data['datasets']
        })

    with open(os.path.join(output_dir, 'objects_summary_report.json'), 'w') as f:
        json.dump(sorted(objects_summary, key=lambda x: x['object']), f, indent=4)

    # Save summary by defects
    defects_summary = []
    for defect, data in statistics['defects'].items():
        defects_summary.append({
            'defect': defect,
            'object_count': len(data['objects']),
            'dataset_count': len(data['datasets']),
            'objects': data['objects'],
            'datasets': data['datasets']
        })

    with open(os.path.join(output_dir, 'defects_summary_report.json'), 'w') as f:
        json.dump(sorted(defects_summary, key=lambda x: x['defect']), f, indent=4)

    print(f"Statistics reports saved to {output_dir}")


def print_statistics_summary(statistics):
    """Print a human-readable summary of the statistics"""
    print("\n=== Dataset Statistics Summary ===")
    print(f"Total datasets analyzed: {len(statistics['datasets'])}")
    print(f"Total unique objects: {len(statistics['objects'])}")
    print(f"Total unique defects: {len(statistics['defects'])}")

    print("\nTop 5 objects by defect variety:")
    sorted_objects = sorted(
        statistics['objects'].items(),
        key=lambda x: len(x[1]['defects']),
        reverse=True
    )[:5]
    for obj, data in sorted_objects:
        print(f"- {obj}: {len(data['defects'])} defects")

    print("\nTop 5 defects by object variety:")
    sorted_defects = sorted(
        statistics['defects'].items(),
        key=lambda x: len(x[1]['objects']),
        reverse=True
    )[:5]
    for defect, data in sorted_defects:
        print(f"- {defect}: appears on {len(data['objects'])} objects")

def process_images_from_json(dataset_root):
    """Process images based on the dataset_info.json files."""
    datasets = [
        'KolektorSDD2', 'MulSen_AD', 'mvtec_ad_2', 'DAGM_anomaly_detection'
    ]

    for dataset_name in datasets:
        dataset_path = os.path.join(dataset_root, dataset_name)
        json_path = os.path.join(dataset_path, 'dataset_info.json')

        if os.path.exists(json_path):
            print(f"Processing images for {dataset_name}...")

            with open(json_path, 'r') as f:
                metadata = json.load(f)

            output_dir = os.path.join(dataset_path, 'annotated')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            for image_info in tqdm(metadata["images"], desc=f"Processing {dataset_name}"):
                if image_info["split"] == "train":
                    continue
                try:
                    image_path = os.path.join(dataset_path, image_info["image_path"])
                    mask_path = os.path.join(dataset_path, image_info["mask_path"])

                    # Read image (support all formats)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}")
                        continue

                    # Read mask (must be grayscale)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
                    if mask is None and image_info["defect_code"] == "bad":
                        print(f"Warning: Could not read mask {mask_path}")
                        continue

                    if mask is not None:
                        # Convert mask to binary if needed
                        if mask.dtype != np.uint8 or len(mask.shape) > 2:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                        # Draw defect boxes
                        result_image = draw_defect_boxes(image, mask)
                    else:
                        result_image = image

                    # Save annotated image
                    rel_path = os.path.dirname(image_info["image_path"])
                    save_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(save_dir, exist_ok=True)

                    output_path = os.path.join(save_dir, os.path.basename(image_info["image_path"]))
                    cv2.imwrite(output_path, result_image)

                except Exception as e:
                    print(f"Error processing {image_info['image_path']}: {str(e)}")
                    continue

if __name__ == "__main__":
    dataset_root = '/home/jiangyuxin/CODE/Datasets'
    output_dir = os.path.join(dataset_root, 'statistics_reports')

    # Step 1: Create JSON metadata files for all datasets
    create_dataset_info_json(dataset_root)

    # Step 2: Process images based on the JSON metadata
    process_images_from_json(dataset_root)

    # Step 3: Analyze and report statistics
    print("\nAnalyzing dataset statistics...")
    statistics = analyze_datasets_statistics(dataset_root)
    save_statistics_report(statistics, output_dir)
    print_statistics_summary(statistics)

    print("\nAll processing completed successfully!")