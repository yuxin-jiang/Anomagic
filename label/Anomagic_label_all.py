import json
import datetime
from collections import defaultdict
import os
import argparse


def merge_ad_datasets(dataset_paths, output_path):
    """
    Merge multiple anomaly detection datasets' JSON files into one JSON file, ensuring field completeness

    :param dataset_paths: Dataset JSON file path dictionary {
        'AITEX': 'path/to/aitex.json',
        'BTech': 'path/to/btech.json',
        ...
    }
    :param output_path: Output JSON file path
    :return: Merged data dictionary
    """
    merged_data = {
        "datasets": [],
        "statistics": defaultdict(int),
        "version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "data_sources": list(dataset_paths.keys())  # Record all data sources
    }

    for dataset_name, dataset_path in dataset_paths.items():
        print(f"Processing {dataset_name}...")
        try:
            # Read dataset data from JSON file
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)

            # Standardize image entries
            standardized_images = []
            for img_entry in dataset_data.get("images", []):
                if not isinstance(img_entry, dict):
                    print(f"Warning: Non-dictionary entry found in 'images' list of {dataset_name}: {img_entry}")
                    continue

                # Handle different field aliases used by datasets
                defect_name = img_entry.get("defect_name",
                                            img_entry.get("defect_type",
                                                          img_entry.get("anomaly_type", "")))

                category = img_entry.get("category",
                                         img_entry.get("class",
                                                       img_entry.get("object_type", "")))

                # Prepend dataset name to paths
                image_path = os.path.join(dataset_name, img_entry.get("cropped_image_path", ""))
                mask_path = os.path.join(dataset_name, img_entry.get("cropped_mask_path", ""))

                # Handle analysis_files paths to avoid duplicate prompt
                analysis_files = img_entry.get("analysis_files", "")
                if analysis_files.startswith("prompt/prompt"):
                    analysis_files = analysis_files[7:]  # Remove duplicate prompt
                analysis_files = os.path.join(dataset_name, analysis_files)

                # Handle analysis_files_short paths to avoid duplicate prompt_short
                analysis_files_short = img_entry.get("analysis_files_short", "")
                if analysis_files_short.startswith("prompt_short/prompt_short"):
                    analysis_files_short = analysis_files_short[len("prompt_short/"):]
                analysis_files_short = os.path.join(dataset_name, analysis_files_short)

                # Build standardized entry
                std_entry = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "defect_code": img_entry.get("defect_code", ""),
                    "defect_name": defect_name,
                    "category": category,
                    "split": img_entry.get("split",
                                           img_entry.get("phase", "test")),  # Default to test
                    "analysis_files": analysis_files,
                    "analysis_files_short": analysis_files_short,
                    "dataset": dataset_name  # Add dataset identifier
                }
                standardized_images.append(std_entry)

            # Process defect types
            standardized_defect_types = []
            for defect in dataset_data.get("defect_types", []):
                if isinstance(defect, dict):
                    # Handle dictionary-form defect types
                    standardized_defect_types.append({
                        "code": defect.get("code", ""),
                        "name": defect.get("name", "")
                    })
                elif isinstance(defect, str):
                    # Handle string-form defect types
                    standardized_defect_types.append({
                        "code": defect,
                        "name": defect
                    })
                else:
                    # Handle other types
                    standardized_defect_types.append({
                        "code": str(defect),
                        "name": str(defect)
                    })

            # Build standardized dataset structure
            standardized_dataset = {
                "name": dataset_name,
                "categories": dataset_data.get("categories", []),
                "defect_types": standardized_defect_types,
                "images": standardized_images
            }

            # Add to merged data
            merged_data["datasets"].append(standardized_dataset)

            # Update statistics
            merged_data["statistics"]["total_datasets"] += 1
            merged_data["statistics"]["total_categories"] += len(set(standardized_dataset["categories"]))  # Use set for deduplication
            merged_data["statistics"]["total_defect_types"] += len(standardized_defect_types)
            merged_data["statistics"]["total_images"] += len(standardized_images)

            # Statistics by split
            for img in standardized_images:
                split = img["split"].lower()  # Normalize to lowercase
                if split in ["train", "test", "val", "validation"]:
                    key = f"{split}_images"
                    merged_data["statistics"][key] = merged_data["statistics"].get(key, 0) + 1

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    # Compute global unique categories and defect types
    all_categories = set()
    all_defect_types = set()

    for dataset in merged_data["datasets"]:
        all_categories.update(dataset["categories"])
        for defect in dataset["defect_types"]:
            # Ensure defect is already in dict form
            all_defect_types.add(f"{defect.get('code', '')}:{defect.get('name', '')}")

    merged_data["statistics"]["unique_categories"] = len(all_categories)
    merged_data["statistics"]["unique_defect_types"] = len(all_defect_types)

    # Save merged data
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully merged {len(merged_data['datasets'])} datasets. Saved to {output_path}")

    # Print summary statistics
    print("\nMerged Statistics Summary:")
    for stat, value in merged_data["statistics"].items():
        print(f"{stat.replace('_', ' ').title()}: {value}")

    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge anomaly detection datasets")
    parser.add_argument("--datasets_config", type=str, required=True, help="Path to JSON file containing dataset_paths dict")
    parser.add_argument("--output_path", type=str, default="merged_ad_datasets.json", help="Output path for merged JSON")
    args = parser.parse_args()

    # Load dataset_paths from config file
    with open(args.datasets_config, 'r', encoding='utf-8') as f:
        dataset_paths = json.load(f)

    # Check if each path exists
    for dataset_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Error: The JSON file for {dataset_name} at {path} does not exist.")
            exit(1)

    # Perform merge
    merged_data = merge_ad_datasets(dataset_paths, args.output_path)

    # Print sample from first dataset
    if merged_data["datasets"]:
        sample_dataset = merged_data["datasets"][0]
        print(f"\nSample dataset '{sample_dataset['name']}':")
        print(f"Categories: {sample_dataset['categories']}")
        print(f"Defect types: {[d['name'] for d in sample_dataset['defect_types'][:2]]}...")

        sample_image = sample_dataset["images"][0]
        print("\nSample image entry:")
        print(json.dumps(sample_image, indent=2))