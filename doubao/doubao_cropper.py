import os
import base64
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm
import time
from volcenginesdkarkruntime import Ark
import argparse


# Configuration
DOUBAO_CONFIG = {
    "API_KEY": "8b4dc34a-50db-443f-82ef-bbe127427b9a",
    "ENDPOINT_ID": "ep-20250408152822-n8w9d",
    "BASE_URL": "https://ark.cn-beijing.volces.com/api/v3",
    "TIMEOUT": 30,
    "MAX_RETRIES": 3,
    "SHORT_TEXT_MAX_LENGTH": 77  # Character limit for short text
}

LONG_PROMPT = (
    "Your task is to analyze industrial defect images. "
    "Please provide a detailed technical description focusing specifically on the area within the red box.\n\n"
    "STRICTLY FOLLOW THIS CAPTION TEMPLATE:\n"
    "    The image depicts [general description of the type of defect], which appears as [specific morphological feature, e.g., irregular patches, linear cracks, rounded spots, flaky fragments] with a [color description, e.g., dull gray, reddish-brown, yellowish-white, dark black] hue and shows [additional detail about texture or structure, e.g., rough edges, smooth surfaces, powdery texture, layered peeling].\n\n"
    "Guidelines:\n"
    "1. Describe the defect's visual characteristics precisely\n"
    "2. Do not speculate about causes or origins\n"
    "3. Use the exact template format provided above"
)

SHORT_PROMPT = (
    "Provide a concise, generic description of the industrial defect type shown in the image. "
    "STRICTLY LIMIT YOUR RESPONSE TO UNDER 77 CHARACTERS (INCLUDING SPACES). "
    "Focus on the defect type's typical characteristics that would apply to similar defects on other objects.\n\n"
    "Example response format:\n"
    "Scratch defect: linear surface mark with visible material removal. (72 chars)"
)


def preprocess_image(image_path, target_size=512):
    """Preprocess image for analysis"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_size, target_size), Image.BICUBIC)
        return img
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None


def analyze_with_doubao(image_path, prompt_type="long"):
    """Analyze image using Doubao API"""
    img = preprocess_image(image_path)
    if not img:
        return None

    # Encode image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    client = Ark(
        base_url=DOUBAO_CONFIG["BASE_URL"],
        api_key=DOUBAO_CONFIG["API_KEY"]
    )

    prompt = LONG_PROMPT if prompt_type == "long" else SHORT_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"
                    }
                }
            ]
        }
    ]

    for attempt in range(DOUBAO_CONFIG['MAX_RETRIES']):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_CONFIG['ENDPOINT_ID'],
                messages=messages
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"API request error: {e}")
            if attempt < DOUBAO_CONFIG['MAX_RETRIES'] - 1:
                time.sleep(2)

    return None


def get_available_datasets(input_dir):
    """Get list of available datasets"""
    datasets = []
    for item in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, item)):
            json_path = os.path.join(input_dir, item, "dataset_info_cropped.json")
            if os.path.exists(json_path):
                datasets.append(item)
    return datasets


def process_dataset(dataset, input_dir):
    """Process a single dataset"""
    dataset_path = os.path.join(input_dir, dataset)
    json_path = os.path.join(dataset_path, "dataset_info_cropped.json")

    print(f"\nProcessing dataset: {dataset}")

    # Create directories for saving prompts
    dataset_prompt_dir = os.path.join(dataset_path, "prompt_cropped")
    dataset_prompt_short_dir = os.path.join(dataset_path, "prompt_short_cropped")
    os.makedirs(dataset_prompt_dir, exist_ok=True)
    os.makedirs(dataset_prompt_short_dir, exist_ok=True)

    # Load JSON metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Remove existing analysis fields if present
    for image_info in metadata["images"]:
        image_info.pop("analysis_files", None)
        image_info.pop("analysis_files_short", None)

    # Save cleaned JSON file
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Process each image
    for image_info in tqdm(metadata["images"], desc="Processing images"):
        try:
            # Use cropped_image_path
            cropped_image_path = image_info.get("cropped_image_path")
            if not cropped_image_path:
                print(
                    f"Warning: No cropped_image_path found for {image_info.get('image_path', 'unknown')}, skipping...")
                continue

            image_path = os.path.join(dataset_path, cropped_image_path)
            object_category = image_info.get("category", "Unknown")
            defect_category = image_info.get("defect_name", "Unknown")

            # Get long analysis from API
            long_analysis = analyze_with_doubao(image_path, "long")

            # Get short analysis from API
            short_analysis = analyze_with_doubao(image_path, "short")

            if long_analysis and short_analysis:
                # Build paths for saving using original image_path's directory structure
                original_relative_path = os.path.dirname(image_info["image_path"])
                base_name = os.path.splitext(os.path.basename(image_info["image_path"]))[0]

                # Save long analysis
                long_filename = f"{base_name}_analysis.txt"
                long_path = os.path.join(dataset_prompt_dir, original_relative_path, long_filename)
                os.makedirs(os.path.dirname(long_path), exist_ok=True)
                with open(long_path, 'w') as f:
                    f.write(long_analysis)

                # Save short analysis
                short_filename = f"{base_name}_analysis_short.txt"
                short_path = os.path.join(dataset_prompt_short_dir, original_relative_path, short_filename)
                os.makedirs(os.path.dirname(short_path), exist_ok=True)
                with open(short_path, 'w') as f:
                    f.write(short_analysis)

                # Compute relative paths
                long_rel_path = os.path.relpath(long_path, dataset_path)
                short_rel_path = os.path.relpath(short_path, dataset_path)

                # Update metadata with both paths
                image_info["analysis_files"] = long_rel_path
                image_info["analysis_files_short"] = short_rel_path

                # Save updated metadata
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                # Print processing info
                print(f"\nObject: {object_category}")
                print(f"Defect: {defect_category}")
                print(f"Long analysis ({len(long_analysis)} chars): {long_analysis}")
                print(f"Short analysis ({len(short_analysis)} chars): {short_analysis}")
                print(f"Cropped image used: {cropped_image_path}")
                print(f"Saved to: {long_path} and {short_path}")
                print("-" * 80)

        except Exception as e:
            print(f"Error processing {image_info.get('image_path', 'unknown')}: {e}")

    print(f"Completed processing dataset: {dataset}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process cropped datasets with Doubao API for image analysis")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing datasets")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets to process (e.g., 'dataset1,dataset2') or 'all'")
    args = parser.parse_args()

    # Ensure prompt directories exist for each dataset
    available_datasets = get_available_datasets(args.input_dir)
    for dataset in available_datasets:
        dataset_prompt_dir = os.path.join(args.input_dir, dataset, "prompt_cropped")
        dataset_prompt_short_dir = os.path.join(args.input_dir, dataset, "prompt_short_cropped")
        os.makedirs(dataset_prompt_dir, exist_ok=True)
        os.makedirs(dataset_prompt_short_dir, exist_ok=True)

    # Determine selected datasets
    if args.datasets == 'all' or not args.datasets:
        selected_datasets = available_datasets
    else:
        selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in available_datasets]

    if not selected_datasets:
        print("No valid datasets selected or found.")
        return

    print(f"\nSelected datasets: {selected_datasets}")

    for dataset in selected_datasets:
        process_dataset(dataset, args.input_dir)

    print("\nAll selected datasets processed successfully!")


if __name__ == "__main__":
    main()