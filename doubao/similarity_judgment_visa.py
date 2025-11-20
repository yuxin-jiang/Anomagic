import os
import base64
from io import BytesIO
from PIL import Image
import json
import time
import random
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

# Prompt templates
LONG_PROMPT = (
    "Your task is to analyze industrial defect images and provide a detailed technical description. "
    "Your response should include:\n\n"
    "1. First, objectively describe the overall image content and context\n"
    "2. Then provide a detailed technical analysis of the defect including:\n"
    "   - Precise description of the defect's visual characteristics and location\n"
    "3. Focus only on observable characteristics - do not speculate about causes or origins\n\n"
    "Example response format:\n"
    "The image shows [general description of the object] and a [type of defect] is visible [location description]. The defect appears as [detailed description] with [notable features]."
)

SHORT_PROMPT = (
    "Provide a concise description of the industrial defect in the image (under 77 characters). "
    "Include the object type and defect type. Focus on key visual characteristics only.\n\n"
    "Example response format:\n"
    "Metal surface with scratch near edge, linear mark"
)

SIMILARITY_PROMPT = (
    "Given a target defect name '{target_defect}' and a list of defect names [{defect_list}], "
    "select the most similar defect name to the target defect from the list. "
    "If there is no highly similar category, analyze the characteristics of the target defect and the available defect names in the list, "
    "and select the defect name that has the highest degree of similarity, even if it is a weak similarity. "
    "Return ONLY the defect name, without any additional explanations or justifications."
)

TOP3_PROMPT = (
    "Given a target object '{target_object}' and a list of objects [{object_list}], "
    "please select the 3 most similar objects to the target object. "
    "If there are no similar objects, answer with 'None'. "
    "Answer with a comma-separated list of the similar objects, e.g., 'object1,object2,object3'."
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


def generate_short_text(long_text, object_category, defect_category):
    """Generate a concise version of the analysis text"""
    return f"{object_category} with {defect_category}"


def analyze_with_doubao(image_path, prompt_type="long"):
    """Analyze image using Doubao API"""
    img = preprocess_image(image_path)
    if not img:
        return None

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


def check_similarity(target_defect, defect_list):
    client = Ark(
        base_url=DOUBAO_CONFIG["BASE_URL"],
        api_key=DOUBAO_CONFIG["API_KEY"]
    )
    defect_str = ", ".join([f"'{d}'" for d in defect_list])
    prompt = SIMILARITY_PROMPT.format(target_defect=target_defect, defect_list=defect_str)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    for attempt in range(DOUBAO_CONFIG['MAX_RETRIES']):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_CONFIG['ENDPOINT_ID'],
                messages=messages
            )
            answer = resp.choices[0].message.content.strip()
            for defect in defect_list:
                if defect != "defect" and defect in answer:
                    return defect
            if "defect" in defect_list and "defect" in answer:
                return "defect"
            if 'If we consider' in answer:
                start_index = answer.find('consider') + len('consider') + 1
                end_index = answer.find(' could be', start_index)
                if start_index != -1 and end_index != -1:
                    return answer[start_index:end_index].strip()
            return answer
        except Exception as e:
            print(f"API request error: {e}")
            if attempt < DOUBAO_CONFIG['MAX_RETRIES'] - 1:
                time.sleep(2)
    return None


def get_top3_similar_objects(target_object, object_list):
    client = Ark(
        base_url=DOUBAO_CONFIG["BASE_URL"],
        api_key=DOUBAO_CONFIG["API_KEY"]
    )
    object_str = ", ".join([f"'{o}'" for o in object_list])
    prompt = TOP3_PROMPT.format(target_object=target_object, object_list=object_str)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    for attempt in range(DOUBAO_CONFIG['MAX_RETRIES']):
        try:
            resp = client.chat.completions.create(
                model=DOUBAO_CONFIG['ENDPOINT_ID'],
                messages=messages
            )
            answer = resp.choices[0].message.content.strip()
            if answer.lower() == 'none':
                return []
            similar_objects = []
            if target_object in object_list:
                similar_objects.append(target_object)
            for obj in object_list:
                if obj != target_object and obj in answer:
                    similar_objects.append(obj)
            if len(similar_objects) >= 3:
                return similar_objects[:3]
            return [obj.strip() for obj in answer.split(',') if obj.strip()]
        except Exception as e:
            print(f"API request error: {e}")
            if attempt < DOUBAO_CONFIG['MAX_RETRIES'] - 1:
                time.sleep(2)
    return []


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Unable to parse JSON file {file_path}.")
        return None


def build_object_dataset_mapping(target_json):
    """Build a mapping of object to its original dataset"""
    object_dataset_map = {}
    for dataset in target_json["datasets"]:
        dataset_name = dataset.get("name", "Unknown")
        for image in dataset["images"]:
            obj = image.get("category", "Unknown")
            if obj not in object_dataset_map:
                object_dataset_map[obj] = dataset_name
    return object_dataset_map


def process_dataset(dataset_info_file, target_json, dataset_name):
    dataset_json = load_json(dataset_info_file)
    if not dataset_json:
        return

    # For VisA_20220922, manually define defects for each category
    if dataset_name == "VisA_20220922":
        category_defects = {
            "candle": ["crack", "deformed candle surface", "deformed wick", "misplaced wick", "color", "bulge", "contamination", "depression"],
            "capsules": ["contamination","color", "deformed", "squeeze", "scratch"],
            "cashew": ["color", "crack", "contaminated","hole", "bulge", "scratch", "deformed"],
            "chewinggum": ["crack", "deformed","scratch", "bulge"],
            "fryum": ["color", "crack", "contamination", "deformed", "scratch"],
            "macaroni1": ["crack", "color", "hole", "contamination", "scratch"],
            "macaroni2": ["crack", "color", "hole", "contamination", "scratch"],
            "pcb1": ["bent lead", "contamination", "missing_component", "misplaced", "scratch"],
            "pcb2": ["bent lead", "contamination", "missing_component", "scratch"],
            "pcb3": ["bent lead", "bent diode", "contamination", "missing_component", "scratch"],
            "pcb4": ["crack", "deformed","contamination", "broken", "missing_component", "scratch", "misplaced"],
            "pipe_fryum": ["contamination", "crack", "color", "hole", "scratch", "deformed"]
        }
    else:
        category_defects = {
            "bottle": ["crack", "color", "hole", "contamination", "scratch"],
            "capsules": ["crack", "color", "hole", "contamination", "squeeze", "scratch"],
            "cable": ["color", "crack", "contaminated", "hole", "bulge", "scratch", "deformed"],
            "carpet": ["color", "contamination", "scratch", "hole", "crack"],
            "grid": ["color", "crack", "contamination", "scratch","bent"],
            "hazelnut": ["crack", "color", "hole", "contamination", "scratch"],
            "leather": ["crack", "color", "hole", "contamination", "scratch"],
            "metal_nut": ["color", "crack", "contamination", "scratch","bent"],
            "pill": ["crack", "deformed", "scratch", "bulge", "hole", "scratch"],
            "screw": ["bent", "crack", "deformed", "scratch", "bulge", "hole", "scratch"],
            "tile": ["color", "contamination", "scratch", "hole", "crack"],
            "toothbrush": ["contamination", "crack", "color", "hole", "scratch"],
            "transistor": ["contamination", "crack", "color", "hole", "scratch"],
            "wood": ["contamination", "crack", "color", "hole", "scratch"],
            "zipper": ["contamination", "crack", "color", "hole", "scratch"]
        }

    # Build mappings
    all_defects = []
    defect_object_map = {}
    object_dataset_map = build_object_dataset_mapping(target_json)

    for dataset in target_json["datasets"]:
        for image in dataset["images"]:
            defect = image.get("defect_name", "Unknown")
            obj = image.get("category", "Unknown")
            if defect not in all_defects:
                all_defects.append(defect)
            if defect not in defect_object_map:
                defect_object_map[defect] = []
            if obj not in defect_object_map[defect]:
                defect_object_map[defect].append(obj)

    result = []
    print(f"Processing dataset: {dataset_name}")

    processed_object_defect = set()

    for image in dataset_json["images"]:
        object_category = image["category"]

        # For VisA_20220922, use manually defined defects
        if dataset_name == "VisA_20220922":
            if object_category in category_defects:
                defect_categories = category_defects[object_category]
            else:
                defect_categories = ["defect"]  # Default defect
        else:
            if object_category in category_defects:
                defect_categories = category_defects[object_category]
            else:
                defect_categories = ["defect"]  # Default defect

        for defect_category in defect_categories:
            if (object_category, defect_category) in processed_object_defect:
                continue

            if defect_category in all_defects:
                similar_defect = defect_category
            else:
                similar_defect = check_similarity(defect_category, all_defects)

            if not similar_defect:
                continue

            current_defect_objects = defect_object_map.get(similar_defect, [])

            if object_category in current_defect_objects:
                similar_objects = [object_category]
                remaining_count = 3 - len(similar_objects)
                if remaining_count > 0:
                    additional_similar_objects = get_top3_similar_objects(object_category, current_defect_objects)
                    additional_similar_objects = [obj for obj in additional_similar_objects
                                                  if obj not in similar_objects and obj != 'None' and obj != 'none']
                    similar_objects.extend(additional_similar_objects[:remaining_count])
            else:
                similar_objects = get_top3_similar_objects(object_category, current_defect_objects)

            similar_objects = [obj for obj in similar_objects if obj != 'None' and obj != 'none']

            if not similar_objects:
                defect_objects = defect_object_map.get(similar_defect, [])
                if defect_objects:
                    similar_objects = [random.choice(defect_objects)] if len(defect_objects) == 1 else defect_objects[:3]
            elif len(similar_objects) < 3:
                defect_objects = defect_object_map.get(similar_defect, [])
                available_objects = [obj for obj in defect_objects if obj not in similar_objects and obj != 'None']
                while len(similar_objects) < 3 and available_objects:
                    random_obj = random.choice(available_objects)
                    similar_objects.append(random_obj)
                    available_objects.remove(random_obj)

            # Ensure each similar object has its correct dataset
            similar_objects_with_dataset = []
            for obj in similar_objects:
                dataset = object_dataset_map.get(obj, "Unknown")
                similar_objects_with_dataset.append({
                    "object": obj,
                    "dataset": dataset
                })

            result.append({
                "object": object_category,
                "defect": defect_category,
                "similar_defect": similar_defect,
                "similar_objects": similar_objects_with_dataset
            })

            print(f"  Object category: {object_category}")
            print(f"  Defect category: {defect_category}")
            print(f"  Similar defect: {similar_defect}")
            object_info = ', '.join([f"{obj['object']} ({obj['dataset']})" for obj in similar_objects_with_dataset])
            print(f"  Similar objects: {object_info}")
            print()

            processed_object_defect.add((object_category, defect_category))

    dataset_dir = os.path.dirname(dataset_info_file)
    output_file = os.path.join(dataset_dir, f"{dataset_name}_similarity_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process datasets for defect and object similarity")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file containing merged_json and datasets list")
    args = parser.parse_args()

    # Load config from JSON file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    merged_json_path = config.get("merged_json")
    datasets_config = config.get("datasets", [])

    target_json = load_json(merged_json_path)
    if not target_json:
        return

    for dataset_entry in datasets_config:
        dataset_file = dataset_entry.get("file")
        dataset_name = dataset_entry.get("name")
        if dataset_file and dataset_name:
            process_dataset(dataset_file, target_json, dataset_name)


if __name__ == "__main__":
    main()