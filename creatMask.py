import os
import cv2
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import argparse


def extract_white_regions(mask):
    """Extract white regions from mask, return contours and rotated rectangle list"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []

    for contour in contours:
        # Calculate minimum rotated rectangle
        rect = cv2.minAreaRect(contour)
        regions.append(rect)

    return regions


def scale_regions(regions, img_size=(256, 256)):
    """Randomly scale the extracted white regions"""
    new_regions = []

    for region in regions:
        (cx, cy), (width, height), angle = region

        # Random scaling (300% to 500%)
        scale_factor = random.uniform(3, 5)
        new_width = max(5, width * scale_factor)  # Ensure minimum size
        new_height = max(5, height * scale_factor)  # Ensure minimum size

        new_regions.append(((cx, cy), (new_width, new_height), angle))

    return new_regions


def rotate_and_translate_regions(regions, img_size=(256, 256)):
    """Randomly rotate and translate the extracted white regions"""
    new_regions = []

    for region in regions:
        (cx, cy), (width, height), angle = region

        # Random rotation (-45° to 45°)
        new_angle = angle + random.uniform(-45, 45)

        # Random translation (±20 pixels)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        new_cx = min(max(cx + tx, 0), img_size[0])
        new_cy = min(max(cy + ty, 0), img_size[1])

        new_regions.append(((new_cx, new_cy), (width, height), new_angle))

    return new_regions


def calculate_overlap_ratio(new_mask, original_mask):
    """Calculate overlap ratio between two masks"""
    intersection = np.logical_and(new_mask == 255, original_mask == 255).sum()
    new_area = np.count_nonzero(new_mask)

    if new_area == 0:
        return 0

    return intersection / new_area


def generate_mask_from_existing(template_mask, max_attempts=50, overlap_threshold=0.3):
    """
    Generate new mask from existing Anomaly_mask by scaling, rotating, and translating white regions
    Parameters:
        template_mask: Original Anomaly_mask (numpy array)
        max_attempts: Maximum attempts
        overlap_threshold: Overlap ratio threshold
    """
    img_size = template_mask.shape[:2]

    # Set pixels greater than 0 to 255
    template_mask = np.where(template_mask > 0, 255, 0).astype(np.uint8)

    # Extract white regions
    regions = extract_white_regions(template_mask)

    # If no regions detected, return empty mask
    if not regions:
        return np.zeros(img_size, dtype=np.uint8)

    best_mask = None
    best_overlap = 0

    for attempt in range(max_attempts):
        # Randomly scale regions
        scaled_regions = scale_regions(regions, img_size)

        # Randomly rotate and translate scaled regions
        new_regions = rotate_and_translate_regions(scaled_regions, img_size)

        # Create new mask
        new_mask = np.zeros(img_size, dtype=np.uint8)
        for region in new_regions:
            box = cv2.boxPoints(region)
            box = np.int0(box)
            cv2.fillPoly(new_mask, [box], 255)

        # Calculate overlap ratio
        overlap_ratio = calculate_overlap_ratio(new_mask, template_mask)

        # Update best mask
        if overlap_ratio > best_overlap:
            best_mask = new_mask
            best_overlap = overlap_ratio

        # Early return if threshold reached
        if overlap_ratio >= overlap_threshold:
            return new_mask

    # Return best result if no qualifying mask found
    return best_mask if best_mask is not None else np.zeros(img_size, dtype=np.uint8)


def process_images(data_path, output_root, json_path):
    """Process all images and generate new masks from existing Anomaly_mask, while saving original masks"""
    with open(json_path, "r") as file:
        mask_info = json.load(file)

    classes = ["candle", "capsules", "cashew", "chewinggum", "fryum",
               "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

    for c in classes:
        print(f"Processing category: {c}")
        anomaly_types = list(mask_info[c].keys())

        # Normal images directory
        normal_dir = os.path.join(data_path, c, "Data/Images/Normal")
        if not os.path.exists(normal_dir):
            print(f"Normal images directory does not exist: {normal_dir}")
            continue

        # Anomaly_mask directory
        anomaly_mask_dir = os.path.join(data_path, c, "Data/Masks")
        if not os.path.exists(anomaly_mask_dir):
            print(f"Anomaly_mask directory does not exist: {anomaly_mask_dir}")
            continue

        image_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.JPG', '.png'))]

        for img_name in image_files:
            img_path = os.path.join(normal_dir, img_name)
            name = os.path.splitext(img_name)[0]

            # Read normal image (for size)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image: {img_path}")
                continue

            img = cv2.resize(img, (256, 256))
            height, width = img.shape[:2]

            for anomaly_type in anomaly_types:
                # Build Anomaly_mask subdirectory path
                type_mask_dir = os.path.join(anomaly_mask_dir)
                if not os.path.exists(type_mask_dir):
                    print(f"Anomaly_type directory does not exist: {type_mask_dir}")
                    continue

                # Get all mask files under this type
                mask_files = [f for f in os.listdir(type_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not mask_files:
                    print(f"No Anomaly_mask found: {type_mask_dir}")
                    continue

                # Generate masks for each normal image
                object_dir = os.path.join(output_root, c)
                defect_dir = os.path.join(object_dir, anomaly_type, name)
                os.makedirs(defect_dir, exist_ok=True)

                # Generate 5 different masks
                for i in range(1):
                    # Randomly select one Anomaly_mask as template
                    template_mask_path = os.path.join(type_mask_dir, random.choice(mask_files))
                    template_mask = cv2.imread(template_mask_path, cv2.IMREAD_GRAYSCALE)

                    if template_mask is None:
                        print(f"Cannot read template mask: {template_mask_path}")
                        continue

                    # Resize template to match normal image
                    original_template = cv2.resize(template_mask, (width, height))
                    original_template = np.where(original_template > 0, 255, 0).astype(np.uint8)

                    # Save original mask
                    # original_save_path = os.path.join(defect_dir, f"original_{i}.png")
                    # cv2.imwrite(original_save_path, original_template)
                    # print(f"Saved original mask to: {original_save_path}")

                    # Generate new mask
                    new_mask = generate_mask_from_existing(original_template)
                    new_mask = np.where(new_mask > 0, 255, 0).astype(np.uint8)
                    # Save new mask
                    new_save_path = os.path.join(defect_dir, f"{i}.png")
                    cv2.imwrite(new_save_path, new_mask)
                    print(f"Generated mask {i} for {anomaly_type} and saved to: {new_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate masks from existing anomaly masks")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with mask info")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_root", type=str, required=True, help="Output root directory")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_root, exist_ok=True)

    # Process all images
    process_images(args.data_path, args.output_root, args.json_path)


if __name__ == "__main__":
    main()