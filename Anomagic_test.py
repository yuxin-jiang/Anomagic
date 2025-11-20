import os
import time
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from ip_adapter.ip_adapter_anomagic import IPAdapter
import argparse
from metauas import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, read_image_as_tensor, safely_load_state_dict


class Timer:
    """High-precision timing utility class"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.start_time = time.perf_counter()
        self.last_checkpoint = self.start_time
        self.total_elapsed = 0.0
    def checkpoint(self, message=""):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_checkpoint
        self.total_elapsed = current_time - self.start_time
        self.last_checkpoint = current_time
        print(f"{message[:30]:<30} | Elapsed this step: {elapsed:.2f}s | Total elapsed: {self.total_elapsed:.2f}s")
        return elapsed


class MetaUASWrapper:
    """MetaUAS quality assessment model wrapper class"""
    def __init__(self, ckt_path, img_size=256):
        set_random_seed(1)
        encoder = 'efficientnet-b4'
        decoder = 'unet'
        encoder_depth = 5
        decoder_depth = 5
        num_crossfa_layers = 3
        alignment_type = 'sa'
        fusion_policy = 'cat'
        self.model = MetaUAS(encoder, decoder, encoder_depth, decoder_depth,
                             num_crossfa_layers, alignment_type, fusion_policy)
        self.model = safely_load_state_dict(self.model, ckt_path)
        self.model.cuda()
        self.model.eval()
        self.img_size = img_size
        self.thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]  # Multiple thresholds
    def evaluate_quality(self, generated_image, reference_image):
        """Evaluate the quality of the generated image, return results under multiple thresholds"""
        def preprocess(img):
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        gen_tensor = preprocess(generated_image).cuda()
        ref_tensor = preprocess(reference_image).cuda()
        with torch.no_grad():
            test_data = {
                "query_image": gen_tensor,
                "prompt_image": ref_tensor,
            }
            predicted_masks = self.model(test_data)
        pred_mask = predicted_masks.squeeze().detach().cpu().numpy()
        # Compute results under multiple thresholds
        results = {}
        for threshold in self.thresholds:
            quality_score = 1.0 - (pred_mask > threshold).mean()
            binary_mask = (pred_mask > threshold).astype(np.uint8)
            results[f"threshold_{threshold}"] = {
                "quality_score": quality_score,
                "binary_mask": binary_mask
            }
        return results


def load_models(ip_adapter, attn_bin, base_model_path, vae_model_path, image_encoder_path, device):
    """Load Diffusion models"""
    timer = Timer()
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(device=device, dtype=torch.float16)
    timer.checkpoint("VAE loaded")
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    ).to(device)
    timer.checkpoint("Base pipeline loaded")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    ip_model = IPAdapter(pipe, image_encoder_path, ip_adapter, attn_bin, device)
    timer.checkpoint("IP adapter loaded")
    return ip_model


def load_dataset_info(similarity_results_path, target_categories=None):
    """Load dataset information"""
    with open(similarity_results_path, "r") as f:
        similarity_results = json.load(f)
    object_defect_pairs = {}
    categories = set()
    defect_types = set()
    for item in similarity_results:
        obj = item["object"]
        defect = item["defect"]
        if target_categories is not None and obj not in target_categories:
            continue
        categories.add(obj)
        defect_types.add(defect)
        object_defect_pairs[(obj, defect)] = item
    return sorted(categories), sorted(defect_types), object_defect_pairs


def load_and_process_mask(mask_path, target_size=(512, 512)):
    """Load and process mask"""
    mask = Image.open(mask_path).convert("L").resize(target_size)
    mask_np = np.array(mask)
    return Image.fromarray((mask_np > 128).astype(np.uint8) * 255)


def load_mask_for_visa(object_category, defect_type, mask_base, style_image_path):
    """Load mask for VisA dataset"""
    image_name = os.path.basename(style_image_path)
    image_index = os.path.splitext(image_name)[0]
    mask_dir = os.path.join(mask_base, object_category, "Anomaly", str(image_index))
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith((".JPG", ".jpg", ".png"))]
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    mask_file = random.choice(mask_files)
    mask_path = os.path.join(mask_dir, mask_file)
    return load_and_process_mask(mask_path)


def get_mask_loader(dataset_type):
    """Return the corresponding mask loading function based on dataset type"""
    if dataset_type == "visa":
        return load_mask_for_visa
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_style_images_paths(args, object_category):
    """Get style image paths"""
    if args.dataset_type in ["visa"]:
        style_ref_path = os.path.join(args.dataset_base, object_category, "Data/Images/Normal")
        if not os.path.exists(style_ref_path):
            print(f"Warning: Style reference directory does not exist {style_ref_path}")
            return None
        style_images = [os.path.join(style_ref_path, f) for f in os.listdir(style_ref_path)
                        if f.endswith((".JPG", ".jpg", ".png"))]
    else:
        style_ref_path = os.path.join(args.dataset_base, object_category, "train/good")
        if not os.path.exists(style_ref_path):
            print(f"Warning: Style reference directory does not exist {style_ref_path}")
            return None
        style_images = [os.path.join(style_ref_path, f) for f in os.listdir(style_ref_path)
                        if f.endswith((".JPG", ".jpg", ".png"))]
    if not style_images:
        print(f"Warning: No style images found in {style_ref_path}")
        return None
    return style_images


def load_reference_data(
        object_category,
        defect_type,
        style_image_path,
        similarity_results_path,
        merged_datasets_path,
        anomaly_image_base,
        mvtec_base,
        mask_base,
        dataset_type,
        args=None
):
    """Load reference data"""
    with open(similarity_results_path, "r") as f:
        similarity_results = json.load(f)
    with open(merged_datasets_path, "r") as f:
        image_paths = json.load(f)
    config = next((item for item in similarity_results
                   if item["object"] == object_category and item["defect"] == defect_type), None)
    if not config:
        raise ValueError(f"No config found for {object_category} with {defect_type}")
    similar_objects = config["similar_objects"]
    base_path = anomaly_image_base
    if style_image_path is None:
        style_images = get_style_images_paths(args, object_category)
        if not style_images:
            raise FileNotFoundError(f"No style images found for {object_category}")
        style_image_path = random.choice(style_images)
    style_image_name = os.path.basename(style_image_path)
    style_image = Image.open(style_image_path).convert("RGB").resize((512, 512))
    defect_data = []
    for obj in similar_objects:
        for dataset in image_paths["datasets"]:
            if dataset["name"] == obj["dataset"]:
                for img_info in dataset["images"]:
                    if (img_info["category"] == obj["object"] and
                            img_info["defect_name"] == config["similar_defect"]):
                        defect_data.append({
                            "img_path": os.path.join(base_path, img_info["image_path"]),
                            "mask_path": os.path.join(base_path, img_info["mask_path"]),
                            "prompt_path": os.path.join(base_path, img_info["analysis_files_short"]),
                        })
    if not defect_data:
        raise ValueError("No defect images found")
    selected_data = random.choice(defect_data)
    defect_image = Image.open(selected_data["img_path"]).convert("RGB").resize((512, 512))
    raw_mask = Image.open(selected_data["mask_path"]).convert("L").resize((512, 512))
    mask_loader = get_mask_loader(dataset_type)
    processed_mask = mask_loader(object_category, defect_type, mask_base, style_image_path)
    with open(selected_data["prompt_path"], "r") as f:
        prompt = f.read()
    return {
        "style_image": style_image,
        "style_image_name": style_image_name,
        "defect_image": defect_image,
        "raw_mask": raw_mask,
        "processed_mask": processed_mask,
        "prompt": prompt,
        "similar_defect": config["similar_defect"],
    }


def apply_initial_mask_constraint(generated_mask, initial_mask):
    """
    Ensure the generated mask remains black in regions where the initial mask is black
    """
    # Convert PIL images to numpy arrays
    gen_mask_np = np.array(generated_mask)
    init_mask_np = np.array(initial_mask)
    # If sizes do not match, resize initial mask to match generated mask
    if gen_mask_np.shape != init_mask_np.shape:
        init_mask_np = np.array(initial_mask.resize(generated_mask.size))
    # Binarize initial mask (0 or 255)
    init_mask_binary = (init_mask_np > 128).astype(np.uint8) * 255
    # Force black in regions where initial mask is black
    constrained_mask = gen_mask_np * (init_mask_binary > 0)
    return Image.fromarray(constrained_mask)


def create_mask_overlay(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Create an image with a semi-transparent mask overlay effect
    :param image: Original image in PIL Image format
    :param mask: Mask in PIL Image format (single channel)
    :param alpha: Mask transparency (0-1)
    :param color: Mask color (RGB tuple)
    :return: PIL Image with mask overlay effect
    """
    # Convert image and mask to numpy arrays
    img_np = np.array(image)
    mask_np = np.array(mask)
    # Ensure mask is binarized
    mask_binary = (mask_np > 128).astype(np.uint8)
    # Create a colored mask
    color_mask = np.zeros_like(img_np)
    color_mask[..., 0] = color[0]  # R channel
    color_mask[..., 1] = color[1]  # G channel
    color_mask[..., 2] = color[2]  # B channel
    # Apply mask to original image
    overlay = img_np.copy()
    overlay = overlay.astype(np.float32)
    color_mask = color_mask.astype(np.float32)
    # Apply color only in mask regions
    overlay[mask_binary > 0] = overlay[mask_binary > 0] * (1 - alpha) + color_mask[mask_binary > 0] * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def generate_with_quality_control(
        ip_model,
        quality_model,
        object_category,
        defect_type,
        style_image_path,
        output_base,
        num_samples_target,
        min_quality_score=0.95,
        min_mask_coverage=0.01,
        max_attempts=100,
        num_inference_steps=50,
        scale=0.1,
        seed=42,
        strength=0.9,
        dataset_type="visa",
        mask_base=None,
        args=None
):
    """Generation process with quality control"""
    # Create standard directory structure
    # scale = round(random.uniform(0.1, 0.5), 2)  # Keep two decimal places
    visa_processed_dir = os.path.join(output_base, "visa_processed")
    category_dir = os.path.join(visa_processed_dir, object_category)
    abnormal_dir = os.path.join(category_dir, "abnormal_images")
    normal_dir = os.path.join(category_dir, "normal_images")
    masks_dir = os.path.join(category_dir, "masks_1")  # Changed to masks_1
    initial_masks_dir = os.path.join(category_dir, "input_masks")  # New initial mask directory
    overlays_dir = os.path.join(category_dir, "mask_overlays")  # New mask overlay directory
    stats_dir = os.path.join(category_dir, "stats")
    reference_dir = os.path.join(category_dir, "reference_images")  # New reference image directory
    # Create subdirectories for each threshold
    threshold_dirs = {}
    for threshold in quality_model.thresholds:
        threshold_dir = os.path.join(category_dir, f"masks_threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        threshold_dirs[threshold] = threshold_dir
    os.makedirs(abnormal_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(initial_masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    # Load reference data
    data = load_reference_data(
        object_category,
        defect_type,
        style_image_path,
        args.similarity_results,
        args.merged_datasets,
        args.anomaly_image_base,
        args.dataset_base,
        mask_base,
        dataset_type,
        args
    )
    # Save original image to normal_images
    original_save_path = os.path.join(normal_dir, data["style_image_name"])
    if not os.path.exists(original_save_path):
        data["style_image"].save(original_save_path)
    successful_samples = 0
    attempt_count = 0
    quality_scores = []
    while successful_samples < num_samples_target and attempt_count < max_attempts:
        attempt_count += 1
        # Record generation start time
        gen_start_time = time.time()
        # Generate image
        torch.manual_seed(seed + attempt_count)
        images = ip_model.generate(
            pil_image=data["defect_image"],
            num_samples=1,
            num_inference_steps=num_inference_steps,
            # prompt=data["prompt"],
            scale=scale,
            image=data["style_image"],
            mask_image_0=data["raw_mask"],
            mask_image=data["processed_mask"],
            strength=strength,
        )
        # Calculate generation time
        gen_time = time.time() - gen_start_time
        if not images:
            print(f"Generation failed - Time: {gen_time:.2f}s")
            continue
        generated_image = images[0]
        # Evaluate quality
        eval_start_time = time.time()
        quality_results = quality_model.evaluate_quality(
            generated_image,
            data["style_image"]
        )
        eval_time = time.time() - eval_start_time
        # Use threshold_0.5 result as main judgment criterion
        main_result = quality_results["threshold_0.5"]
        quality_score = main_result["quality_score"]
        anomaly_mask = main_result["binary_mask"]
        anomaly_mask_pil = Image.fromarray(anomaly_mask * 255).resize(data["processed_mask"].size)
        constrained_mask = apply_initial_mask_constraint(
            anomaly_mask_pil,
            data["processed_mask"]
        )
        constrained_mask_np = np.array(constrained_mask)
        # Recalculate coverage (after constraint)
        total_pixels = constrained_mask_np.size
        constrained_coverage = (constrained_mask_np > 0).sum() / total_pixels
        # Save qualified samples (meeting both quality score and coverage requirements)
        if quality_score >= min_quality_score and constrained_coverage >= min_mask_coverage:
            successful_samples += 1
            quality_scores.append(quality_score)
            # Generate filename (using style image name + sequence number)
            base_name = os.path.splitext(data["style_image_name"])[0]
            gen_image_name = f"{base_name}_{successful_samples}.png"
            mask_name = f"{base_name}_{successful_samples}_mask.png"
            initial_mask_name = f"{base_name}_{successful_samples}_mask.png"
            overlay_name = f"{base_name}_{successful_samples}_overlay.png"
            ref_image_name = f"{base_name}_{successful_samples}_reference.png"
            # Save files
            save_start_time = time.time()
            gen_save_path = os.path.join(abnormal_dir, gen_image_name)
            generated_image.save(gen_save_path)
            mask_save_path = os.path.join(masks_dir, mask_name)
            constrained_mask.save(mask_save_path)
            # Save initial mask
            initial_mask_save_path = os.path.join(initial_masks_dir, initial_mask_name)
            data["processed_mask"].save(initial_mask_save_path)
            overlay_image = create_mask_overlay(generated_image, constrained_mask, alpha=0.5, color=(255, 0, 0))
            overlay_save_path = os.path.join(overlays_dir, overlay_name)
            overlay_image.save(overlay_save_path)
            # Save reference image, corresponding to the generated defect image
            ref_save_path = os.path.join(reference_dir, ref_image_name)
            data["defect_image"].save(ref_save_path)
            # Save masks under all thresholds
            for threshold, result in quality_results.items():
                threshold_value = float(threshold.split("_")[1])
                threshold_mask = Image.fromarray(result["binary_mask"] * 255).resize(data["processed_mask"].size)
                constrained_threshold_mask = apply_initial_mask_constraint(threshold_mask, data["processed_mask"])
                threshold_mask_path = os.path.join(threshold_dirs[threshold_value], mask_name)
                constrained_threshold_mask.save(threshold_mask_path)
            save_time = time.time() - save_start_time
            total_time = gen_time + eval_time + save_time
            print(f"Generated qualified sample {successful_samples}/{num_samples_target}, "
                  f"quality score: {quality_score:.2f}, "
                  f"mask coverage: {constrained_coverage * 100:.2f}%, "
                  f"total time: {total_time:.2f}s (generation: {gen_time:.2f}s, evaluation: {eval_time:.2f}s, save: {save_time:.2f}s)")
        else:
            total_time = gen_time + eval_time
            print(f"Sample unqualified - quality score: {quality_score:.2f} (required: >= {min_quality_score}), "
                  f"mask coverage: {constrained_coverage * 100:.2f}% (required: >= {min_mask_coverage * 100:.2f}%), "
                  f"time: {total_time:.2f}s (generation: {gen_time:.2f}s, evaluation: {eval_time:.2f}s)")
    # Save statistics
    if quality_scores:
        stats_file = os.path.join(stats_dir, f"stats_{defect_type}.txt")
        with open(stats_file, "w") as f:
            f.write(f"Generation Statistics for {object_category}/{defect_type}\n")
            f.write(f"Total attempts: {attempt_count}\n")
            f.write(f"Successful samples: {successful_samples}\n")
            f.write(f"Average quality: {np.mean(quality_scores):.4f}\n")
            f.write(f"Min quality: {min(quality_scores):.4f}\n")
            f.write(f"Max quality: {max(quality_scores):.4f}\n")
            f.write(f"Min mask coverage: {min_mask_coverage}\n")
            f.write(f"Thresholds used: {quality_model.thresholds}\n")
    return quality_scores


def main():
    parser = argparse.ArgumentParser(description="Anomaly Image Generation with Quality Control")
    # Model path parameters
    parser.add_argument("--ip_adapter", type=str, required=True,
                        help="IP Adapter model checkpoint path")
    parser.add_argument("--attn_bin", type=str, required=True,
                        help="IP Adapter attention bin path")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base model path (Stable Diffusion)")
    parser.add_argument("--vae_model_path", type=str, required=True,
                        help="VAE model path")
    parser.add_argument("--image_encoder_path", type=str, required=True,
                        help="Image encoder path")
    parser.add_argument("--quality_model_path", type=str, required=True,
                        help="Quality assessment model path")
    # Dataset path parameters
    parser.add_argument("--dataset_type", type=str, choices=["mvtec", "visa"], required=True,
                        help="Dataset type (mvtec/visa)")
    parser.add_argument("--similarity_results", type=str, required=True,
                        help="Similarity results JSON path")
    parser.add_argument("--dataset_base", type=str, required=True,
                        help="Dataset base path")
    parser.add_argument("--mask_base", type=str, required=True,
                        help="Mask file base path")
    parser.add_argument("--anomaly_image_base", type=str, required=True,
                        help="Anomaly image base path")
    parser.add_argument("--merged_datasets", type=str, required=True,
                        help="Merged datasets JSON path")
    # Output parameters
    parser.add_argument("--output_base", type=str, required=True,
                        help="Results output base path")
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Target number of qualified images to generate per sample")
    parser.add_argument("--max_attempts", type=int, default=100,
                        help="Maximum attempts per sample")
    parser.add_argument("--min_quality", type=float, default=0.7,
                        help="Minimum quality score threshold")
    parser.add_argument("--min_mask_coverage", type=float, default=0.001,
                        help="Minimum mask coverage threshold (0-1)")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--ip_scale", type=float, default=1,
                        help="IP Adapter scale factor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Denoising strength")
    args = parser.parse_args()

    # Set categories based on dataset_type if not specified
    mvtec_categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    visa_categories = ["candle", "capsules", "cashew", "chewinggum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "fryum", "pipe_fryum"]
    if args.dataset_type == "mvtec":
        args.categories = mvtec_categories
    else:
        args.categories = visa_categories
    print(f"Processing categories: {args.categories} for dataset_type: {args.dataset_type}")

    # Environment setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load models
    print("Loading Diffusion models...")
    ip_model = load_models(
        args.ip_adapter,
        args.attn_bin,
        args.base_model_path,
        args.vae_model_path,
        args.image_encoder_path,
        device
    )
    print("Loading quality assessment model...")
    quality_model = MetaUASWrapper(args.quality_model_path)
    # Load dataset information
    print("\nLoading dataset information...")
    all_categories, all_defect_types, all_object_defect_pairs = load_dataset_info(
        args.similarity_results, args.categories
    )
    # Filter categories based on parameters
    object_defect_pairs = {k: v for k, v in all_object_defect_pairs.items()
                           if k[0] in args.categories}
    # Group defects by object category
    from collections import defaultdict
    category_to_defects = defaultdict(list)
    for (category, defect), config in object_defect_pairs.items():
        category_to_defects[category].append((defect, config))
    categories = sorted(category_to_defects.keys())
    if not categories:
        print(f"Warning: No data found for matching categories {args.categories}")
        return
    print(f"\nAvailable categories: {all_categories}")
    print(f"Processing {len(categories)} categories: {categories}")
    total_timer = Timer()
    all_quality_scores = []
    # Process each object category
    for object_category in categories:
        # Get all possible defect types for this category
        available_defects = [d for d, _ in category_to_defects[object_category]]
        # Get style image paths
        style_images = get_style_images_paths(args, object_category)
        if not style_images:
            print(f"Warning: No style images found for {object_category}")
            continue
        # Process each style image
        for style_image_path in style_images:
            # Randomly select a defect for the current style image
            defect_type, config = random.choice(category_to_defects[object_category])
            print(
                f"\n{'=' * 30}\nProcessing: {object_category} - Style image: {os.path.basename(style_image_path)} - Randomly selected defect: {defect_type}\n{'=' * 30}")
            quality_scores = generate_with_quality_control(
                ip_model=ip_model,
                quality_model=quality_model,
                object_category=object_category,
                defect_type=defect_type,
                style_image_path=style_image_path,
                output_base=args.output_base,
                num_samples_target=args.num_samples,
                min_quality_score=args.min_quality,
                min_mask_coverage=args.min_mask_coverage,
                max_attempts=args.max_attempts,
                num_inference_steps=args.num_inference_steps,
                scale=args.ip_scale,
                seed=args.seed,
                strength=args.strength,
                dataset_type=args.dataset_type,
                mask_base=args.mask_base,
                args=args
            )
            all_quality_scores.extend(quality_scores)
    # Print final statistics
    print("\n" + "=" * 50)
    print("Final generation results statistics:")
    print(f"Total processed categories: {len(categories)}")
    if all_quality_scores:
        print(f"Total qualified samples generated: {len(all_quality_scores)}")
        print(f"Average quality score: {np.mean(all_quality_scores):.2f} ± {np.std(all_quality_scores):.2f}")
        print(f"Highest quality score: {max(all_quality_scores):.2f}")
        print(f"Lowest quality score: {min(all_quality_scores):.2f}")
    print(f"Total runtime: {total_timer.total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()