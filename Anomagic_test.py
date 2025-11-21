import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from ip_adapter.ip_adapter_anomagic import Anomagic
import argparse
from metauas import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, read_image_as_tensor, safely_load_state_dict


class BatchAnomalyGenerator:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anomagic_model = None
        self.quality_model = None
        self.dataset_info = None
        self.all_samples = []

    def load_models(self, ip_ckpt, ip_ckpt_1, base_model_path, vae_model_path, image_encoder_path,
                    quality_model_path=None, dataset_info_path=None):
        """Load all required models and dataset information."""
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        vae = AutoencoderKL.from_pretrained(vae_model_path).to(device=self.device, dtype=torch.float16)

        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        ).to(self.device)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Verify image_encoder_path is a valid local directory
        if not os.path.isdir(image_encoder_path):
            raise ValueError(f"Image encoder path {image_encoder_path} is not a valid directory")
        print(f"Loading image encoder from local path: {image_encoder_path}")

        try:
            self.anomagic_model = Anomagic(pipe, image_encoder_path, ip_ckpt, ip_ckpt_1, self.device)
        except Exception as e:
            print(f"Error loading Anomagic with image_encoder_path {image_encoder_path}: {str(e)}")
            raise

        if quality_model_path:
            set_random_seed(1)
            self.quality_model = MetaUAS(
                'efficientnet-b4',
                'unet',
                5,
                5,
                3,
                'sa',
                'cat'
            )
            self.quality_model = safely_load_state_dict(self.quality_model, quality_model_path)
            self.quality_model.to(self.device).eval()

        if dataset_info_path:
            try:
                with open(dataset_info_path, 'r') as f:
                    self.dataset_info = json.load(f)

                self.all_samples = []
                for dataset in self.dataset_info.get('datasets', []):
                    self.all_samples.extend(dataset.get('images', []))

                if not self.all_samples:
                    raise ValueError("No valid sample data found in JSON file")
                print(f"Loaded dataset information with {len(self.all_samples)} samples")
            except Exception as e:
                raise ValueError(f"Failed to load dataset information: {str(e)}")

    def _get_random_references(self, num_samples=1):
        """Randomly select reference samples from JSON data."""
        if not self.all_samples:
            raise ValueError("No available sample data")

        if num_samples > len(self.all_samples):
            num_samples = len(self.all_samples)
            print(f"Warning: Requested reference samples exceed available samples, using {num_samples} samples")

        selected_samples = random.sample(self.all_samples, num_samples)
        base_path = "Datasets"

        processed_samples = []
        for sample in selected_samples:
            image_path = os.path.join(base_path, sample['image_path'])
            mask_path = os.path.join(base_path, sample['mask_path'])
            prompt = self._get_prompt_for_sample(sample, base_path)

            processed_samples.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'prompt': prompt,
                'sample_id': os.path.splitext(os.path.basename(sample['image_path']))[0],
                'defect_name': sample.get('defect_name', 'unknown'),
                'dataset': sample.get('dataset', 'unknown')
            })

        return processed_samples

    def _get_prompt_for_sample(self, sample, base_path):
        """Retrieve prompt text for a sample."""
        if 'analysis_files_short' in sample:
            prompt_path = os.path.join(base_path, sample['analysis_files_short'])
            try:
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
            except Exception as e:
                print(f"Warning: Failed to read short prompt file {prompt_path}: {str(e)}")

        if 'analysis_files' in sample:
            prompt_path = os.path.join(base_path, sample['analysis_files'])
            try:
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
            except Exception as e:
                print(f"Warning: Failed to read prompt file {prompt_path}: {str(e)}")

        defect_name = sample.get('defect_name', 'defect')
        return f"The image shows an object with a {defect_name} defect."

    def _prepare_output_structure(self, output_root):
        """Prepare output directory structure."""
        subdirs = {
            "input_images": os.path.join(output_root, "input_images"),
            "reference_images": os.path.join(output_root, "reference_images"),
            "reference_masks": os.path.join(output_root, "reference_masks"),
            "generated_images": os.path.join(output_root, "generated_images"),
            "masks": os.path.join(output_root, "masks"),
            "mask_overlays": os.path.join(output_root, "mask_overlays"),
            "prompts": os.path.join(output_root, "prompts")
        }

        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for t in thresholds:
            subdirs[f"masks_threshold_{t}"] = os.path.join(output_root, f"masks_threshold_{t}")

        for dir_path in subdirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return subdirs

    def _preprocess_for_quality(self, img):
        """Preprocess image for quality evaluation with consistent dimensions."""
        img = img.resize((256, 256))
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def generate_anomaly_image(self,
                              normal_image,
                              normal_image_path,
                              reference_data_list,
                              output_root,
                              num_inference_steps=50,
                              ip_scale=0.3,
                              seed=42,
                              strength=0.3,
                              evaluate_quality=True,
                              num_anomaly_images=1):
        """Generate anomaly images using single-image generation with reference masks."""
        subdirs = self._prepare_output_structure(output_root)
        target_size = (512, 512)
        normal_image = normal_image.resize(target_size)
        results = []

        if evaluate_quality and self.quality_model:
            ref_tensor = self._preprocess_for_quality(normal_image)
            print(f"Reference tensor shape for {normal_image_path}: {ref_tensor.shape}")

        for ref_idx, ref_data in enumerate(reference_data_list):
            try:
                reference_image = Image.open(ref_data['image_path']).convert("RGB").resize(target_size)
                if not os.path.exists(ref_data['mask_path']):
                    print(f"Error: Reference mask not found at {ref_data['mask_path']}")
                    continue
                reference_mask = Image.open(ref_data['mask_path']).convert("L").resize(target_size)
                reference_mask_np = np.array(reference_mask)
                reference_mask = Image.fromarray((reference_mask_np > 128).astype(np.uint8) * 255)

                reference_mask_binary = (reference_mask_np > 128).astype(np.uint8) * 255

                # 计算白色区域占比
                white_pixels = np.sum(reference_mask_binary > 0)
                total_pixels = reference_mask_binary.size
                white_ratio = white_pixels / total_pixels

                # 如果白色区域占比小于3%，则扩大到5%
                if white_ratio < 0.03:
                    print(f"White area ratio is {white_ratio:.4f}, expanding to at least 5%")

                    # 计算需要膨胀的程度
                    current_area = white_pixels
                    target_area = int(total_pixels * 0.05)  # 目标为5%

                    # 使用形态学膨胀操作扩大掩码
                    kernel_size = 3
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)

                    # 迭代膨胀直到达到目标面积或最大迭代次数
                    max_iterations = 20
                    expanded_mask = reference_mask_binary.copy()

                    for i in range(max_iterations):
                        current_white = np.sum(expanded_mask > 0)
                        if current_white >= target_area:
                            break

                        # 执行膨胀操作
                        expanded_mask = cv2.dilate(expanded_mask, kernel, iterations=1)

                        # 检查是否达到目标
                        new_white = np.sum(expanded_mask > 0)
                        if new_white == current_white:  # 如果没有变化，停止迭代
                            break

                    # 更新掩码
                    reference_mask_binary = expanded_mask

                    new_white_ratio = np.sum(expanded_mask > 0) / total_pixels
                    print(f"After expansion, white area ratio is {new_white_ratio:.4f}")


                mask_image = Image.fromarray(reference_mask_binary)
                mask_image                                                = reference_mask

                print(f"Processing reference {ref_idx}: {ref_data['image_path']}")
                print(f"Reference image mode: {reference_image.mode}, size: {reference_image.size}")
                print(f"Normal image mode: {normal_image.mode}, size: {normal_image.size}")
                print(f"Normal mask mode: {mask_image.mode}, size: {mask_image.size}")
                print(f"Reference mask mode: {reference_mask.mode}, size: {reference_mask.size}")

                images = []
                for img_idx in range(num_anomaly_images):
                    torch.manual_seed(seed + ref_idx + img_idx)
                    single_image = self.anomagic_model.generate(
                        pil_image=reference_image,
                        num_samples=1,
                        num_inference_steps=num_inference_steps,
                        prompt=ref_data['prompt'],
                        scale=ip_scale,
                        image=normal_image,
                        mask_image=mask_image,
                        mask_image_0=reference_mask,
                        strength=strength,
                    )
                    if not single_image:
                        print(f"Warning: No image generated for anomaly {img_idx}, reference {ref_idx}")
                        continue
                    images.append(single_image[0])

                print(f"Generated {len(images)} images for reference {ref_idx}, normal image {normal_image_path}")
                if not images:
                    raise RuntimeError("No images generated for reference")

                normal_id = os.path.splitext(os.path.basename(normal_image_path))[0]
                for img_idx, generated_image in enumerate(images):
                    if not isinstance(generated_image, Image.Image):
                        print(f"Error: Generated image {img_idx} is not a PIL Image, type: {type(generated_image)}")
                        continue

                    save_name = f"{normal_id}_ref{ref_data['sample_id']}_anomaly{img_idx}"

                    # Save to separate folders instead of all in input_images
                    normal_image.save(os.path.join(subdirs["input_images"], f"{save_name}_normal.png"))
                    reference_image.save(os.path.join(subdirs["reference_images"], f"{save_name}_reference.png"))
                    mask_image.save(os.path.join(subdirs["reference_masks"], f"{save_name}_mask.png"))
                    reference_mask.save(os.path.join(subdirs["reference_masks"], f"{save_name}_ref_mask.png"))
                    generated_image.save(os.path.join(subdirs["generated_images"], f"{save_name}_generated.png"))

                    with open(os.path.join(subdirs["prompts"], f"{save_name}_prompt.txt"), 'w') as f:
                        f.write(ref_data['prompt'])

                    quality_results = None
                    if evaluate_quality and self.quality_model:
                        try:
                            gen_tensor = self._preprocess_for_quality(generated_image)
                            print(f"Generated tensor shape for {save_name}: {gen_tensor.shape}")

                            with torch.no_grad():
                                pred_masks = self.quality_model({
                                    "query_image": gen_tensor,
                                    "prompt_image": ref_tensor
                                })
                            print(f"Prediction masks shape for {save_name}: {pred_masks.shape}")
                            pred_mask = pred_masks.squeeze().detach().cpu().numpy()

                            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                            quality_results = {
                                f"threshold_{t}": {
                                    "quality_score": float(1.0 - (pred_mask > t).mean()),
                                    "binary_mask": (pred_mask > t).astype(np.uint8).tolist()
                                } for t in thresholds
                            }

                            for t in thresholds:
                                threshold_mask = np.array(quality_results[f"threshold_{t}"]["binary_mask"]).astype(np.uint8)
                                Image.fromarray(threshold_mask * 255).save(
                                    os.path.join(subdirs[f"masks_threshold_{t}"], f"{save_name}_mask.png"))

                            main_mask = np.array(quality_results["threshold_0.5"]["binary_mask"]).astype(np.uint8)
                            Image.fromarray(main_mask * 255).save(
                                os.path.join(subdirs["masks"], f"{save_name}_mask.png"))

                            overlay_img = generated_image.copy()
                            mask_overlay = Image.fromarray(main_mask * 255).resize(generated_image.size).convert("L")
                            red_mask = Image.new("RGBA", generated_image.size, (255, 0, 0, 128))
                            overlay_img.paste(red_mask, mask=mask_overlay)
                            overlay_img.save(os.path.join(subdirs["mask_overlays"], f"{save_name}_overlay.png"))

                            with open(os.path.join(output_root, f"{save_name}_quality.json"), 'w') as f:
                                json.dump(quality_results, f, indent=2)

                        except Exception as e:
                            print(f"Quality evaluation failed for {save_name}: {str(e)}")
                            quality_results = None

                    results.append({
                        'normal_image': normal_image_path,
                        'reference_image': ref_data['image_path'],
                        'generated_image': os.path.join(subdirs["generated_images"], f"{save_name}_generated.png"),
                        'prompt': ref_data['prompt'],
                        'sample_id': ref_data['sample_id'],
                        'anomaly_index': img_idx,
                        'mask_path': ref_data['mask_path'],
                        'ref_mask_path': ref_data['mask_path'],
                        'quality': quality_results,
                        'defect_name': ref_data['defect_name'],
                        'dataset': ref_data['dataset']
                    })

            except Exception as e:
                print(f"Error processing reference sample {ref_idx} for {normal_image_path}: {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Anomaly Image Generation with Quality Control")

    parser.add_argument("--ip_ckpt", type=str, default="checkpoints/anomagic_0.bin",
                        help="IP Adapter模型 checkpoint 路径")
    parser.add_argument("--ip_ckpt_1", type=str, default="checkpoints/att.bin",
                        help="IP Adapter模型 checkpoint 路径")
    parser.add_argument("--base_model", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE",
                        help="基础模型路径 (Stable Diffusion)")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="VAE模型路径")
    parser.add_argument("--image_encoder", type=str, default="models/image_encoder/",
                        help="图像编码器路径")
    parser.add_argument("--quality_model", type=str, default="weights/metauas-256.ckpt",
                        help="质量评估模型路径")
    parser.add_argument("--dataset_info_path", type=str, default="Datasets/merged_ad_datasets.json",
                        help="合并数据集 JSON 路径")
    parser.add_argument("--normal_images_dir", type=str, default="Datasets",
                        help="Directory containing normal images")
    parser.add_argument("--output_base", type=str, default="output",
                        help="结果输出基础路径")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of normal images to process")
    parser.add_argument("--steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--ip_scale", type=float, default=0.1,
                        help="IP Adapter 缩放因子")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="去噪强度")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="是否评估生成图像质量")
    parser.add_argument("--num_anomaly_images", type=int, default=1,
                        help="Number of anomaly images to generate per reference sample")
    parser.add_argument("--num_references", type=int, default=8,
                        help="Number of reference samples per normal image")
    parser.add_argument("--cuda_device", type=str, default="0",
                        help="CUDA 设备 ID")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="Hugging Face 镜像端点")

    args = parser.parse_args()

    # 环境设置
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = BatchAnomalyGenerator()

    try:
        generator.load_models(
            ip_ckpt=args.ip_ckpt,
            ip_ckpt_1=args.ip_ckpt_1,
            base_model_path=args.base_model,
            vae_model_path=args.vae_model,
            image_encoder_path=args.image_encoder,
            quality_model_path=args.quality_model,
            dataset_info_path=args.dataset_info_path
        )
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return

    normal_images = []
    for root, _, files in os.walk(args.normal_images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                normal_images.append(os.path.join(root, file))

    # if args.num_samples:
    #     normal_images = normal_images[:args.num_samples]

    all_results = []
    for img_path in normal_images:
        try:
            print(f"\nProcessing image {img_path}")
            img = Image.open(img_path).convert("RGB")
            references = generator._get_random_references(args.num_references)

            results = generator.generate_anomaly_image(
                normal_image=img,
                normal_image_path=img_path,
                reference_data_list=references,
                output_root=args.output_base,
                num_inference_steps=args.steps,
                ip_scale=args.ip_scale,
                seed=args.seed,
                strength=args.strength,
                evaluate_quality=args.evaluate,
                num_anomaly_images=args.num_anomaly_images
            )
            all_results.extend(results)
            for result in results:
                print(f"Generated: {result['generated_image']}, Defect: {result['defect_name']}, Dataset: {result['dataset']}")

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue

    summary_path = os.path.join(args.output_base, "summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Processing complete, results saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary JSON: {str(e)}")


if __name__ == "__main__":
    main()