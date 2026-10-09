import os
import torch
import argparse
import pandas as pd
import numpy as np
import tarfile
import pickle
import io
import base64
from tqdm import tqdm
from PIL import Image
from geopy.distance import geodesic
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from GeoRanker.georanker import get_vlm_for_sequence_regression
import functools
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def encode_image_to_base64(image_bytes):
    image = Image.open(image_bytes)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def get_image_from_tar(tar_obj, tar_index, img_id):
    try:
        member = tar_index[img_id]
        f = tar_obj.extractfile(member)
        return f
    except KeyError:
        return None

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./dataset/georanker/checkpoints")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--parquet_path", type=str, default="./dataset/router/10w_data_for_router.parquet")
    parser.add_argument("--image_data_path", type=str, default='./dataset/mp16-pro/mp-16-images.tar')
    parser.add_argument("--member_info_path", type=str, default='./dataset/mp16-pro/metadata/tar_index.pkl')
    parser.add_argument("--output_path", type=str, default="./dataset/router/georanker_results.parquet")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--topn", type=int, default=10, help="Number of candidates to rank per sample")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--device_map", type=str, default="cuda:0")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)

    print(f"Loading model from {args.model_path}...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["value_head"],
    )

    model = get_vlm_for_sequence_regression(
        base_model_name_or_path=args.base_model,
        model_name_or_path=args.base_model,
        model_type="reward",
        lora_config=lora_config,
        bf16=True,
        use_flash_attention_2=True,
        device_map=args.device_map,
        lora_path=args.model_path,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)
    processor.tokenizer.padding_side = 'left'

    with open(args.member_info_path, 'rb') as f:
        tar_index = pickle.load(f)
    tar_obj = tarfile.open(args.image_data_path, 'r')

    all_messages = []
    print("Pre-processing messages...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_img_b64 = encode_image_to_base64(get_image_from_tar(tar_obj, tar_index, row['img_id']))

        cand_gps = row['ref_gps'][:args.topn]
        cand_ids = row['ref_img_id'][:args.topn]
        cand_texts = row['ref_texts'][:args.topn]
        
        neg_gps_list = row['ref_gps'][-5:]
        neg_texts_list = row['ref_texts'][-5:]
        neg_prompt = "; ".join([f"latitude: {ng[0]}, longitude: {ng[1]}, {' '.join(nt)}" for ng, nt in zip(neg_gps_list, neg_texts_list)])

        for i in range(args.topn):
            ref_img_bytes = get_image_from_tar(tar_obj, tar_index, cand_ids[i])
            ref_img_b64 = encode_image_to_base64(ref_img_bytes)
            
            msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img_b64},
                    {"type": "text", "text": f"How far is this place from latitude: {cand_gps[i][0]}, longitude: {cand_gps[i][1]}, {' '.join(cand_texts[i])}?"},
                    {"type": "image", "image": ref_img_b64},
                    {"type": "text", "text": f"Negative examples: {neg_prompt}"},
                ],
            }]
            all_messages.append(msg)

    all_rewards = []
    print(f"Starting inference with total {len(all_messages)} candidates...")
    for i in tqdm(range(0, len(all_messages), args.batch_size), desc="Computing rewards"):
        batch_msgs = all_messages[i : i + args.batch_size]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(args.device_map)

        with torch.no_grad():
            rewards, _ = model(**inputs, return_output=True)
            all_rewards.extend(rewards.cpu().float().numpy().flatten().tolist())

    print("Post-processing rankings...")
    all_rewards = np.array(all_rewards).reshape(-1, args.topn) # (num_samples, topn)
    
    final_results = []
    for idx in tqdm(range(len(df)), desc="Sorting"):
        sample_rewards = all_rewards[idx]
        row = df.iloc[idx]
        
        cand_gps = row['ref_gps'][:args.topn]
        cand_ids = row['ref_img_id'][:args.topn]
        gt_gps = row['gps']
        sorted_indices = np.argsort(sample_rewards)[::-1]
        
        res_gps = [cand_gps[i] for i in sorted_indices]
        res_ids = [cand_ids[i] for i in sorted_indices]
        res_dists = [geodesic(gt_gps, g).km for g in res_gps]

        final_results.append({
            "sorted_candidates_gps": res_gps,
            "sorted_img_id": res_ids,
            "sorted_distances": res_dists,
            "top1_gps": res_gps[1],
            "top1_distance": res_dists[1]
        })

    results_df = pd.DataFrame(final_results)
    output_df = pd.concat([df, results_df], axis=1)
    output_df.to_parquet(args.output_path)
    print(f"Complete. Saved to {args.output_path}")

    total_samples = len(output_df)
    print("\n" + "="*30)
    print(f"Evaluation Results (Total: {total_samples})")
    print("-" * 30)
    
    levels = [2500, 750, 200, 25, 1]
    for level in levels:
        count = (output_df['top1_distance'] <= level).sum()
        acc = (count / total_samples) * 100
        print(f"{level:>4}km level accuracy: {acc:>6.2f}% ({count}/{total_samples})")
    print("="*30)

if __name__ == "__main__":
    evaluate()