import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
import numpy as np
import io
import base64
from PIL import Image
import tarfile
import pickle

from GeoRanker.georanker import get_vlm_for_sequence_regression

def evaluate(args):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["value_head"],
    )

    model = get_vlm_for_sequence_regression(
        base_model_name_or_path=args.model,
        model_name_or_path=args.model,
        model_type="reward",
        lora_config=lora_config,
        bf16=True,
        use_flash_attention_2=True,
        device_map="auto",
        lora_path=args.model_path,
    )
    
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    if args.dataset == 'im2gps3k':
        data = pd.read_csv('./dataset/im2gps3k/im2gps3k.csv')
    elif args.dataset == 'yfcc4k':
        data = pd.read_csv('./dataset/yfcc4k/yfcc4k.csv')
    else:
        data = pd.read_csv(args.dataset)

    with open(args.tar_index_path, 'rb') as f:
        tar_index = pickle.load(f)
    tar_obj = tarfile.open(args.image_data_path)

    messages = []
    ref_gps_all = []
    ref_ids_all = []

    for i in tqdm(range(data.shape[0]), desc='Processing messages'):
        sample_messages = []
        sample_ref_gps = []
        sample_ref_ids = []

        candidate_gps_lis = np.array(eval(data.loc[i, 'candidate_gps_lis']))[:args.topn]
        candidate_img_lis = np.array(eval(data.loc[i, 'candidate_img_lis']))[:args.topn]
        candidate_text_lis = np.array(eval(data.loc[i, 'candidate_text_lis']))[:args.topn]
        zs_gps_lis = np.array(eval(data.loc[i, 'zs_gps_lis']))[:args.topn_zs]

        main_image_str = f"file://{args.image_path}{data.iloc[i]['IMG_ID']}"

        neg_gps_raw = np.array(eval(data.loc[i, 'candidate_gps_lis']))[-5:]
        neg_texts_raw = np.array(eval(data.loc[i, 'candidate_text_lis']))[-5:]
        neg_samples = []
        for ng, nt in zip(neg_gps_raw, neg_texts_raw):
            neg_samples.append(f"latitude: {ng[0]}, longitude: {ng[1]}, {' '.join([str(x) for x in nt if x])}")

        for j in range(len(candidate_gps_lis)):
            gps = candidate_gps_lis[j]
            ref_id = candidate_img_lis[j]
            ref_text = ' '.join([str(elem) for elem in candidate_text_lis[j] if elem])
            
            ref_image = tar_obj.extractfile(tar_index[ref_id])
            ref_image_data = Image.open(ref_image)
            buffered = io.BytesIO()
            ref_image_data.save(buffered, format="JPEG")
            ref_image_str = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

            msg = [{"role": "user", "content": [
                {"type": "image", "image": main_image_str},
                {"type": "text", "text": f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}, {ref_text}?"},
                {"type": "image", "image": ref_image_str},
                {"type": "text", "text": f"Negative examples: {'; '.join(neg_samples)}"},
            ]}]
            sample_messages.append(msg)
            sample_ref_gps.append(gps)
            sample_ref_ids.append(ref_id)

        for k in range(len(zs_gps_lis)):
            gps = zs_gps_lis[k]
            msg = [{"role": "user", "content": [
                {"type": "image", "image": main_image_str},
                {"type": "text", "text": f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}?"},
                {"type": "text", "text": f"Negative examples: {'; '.join(neg_samples)}"},
            ]}]
            sample_messages.append(msg)
            sample_ref_gps.append(gps)
            sample_ref_ids.append(f"zs_{k}")

        messages.extend(sample_messages)
        ref_gps_all.append(sample_ref_gps)
        ref_ids_all.append(sample_ref_ids)

    all_rewards = []
    BSZ = args.batch_size
    for i in tqdm(range(0, len(messages), BSZ), desc='Computing rewards'):
        batch_msgs = messages[i:i + BSZ]
        text_prompt = processor.apply_chat_template(batch_msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        
        inputs = processor(
            text=text_prompt, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            rewards, _ = model(**inputs, return_output=True)
            all_rewards.extend(rewards.cpu().float().tolist())

    final_results = []
    for i in tqdm(range(data.shape[0]), desc='Formatting output'):
        num_cands = len(ref_gps_all[i])
        start_idx = i * (args.topn + args.topn_zs)
        start_ptr = sum([len(x) for x in ref_gps_all[:i]])
        end_ptr = start_ptr + num_cands
        
        sample_rewards = np.array(all_rewards[start_ptr:end_ptr])
        sample_gps = np.array(ref_gps_all[i])
        sample_ids = np.array(ref_ids_all[i])
        
        sort_idx = np.argsort(-sample_rewards)
        
        sorted_gps = sample_gps[sort_idx].tolist()
        sorted_ids = sample_ids[sort_idx].tolist()
        
        gt_gps = (data.loc[i, 'LAT'], data.loc[i, 'LON'])
        pred_gps = sorted_gps[0]
        pred_dist = geodesic(gt_gps, pred_gps).km
        
        sorted_distances = [geodesic(gt_gps, g).km for g in sorted_gps]

        res_row = data.iloc[i].to_dict()
        res_row.update({
            'retrieval_sorted_candidates_gps': sorted_gps,
            'retrieval_sorted_img_id': sorted_ids,
            'retrieval_sorted_distances': sorted_distances,
            'retrieval_prediction': pred_gps,
            'retrieval_distance': pred_dist,
            'max_reward': float(sample_rewards[sort_idx[0]])
        })
        final_results.append(res_row)

    output_df = pd.DataFrame(final_results)
    os.makedirs('./dataset/router/', exist_ok=True)
    save_path = f'./dataset/router/{args.dataset}_georanker_results.parquet'
    output_df.to_parquet(save_path, index=False)
    
    print(f"\nResults saved to {save_path}")
    for lv in [2500, 750, 200, 25, 1]:
        acc = (output_df['retrieval_distance'] < lv).mean()
        print(f'{lv}km level Accuracy: {acc:.2%}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--model_path', type=str, required=True, help='LoRA path')
    parser.add_argument('--dataset', type=str, default='im2gps3k')
    parser.add_argument('--image_path', type=str, default='dataset/im2gps3k/im2gps3ktest/')
    parser.add_argument('--image_data_path', type=str, default='./dataset/mp16-pro/mp-16-images.tar')
    parser.add_argument('--tar_index_path', type=str, default='./dataset/mp16-pro/metadata/tar_index.pkl')
    parser.add_argument('--topn', type=int, default=10)
    parser.add_argument('--topn_zs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    evaluate(args)