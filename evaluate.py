import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import tarfile
import pickle
import io
import base64
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from peft import PeftModel, LoraConfig
from qwen_vl_utils import process_vision_info

from utils.geo_router import get_vlm_for_sequence_regression

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings
warnings.filterwarnings('ignore')

class RouterTestDataset(Dataset):
    def __init__(self, parquet_path):
        self.data = pd.read_parquet(parquet_path)
        if isinstance(self.data['retrieval_sorted_candidates_gps'].iloc[0], str):
             self.data['retrieval_sorted_candidates_gps'] = self.data['retrieval_sorted_candidates_gps'].apply(eval)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]

class HybridDataCollator:
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.query_root = args.query_image_root
        
        print(f"Loading Tar Index from {args.tar_index_path}...")
        with open(args.tar_index_path, 'rb') as f:
            self.tar_index = pickle.load(f)

        self.tar_path = args.gallery_image_tar

    def __call__(self, batch_rows):
        messages_lis = []
        meta_data = []

        with tarfile.open(self.tar_path, 'r') as tar_obj:
            for row in batch_rows:
                img_name = str(row['IMG_ID'])
                query_path = os.path.join(self.query_root, img_name)
                
                try:
                    q_img = Image.open(query_path).convert('RGB')
                    buffered = io.BytesIO()
                    q_img.save(buffered, format="JPEG")
                    q_img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    q_img_str = f"data:image/jpeg;base64,{q_img_b64}"
                except Exception as e:
                    print(f"[Warning] Failed to load query image: {query_path}, skipping prompt img.")
                    q_img_str = None

                ref_id_list = row['retrieval_sorted_img_id']
                if isinstance(ref_id_list, np.ndarray): ref_id_list = ref_id_list.tolist()
                
                ref_id = ref_id_list[0]

                try:
                    member = self.tar_index[ref_id]
                    f = tar_obj.extractfile(member)
                    r_img = Image.open(f).convert('RGB')
                    r_buffered = io.BytesIO()
                    r_img.save(r_buffered, format="JPEG")
                    r_img_b64 = base64.b64encode(r_buffered.getvalue()).decode('utf-8')
                    r_img_str = f"data:image/jpeg;base64,{r_img_b64}"
                except Exception as e:
                    print(f"[Warning] Failed to extract ref image: {ref_id}")
                    r_img_str = None

                vlm_pred_text = f"VLM Prediction: {row['vlm_prediction']}"
                ret_pred_gps = f"Retrieval Prediction: {row['retrieval_prediction']}"
                
                cands_gps = row['retrieval_sorted_candidates_gps']
                if isinstance(cands_gps, np.ndarray): cands_gps = cands_gps.tolist()
                cand_gps_str = str(cands_gps[1:]) if len(cands_gps) > 1 else str(cands_gps)

                content = [
                    {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                ]
                if q_img_str:
                    content.append({"type": "image", "image": q_img_str})
                
                content.extend([
                    {"type": "text", "text": f"{vlm_pred_text}"},
                    {"type": "text", "text": f"{ret_pred_gps}"},
                ])

                if r_img_str:
                    content.append({"type": "image", "image": r_img_str})

                content.append({"type": "text", "text": f"retrieval candidate gps: {cand_gps_str}"})
                
                messages = [{"role": "user", "content": content}]
                messages_lis.append(messages)
                
                meta_data.append({
                    'vlm_distance': row['vlm_distance'],
                    'retrieval_distance': row['retrieval_distance']
                })

        if 'Qwen2' in self.args.base_model:
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_lis]
            image_inputs, video_inputs = process_vision_info(messages_lis)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        elif 'Qwen3' in self.args.base_model:
            inputs = self.processor.apply_chat_template(
                messages_lis,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )

        return inputs, meta_data

def evaluate(args):
    print("Loading model...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
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
        device_map="auto",
        lora_path=args.router_checkpoint,
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)
    processor.tokenizer.padding_side = "left"
    
    dataset = RouterTestDataset(args.parquet_path)
    collator = HybridDataCollator(processor, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=4, shuffle=False)
    
    all_logits = []
    all_meta = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for inputs, meta in tqdm(dataloader):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            logits, _ = model(**inputs, return_output=True)
            
            if logits.ndim == 2 and logits.shape[1] == 1:
                scores = logits.view(-1)
            elif logits.ndim == 2 and logits.shape[1] == 2:
                scores = logits[:, 1] - logits[:, 0]
            else:
                scores = logits.view(-1)
                
            all_logits.extend(scores.cpu().float().numpy().tolist())
            all_meta.extend(meta)
            
    df_res = pd.DataFrame(all_meta)
    df_res['score'] = all_logits
    
    thresholds = [-1.0, -0.5,-0.45,-0.4,-0.35,-0.3,-0.25, -0.2,-0.15,-0.1,-0.05, 0.0, 0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.4,0.45, 0.5, 1.0]
    if args.manual_threshold is not None:
        thresholds = [args.manual_threshold]
        
    levels = [1, 25, 200, 750, 2500]
    
    print("\n" + "="*80)
    print(f"EVALUATION REPORT | Total Samples: {len(df_res)}")
    print("="*80)
    
    # --- Baseline Metrics ---
    def calc_acc(dists, lv):
        return (dists <= lv).mean() * 100
    
    def get_metrics_line(dists):
        accs = [calc_acc(dists, lv) for lv in levels]
        mean_acc = np.mean(accs)
        acc_strs = [f"{a:.2f}" for a in accs]
        return " | ".join(acc_strs) + f" | {mean_acc:.2f}"

    header_levels = " | ".join([f"@{lv}km" for lv in levels])
    print(f"{'Method':<15} | {header_levels} | Mean")
    print("-" * 80)
    

    print(f"{'Pure Ret':<15} | " + get_metrics_line(df_res['retrieval_distance']))
    
    print(f"{'Pure VLM':<15} | " + get_metrics_line(df_res['vlm_distance']))
    
    oracle_dists = np.minimum(df_res['vlm_distance'], df_res['retrieval_distance'])
    print(f"{'Oracle':<15} | " + get_metrics_line(oracle_dists))

    print("-" * 80)
    
    for th in thresholds:
        routed_dists = np.where(
            df_res['score'] > th, 
            df_res['vlm_distance'],       # Choice: VLM
            df_res['retrieval_distance']  # Choice: Retrieval
        )
        
        vlm_ratio = (df_res['score'] > th).mean() * 100
        
        metrics_str = get_metrics_line(routed_dists)
        
        print(f"Router(th={th:<4}) | {metrics_str} | VLM%: {vlm_ratio:.1f}%")

    # output predictions
    # df_res.to_parquet('./routing_accuracy_analysis/yfcc4k_predictions.parquet')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--router_checkpoint", type=str, required=True, help="Path to your trained LoRA folder")
    
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the test parquet file")
    parser.add_argument("--query_image_root", type=str, required=True, help="Root folder for local query images")
    parser.add_argument("--gallery_image_tar", type=str, default="./dataset/mp16-pro/mp-16-images.tar", help="Path to MP16 tar file")
    parser.add_argument("--tar_index_path", type=str, default="./dataset/mp16-pro/metadata/tar_index.pkl", help="Path to MP16 tar index")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--manual_threshold", type=float, default=None, help="If set, only test this single threshold")
    
    args = parser.parse_args()
    
    evaluate(args)

    # must run on one gpu
    # CUDA_VISIBLE_DEVICES=0 python evaluate.py --router_checkpoint= --parquet_path= --query_image_root=