import sys
import os
import random
import torch
import torch.nn as nn
import argparse
import tarfile
import pickle
import io
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoProcessor, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F

from utils.geo_router import get_vlm_for_sequence_regression

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class MultiModalDataCollator:
    def __init__(self, processor, args, image_data_path='xxx/dataset/mp16-pro/mp-16-images.tar', member_info_path='xxx/dataset/mp16-pro/tar_index.pkl'):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = 'left'
        self.image_data_path = image_data_path
        self.args = args
        with open(member_info_path, 'rb') as f:
            self.tar_index = pickle.load(f)

    def __call__(self, examples):
        messages_lis = []
        labels = []
        # for weighted loss
        vlm_distances = []
        retrieval_distances = []

        with tarfile.open(self.image_data_path, 'r') as tar_obj:
            for example in examples:
                image_id = example['img_id']
                image_file = tar_obj.extractfile(self.tar_index[image_id])
                image_data = Image.open(image_file)
                
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                query_image_str = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

                vlm_pred_text = f"VLM Prediction: {example['vlm_prediction']}"
                retrieval_top1_id = example['retrieval_sorted_img_id'][1]
                retrieval_top1_file = tar_obj.extractfile(self.tar_index[retrieval_top1_id])
                ret_image_data = Image.open(retrieval_top1_file)
                ret_buffered = io.BytesIO()
                ret_image_data.save(ret_buffered, format="JPEG")
                ret_image_str = f"data:image/jpeg;base64,{base64.b64encode(ret_buffered.getvalue()).decode('utf-8')}"
                ret_pred_gps = f"Retrieval Prediction: {example['retrieval_prediction']}"

                if args.wo_ctx: 
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                            {"type": "image", "image": query_image_str},
                            {"type": "text", "text": f"{vlm_pred_text}"},
                            {"type": "text", "text": f"{ret_pred_gps}"},
                            {"type": "image", "image": ret_image_str},
                        ],
                    }]
                elif args.wo_retrieval:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                            {"type": "image", "image": query_image_str},
                            {"type": "text", "text": f"{vlm_pred_text}"},
                        ],
                    }]
                elif args.wo_generation:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                            {"type": "image", "image": query_image_str},
                            {"type": "text", "text": f"{ret_pred_gps}"},
                            {"type": "image", "image": ret_image_str},
                            {"type": "text", "text": f"retrieval candidate gps: {example['retrieval_sorted_candidates_gps'][1:]}"}
                        ],
                    }]
                elif args.wo_retrievalANDgeneration:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                            {"type": "image", "image": query_image_str},
                        ],
                    }]
                else:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task: Decide whether to use VLM or Retrieval for this image's geolocalization."},
                            {"type": "image", "image": query_image_str},
                            {"type": "text", "text": f"{vlm_pred_text}"},
                            {"type": "text", "text": f"{ret_pred_gps}"},
                            {"type": "image", "image": ret_image_str},
                            {"type": "text", "text": f"retrieval candidate gps: {example['retrieval_sorted_candidates_gps'][1:]}"}
                        ],
                    }]
                messages_lis.append(messages)
                
                labels.append(int(example['label']))
                vlm_distances.append(float(example["vlm_distance"]))
                retrieval_distances.append(float(example["retrieval_distance"]))

        if 'Qwen2' in self.args.model_path:
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_lis]
            image_inputs, video_inputs = process_vision_info(messages_lis)
            tokenized_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        elif 'Qwen3' in self.args.model_path:
            tokenized_inputs = self.processor.apply_chat_template(
                messages_lis,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )

        tokenized_inputs['labels'] = torch.tensor(labels, dtype=torch.float)

        tokenized_inputs["vlm_distance"] = torch.tensor(vlm_distances, dtype=torch.float)
        tokenized_inputs["retrieval_distance"] = torch.tensor(retrieval_distances, dtype=torch.float)

        return tokenized_inputs

class RouterTrainer(Trainer):
    def __init__(self, model=None, huggingface_args=None, aux_args=None, **kwargs):
        super().__init__(model=model, args=huggingface_args, **kwargs)
        self.aux_args = aux_args

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        _ = inputs.pop("labels", None) 
        d_v = inputs.pop("vlm_distance").float()
        d_r = inputs.pop("retrieval_distance").float()

        logits, outputs = model(**inputs, return_output=True)
        s_logit = logits.view(-1).float()

        eps = 1e-6
        log_diff = torch.log(d_r + eps) - torch.log(d_v + eps)
        
        soft_labels = torch.sigmoid(args.alpha * log_diff)

        loss_per_sample = self.loss_fn(s_logit, soft_labels)
        
        loss = loss_per_sample.mean()

        return (loss, (loss, s_logit)) if return_outputs else loss

class VLMRouterFinetuner(object):
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.model_save_path = args.model_save_path

        self.__model_init__()
        self.__data_init__()
        self.__auto_trainer_init__()

        self.auto_train()

    def __model_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=2*self.args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
            modules_to_save=["value_head"],
        )

        self.model = get_vlm_for_sequence_regression(
            base_model_name_or_path=self.model_path,
            model_name_or_path=self.model_path,
            model_type="reward",
            lora_config=self.lora_config,
            normalize_reward=False,
            use_flash_attention_2=True,
            init_value_head=True,
            value_head_prefix="value_head",
            device_map=None,
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def __data_init__(self):
        full_dataset = load_dataset("parquet", data_files=self.args.dataset)['train']
        
        full_dataset = full_dataset.shuffle(seed=2025)

        if len(full_dataset) > self.args.num_samples:
            full_dataset = full_dataset.select(range(self.args.num_samples))
        
        eval_size = 3000
        if len(full_dataset) <= eval_size:
            eval_size = int(len(full_dataset) * 0.1)
            
        self.eval_set = full_dataset.select(range(eval_size))
        self.train_set = full_dataset.select(range(eval_size, len(full_dataset)))
        
        print(f"Dataset split: Train={len(self.train_set)}, Eval={len(self.eval_set)}")

        self.data_collator = MultiModalDataCollator(
            processor=self.processor, 
            args=self.args, 
            image_data_path=self.args.image_data_path, 
            member_info_path=self.args.member_info_path
        )

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = (logits > 0).astype(int) 
        labels = labels.astype(int)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions)
        }

    def __auto_trainer_init__(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        grad_acc = self.args.total_batch_size // (self.args.per_device_train_batch_size * world_size)

        self.training_args = TrainingArguments(
            output_dir=self.args.model_save_path,
            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=max(1, grad_acc),
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            save_steps=0.02,
            bf16=True,
            # report_to=self.args.report_to,
            report_to=[],
            deepspeed="./utils/zero2.json",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            label_names=["labels"],
            metric_for_best_model="accuracy",
            greater_is_better=True,
            # load_best_model_at_end=True,  # eval
            # eval_strategy="steps",    # eval
            # eval_steps=0.02,    # eval
            # save_total_limit=3,    # eval
            dataloader_num_workers=4,
            dataloader_prefetch_factor=1,
        )

        self.trainer = RouterTrainer(
            model=self.model,
            huggingface_args=self.training_args,
            aux_args=self.args,
            train_dataset=self.train_set,
            # eval_dataset=self.eval_set,    # eval
            data_collator=self.data_collator,
            tokenizer=self.processor.tokenizer,
            compute_metrics=self.compute_metrics,
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=15)]    # eval
        )

    def auto_train(self):
        self.model.train()
        self.trainer.train(resume_from_checkpoint=self.args.continue_ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default='./ckpt/ce/')
    parser.add_argument("--dataset", type=str, default='./dataset/router/router_dataset.parquet')
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--continue_ckpt_path", type=str, default=None)
    parser.add_argument("--image_data_path", type=str, default='./dataset/mp16-pro/mp-16-images.tar')
    parser.add_argument("--member_info_path", type=str, default='./dataset/mp16-pro/metadata/tar_index.pkl')
    parser.add_argument("--alpha", type=float, default=1.6)
    # for ablation
    parser.add_argument("--wo_ctx", type=str2bool, default=False)
    parser.add_argument("--wo_retrieval", type=str2bool, default=False)
    parser.add_argument("--wo_generation", type=str2bool, default=False)
    parser.add_argument("--wo_retrievalANDgeneration", type=str2bool, default=False)
    args = parser.parse_args()

    VLMRouterFinetuner(args)

    # deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune_geo_router.py --model_save_path=./ckpt/