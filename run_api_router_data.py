import os
import asyncio
import httpx
import pandas as pd
import numpy as np
import tarfile
import pickle
import io
import base64
import json
import re
import argparse
from tqdm.asyncio import tqdm
from PIL import Image
from geopy.distance import geodesic

def encode_image_to_base64(image_bytes):
    if image_bytes is None:
        return None
    image = Image.open(image_bytes)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_from_tar_sync(tar_path, tar_index, img_id):
    try:
        with tarfile.open(tar_path, 'r') as tar_obj:
            member = tar_index[img_id]
            f = tar_obj.extractfile(member)
            return io.BytesIO(f.read())
    except Exception:
        return None

def extract_coordinates(text):
    try:
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        data = json.loads(clean_text)
        return [float(data.get("latitude", 0)), float(data.get("longitude", 0))]
    except Exception:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        # print(text)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
    return [0.0, 0.0]

async def call_gemini_api_async(client, base64_image, model_name, api_key, base_url):
    prompt = """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
    Please give me the location of the given image.
    Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
    Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "temperature": 0.0,
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    for attempt in range(3):
        try:
            response = await client.post(
                base_url, 
                headers=headers, 
                json=payload, 
                timeout=30.0
            )
            if response.status_code == 200:
                res_json = response.json()
                return res_json['choices'][0]['message']['content']
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            if attempt == 2:
                print(f"Request failed after 3 attempts: {e}")
    
    return '{"latitude": 0.0, "longitude": 0.0}'

async def process_row(row, client, args, tar_index):
    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(
        None, get_image_from_tar_sync, args.image_data_path, tar_index, row['img_id']
    )
    
    if img_bytes is None:
        print('img_bytes is None')
        return "None", [0.0, 0.0]
    
    base64_img = encode_image_to_base64(img_bytes)
    raw_output = await call_gemini_api_async(client, base64_img, args.model_name, args.api_key, args.base_url)
    pred_gps = extract_coordinates(raw_output)
    
    return raw_output, pred_gps

async def main_async(df, args, tar_index):
    sem = asyncio.Semaphore(args.concurrency)
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for _, row in df.iterrows():
            async def wrapped_task(r=row):
                async with sem:
                    return await process_row(r, client, args, tar_index)
            tasks.append(wrapped_task())
        
        results = await tqdm.gather(*tasks, desc=f"Processing {args.model_name}")
        return results

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="xxx")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-nothinking")
    
    parser.add_argument("--parquet_path", type=str, default="./dataset/router/10w_data_for_router.parquet")
    parser.add_argument("--image_data_path", type=str, default='./dataset/mp16-pro/mp-16-images.tar')
    parser.add_argument("--member_info_path", type=str, default='./dataset/mp16-pro/metadata/tar_index.pkl')
    
    parser.add_argument("--start_idx", type=int, default=0, help="Start row index")
    parser.add_argument("--end_idx", type=int, default=100, help="End row index")
    parser.add_argument("--concurrency", type=int, default=100, help="Max concurrent requests")
    
    args = parser.parse_args()

    output_filename = f"./dataset/router/{args.model_name}_results_{args.start_idx}_{args.end_idx}.parquet"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    df_existing = None
    if os.path.exists(output_filename):
        print(f"Found existing result file: {output_filename}. Checking for missing predictions...")
        df_existing = pd.read_parquet(output_filename)
        
        mask_missing = df_existing['prediction'].apply(lambda x: x[0] == 0.0 and x[1] == 0.0)
        df_todo = df_existing[mask_missing].copy()
        
        if len(df_todo) == 0:
            print("All rows in existing file are already processed. Exiting.")
            return
        print(f"Total rows to re-process: {len(df_todo)}")
    else:
        print(f"No existing file. Loading data from {args.start_idx} to {args.end_idx}...")
        df_full = pd.read_parquet(args.parquet_path)
        df_todo = df_full.iloc[args.start_idx : args.end_idx].copy()
        df_todo['output'] = None
        df_todo['prediction'] = None

    with open(args.member_info_path, 'rb') as f:
        tar_index = pickle.load(f)
    print("Loading success!")

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main_async(df_todo, args, tar_index))

    outputs, predictions = zip(*results)
    df_todo['output'] = list(outputs)
    df_todo['prediction'] = list(predictions)
    print(df_todo)

    if df_existing is not None:
        df_existing.update(df_todo)
        df_final = df_existing
    else:
        df_final = df_todo

    print("Calculating distances...")
    def calc_dist(row):
        try:
            if row['prediction'][0] == 0.0 and row['prediction'][1] == 0.0:
                return 20000.0
            return geodesic(row['gps'], row['prediction']).km
        except Exception:
            return 20000.0
    
    df_final['distance'] = df_final.apply(calc_dist, axis=1)

    df_final.to_parquet(output_filename)
    print(f"Saved updated results to {output_filename}")

    total = len(df_final)
    levels = [2500, 750, 200, 25, 1]
    print("\n" + "="*30)
    print(f"Updated Results for {args.model_name}")
    for lv in levels:
        acc = (df_final['distance'] <= lv).sum() / total * 100
        print(f"{lv:>4}km level accuracy: {acc:>6.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()