import os
import asyncio
import httpx
import pandas as pd
import numpy as np
import io
import base64
import json
import re
import argparse
from tqdm.asyncio import tqdm
from PIL import Image
from geopy.distance import geodesic

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def encode_image_from_path(image_path):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def extract_coordinates(text):
    try:
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        data = json.loads(clean_text)
        return [float(data.get("latitude", 0)), float(data.get("longitude", 0))]
    except Exception:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
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
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        "temperature": 0.0,
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    for attempt in range(3):
        try:
            response = await client.post(base_url, headers=headers, json=payload, timeout=60.0)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429:
                await asyncio.sleep(2 ** attempt) 
        except Exception:
            if attempt == 2: return '{"latitude": 0.0, "longitude": 0.0}'
    return '{"latitude": 0.0, "longitude": 0.0}'

async def process_row(row, client, args):
    img_path = os.path.join(args.image_dir, str(row['IMG_ID']))
    
    loop = asyncio.get_event_loop()
    base64_img = await loop.run_in_executor(None, encode_image_from_path, img_path)
    
    if base64_img is None:
        return "None", [0.0, 0.0]
    
    raw_output = await call_gemini_api_async(client, base64_img, args.model_name, args.api_key, args.base_url)
    pred_gps = extract_coordinates(raw_output)
    return raw_output, pred_gps

async def main_async(df, args):
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        tasks = []
        for _, row in df.iterrows():
            async def wrapped_task(r=row):
                async with sem:
                    return await process_row(r, client, args)
            tasks.append(wrapped_task())
        return await tqdm.gather(*tasks, desc=f"Querying {args.model_name}")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="xxx")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-nothinking", help="gemini-2.0-flash,gemini-2.5-flash-nothinking,gpt-5-mini,qwen3-vl-8b-instruct,qwen3-vl-32b-instruct")
    parser.add_argument("--dataset_name", type=str, choices=['im2gps3k', 'yfcc4k'], required=True)
    # parser.add_argument("--csv_path", type=str, required=True, help="Path to im2gps3k.csv or yfcc4k.csv")
    # parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()

    if args.dataset_name=='im2gps3k':
        args.csv_path = './dataset/im2gps3k/im2gps3k.csv'
        args.image_dir = './dataset/im2gps3k/im2gps3ktest'
    elif args.dataset_name == 'yfcc4k':
        args.csv_path = './dataset/yfcc4k/yfcc4k.csv'
        args.image_dir = './dataset/yfcc4k/yfcc4k'

    output_path = f"./dataset/router/{args.dataset_name}_{args.model_name}_results.parquet"
    os.makedirs("./dataset/router/", exist_ok=True)

    df_raw = pd.read_csv(args.csv_path)
    if 'gps' not in df_raw.columns:
        df_raw['gps'] = df_raw.apply(lambda x: [x['LAT'], x['LON']], axis=1)

    if os.path.exists(output_path):
        print(f"Loading existing progress from {output_path}")
        df_final = pd.read_parquet(output_path)
        mask_todo = df_final['prediction'].apply(lambda x: x is None or (x[0] == 0.0 and x[1] == 0.0))
        df_todo = df_final[mask_todo].copy()
    else:
        df_final = df_raw.copy()
        df_final['output'] = None
        df_final['prediction'] = None
        df_todo = df_final.copy()

    if len(df_todo) == 0:
        print("Everything is already processed.")
        return

    print(f"Starting API requests for {len(df_todo)} samples...")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main_async(df_todo, args))

    outputs, predictions = zip(*results)
    df_todo['output'] = list(outputs)
    df_todo['prediction'] = list(predictions)
    
    df_final.update(df_todo)

    print("Calculating final distances...")
    def get_dist(row):
        try:
            p = row['prediction']
            if p is None or (p[0] == 0.0 and p[1] == 0.0): return 20000.0
            return geodesic(row['gps'], p).km
        except: return 20000.0

    df_final['distance'] = df_final.apply(get_dist, axis=1)

    df_final.to_parquet(output_path, index=False)
    print(f"Results saved to {output_path}")

    total = len(df_final)
    for lv in [2500, 750, 200, 25, 1]:
        acc = (df_final['distance'] <= lv).sum() / total
        print(f"{lv:>4}km level Accuracy: {acc:>6.2%}")

if __name__ == "__main__":
    run()