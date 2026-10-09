# GeoRouter: Dynamic Paradigm Routing for Worldwide Image Geolocalization

This is the repository for GeoRouter, containing the code, datasets, and checkpoints.

## Environment

```bash
# g3
conda create -n g3 python=3.10
pip install torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.42.0 accelerate huggingface_hub pandas faiss-gpu datasets pyproj

# georanker
# following the original paper to download the env file
mkdir -p ~/conda_path/envs/your_env_name
tar -xzf georanker.tar.gz -C ~/conda_path/envs/your_env_name

# qwen3-VL conda env
conda create -n qwen3vl python=3.11
conda activate qwen3vl
pip install transformers==4.57.6 qwen-vl-utils==0.0.14 pandas geopy scikit-learn deepspeed datasets peft torchvision wandb
pip install flash-attn --no-build-isolation
conda install mpi4py
```

## Dataset

```bash
mkdir dataset
cd dataset
mkdir im2gps3k yfcc4k mp16-pro g3
```

IM2GPS3K: [images](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip) and [metadata](https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv); YFCC4K: [images](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip) and [metadata](https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv).

For the G3 model and MP16-Pro dataset, please refer to their respective original papers for details.

The GeoRouting dataset is available under `./dataset/router/`.

❗ Due to the large size of the dataset, we have split it into multiple parts. Please use the following code to merge them before use.

```python
>>> df = pd.concat([
...     pd.read_parquet('./dataset/router/router_dataset_part1.parquet'),
...     pd.read_parquet('./dataset/router/router_dataset_part2.parquet'),
...     pd.read_parquet('./dataset/router/router_dataset_part3.parquet'),
... ]).reset_index(drop=True)
>>> df.columns
Index(['img_id', 'gps', 'ref_gps', 'ref_img_id', 'ref_texts', 'vlm_output',
       'vlm_prediction', 'vlm_distance', 'retrieval_sorted_candidates_gps',
       'retrieval_sorted_img_id', 'retrieval_sorted_distances',
       'retrieval_prediction', 'retrieval_distance', 'label'],
      dtype='object')
```

### Column Descriptions

| Column | Description |
|--------|-------------|
| `img_id` | Unique identifier of the query image |
| `gps` | Ground-truth GPS coordinates of the query image |
| `ref_gps` | GPS coordinates of the retrieved candidate images |
| `ref_img_id` | Image IDs of the retrieved candidate images |
| `ref_texts` | Textual descriptions of the retrieved candidate images |
| `vlm_output` | Raw output from the generation-based model (VLM) |
| `vlm_prediction` | Predicted GPS coordinates from the generation-based approach |
| `vlm_distance` | Geodesic distance error (km) of the generation-based prediction |
| `retrieval_sorted_candidates_gps` | GPS coordinates of retrieved candidates after ranking |
| `retrieval_sorted_img_id` | Image IDs of retrieved candidates after ranking |
| `retrieval_sorted_distances` | Distance errors of retrieved candidates after ranking |
| `retrieval_prediction` | Predicted GPS coordinates from the retrieval-based approach |
| `retrieval_distance` | Geodesic distance error (km) of the retrieval-based prediction |
| `label` | Binary classification label indicating the routing decision |

## File Structure

```
.
├── ckpt/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── dataset/
│   ├── g3 -> retrieval index, results from g3
│   ├── georanker -> georanker env, checkpoints
│   ├── im2gps3k -> images and metadata for im2gps3k
│   ├── mp16-pro -> images and metadata for mp16-pro
│   ├── router -> main datasets
│   └── yfcc4k -> images and metadata for yfcc4k
├── G3/
│   ├── rff/
│   │   ├── functional.py
│   │   └── layers.py
│   ├── G3.py -> model file for G3
│   └── utils.py
├── GeoRanker/
│   └── georanker.py -> model file for georanker
├── utils/
│   ├── georouter.py -> model file for georouter
│   └── zero2.json -> deepspeed settings
├── evaluate.py
├── finetune_geo_router.py -> script for fine-tuning
├── README.md
├── run_api_im2gps_yfcc.py
├── run_api_router_data.py
├── run_georanker_im2gps_yfcc.py
└── run_georanker_router_data.py
```

## Running Code

1. generate retrieval-based and generation-based predictions

```bash
(g3) python run_api_im2gps_yfcc.py / python run_api_router_data / run_georanker_im2gps_yfcc.py / run_georanker_router_data.py
```

2. training georouter

```bash
(georanker) deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune_geo_router.py --model_save_path=./ckpt/
```

3. evaluate

```bash
(georanker) CUDA_VISIBLE_DEVICES=0 python evaluate.py --router_checkpoint=xxx --parquet_path=xxx --query_image_root=xxx
```

if you want to train georouter with qwen3-vl family, try to run code with qwen3vl env.