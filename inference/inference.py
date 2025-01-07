import argparse
import sys, os
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from dataset.MatchVision_commentary_new_benchmark_from_npy import MatchVisionCommentary_new_benchmark_from_npy_Dataset
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AdamW
import torch
from torch.nn import DataParallel
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider
import wandb
from utils.score_helper import calculate_metrics_of_set
from optimizer.optimizer_utls import optimizer_commentary_new_benchmark
import csv


# Use CIDEr score to do validation
def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict =dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        gt_captions_dict[i] = [caption]
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()

def inference(args):
    dataset_type = MatchVisionCommentary_new_benchmark_from_npy_Dataset
    commentary_model_type = matchvoice_model_all_blocks
    device_ids = args.device_ids
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    valid_json = []
    valid_video_base_dir = []
    if args.valid_matchtime:
        valid_json.append(args.matchtime_json)
        valid_video_base_dir.append(args.matchtime_video_base)
    if args.valid_soccerreplay:
        valid_json.append(args.soccerreplay1988_json)
        valid_video_base_dir.append(args.soccerreplay1988_video_base)
    
    val_dataset = dataset_type(json_file=valid_json,
                       video_base_dir=valid_video_base_dir)
    
    torch.cuda.init()
    torch.manual_seed(42)
    
    val_data_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, drop_last=False, shuffle=True, pin_memory=True, collate_fn=val_dataset.collater)

    

    print("===== Video features data loaded! =====")
    model = commentary_model_type(num_features=args.num_features, need_temporal=args.need_temporal, open_visual_encoder=args.open_visual_encoder, open_llm_decoder=args.open_llm_decoder)

    model = model.to(devices[0]) 
    model = DataParallel(model, device_ids=device_ids) 
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print("===== Model loaded! =====")

    output_csv_path = args.csv_out_path
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['video_path', 'anonymized', 'temp_res_text'])  # 写入标题行


    val_pbar = tqdm(val_data_loader)
    with torch.no_grad():
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        model = model.to(devices[0]) 
        torch.cuda.empty_cache()
        for samples in val_pbar:
            samples["frames"] = samples["frames"].to(devices[0])
            temp_res_text, anonymized, video_path = model(samples, True)
            with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for res_text, anon, path in zip(temp_res_text, anonymized, video_path):
                    writer.writerow([path, anon, res_text])  # 将每一行写入文件




if __name__ == "__main__":


    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description="Train a model with FRANZ dataset.")
    parser.add_argument("--need_temporal", type=str, default="yes")
    parser.add_argument("--tokenizer_name", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--valid_batch_size", type=int, default=40)
    parser.add_argument("--valid_num_workers", type=int, default=20)

    parser.add_argument("--valid_matchtime", type=bool, default=True)
    parser.add_argument("--valid_soccerreplay", type=bool, default=False)

    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--device_ids", type=int, nargs="+", default=[4])
    parser.add_argument("--open_visual_encoder", type=bool, default=False)
    parser.add_argument("--open_llm_decoder", type=bool, default=False)

    parser.add_argument("--ckpt_path", type=str, default="FILE_PATH_TO_MODEL_CHECKPOINT")
    parser.add_argument("--csv_out_path", type=str, default="inference/sample.csv")

    parser.add_argument("--matchtime_json", type=str, default="train_data/json/MatchTime/classification_test.json")
    parser.add_argument("--matchtime_video_base", type=str, default="FOLDER_OF_VIDEO_CLIPS_OF_MATCHTIME")
    parser.add_argument("--soccerreplay1988_json", type=str, default="train_data/json/SoccerReplay-1988/classification_test.json")
    parser.add_argument("--soccerreplay1988_video_base", type=str, default="FOLDER_OF_VIDEO_CLIPS_OF_SOCCERREPLAY_1988")
    
    args = parser.parse_args()
    inference(args)
