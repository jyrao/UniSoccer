import argparse
import sys, os
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from dataset.MatchVision_commentary_from_npy import MatchVisionCommentary_from_npy_Dataset
from model.matchvoice_model import matchvoice_model
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AdamW
import torch
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider
import wandb
from utils.score_helper import calculate_metrics_of_set


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

def train(args):
    dataset_type = MatchVisionCommentary_from_npy_Dataset
    commentary_model_type = matchvoice_model
    if args.use_wandb:
        wandb.init(project=args.wandb_name, entity="jy_rao")
        wandb.config = {
            "remark": args.wandb_name
        }

    train_json = []
    valid_json = []
    train_video_base_dir = []
    valid_video_base_dir = []
    train_npy_dir = []
    valid_npy_dir = []

    if args.train_matchtime:
        train_json.append(args.matchtime_train_json)
        train_video_base_dir.append(args.matchtime_video_folder)
        valid_json.append(args.matchtime_valid_json)
        valid_video_base_dir.append(args.matchtime_video_folder)
    if args.train_soccerreplay:
        train_json.append(args.soccerreplay1988_train_json)
        train_video_base_dir.append(args.soccerreplay1988_video_folder)
        valid_json.append(args.soccerreplay1988_valid_json)
        valid_video_base_dir.append(args.soccerreplay1988_video_folder)

    if args.train_matchtime:
        if args.train_soccerreplay:
            word_world_file = "/remote-home/jiayuanrao/MatchTime/words_world/merge.pkl"
        else:
            word_world_file = "/remote-home/jiayuanrao/MatchTime/words_world/match_time.pkl"
    else:
        word_world_file = "/remote-home/jiayuanrao/MatchTime/words_world/franz.pkl"


    train_dataset = dataset_type(json_file=train_json,
                       video_base_dir=train_video_base_dir,
                       npy_dir=train_npy_dir)
    val_dataset = dataset_type(json_file=valid_json,
                       video_base_dir=valid_video_base_dir,
                       npy_dir=valid_npy_dir)
    torch.cuda.init()
    torch.manual_seed(42)

    
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, drop_last=True, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, drop_last=False, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)

    print("===== Video features data loaded! =====")
    model = commentary_model_type(device=args.device, num_features=args.num_features, need_temporal=args.need_temporal, file_path=word_world_file)
    if args.continue_train:
        checkpoint = torch.load(args.load_ckpt)
        current_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in checkpoint.items() if k in current_state_dict and v.size() == current_state_dict[k].size()}
        current_state_dict.update(filtered_state_dict)
        model.load_state_dict(current_state_dict)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.model_output_dir, exist_ok=True)
    print("===== Model and Checkpoints loaded! =====")
    
    max_val_CIDEr = max(float(0), args.pre_max_CIDEr)
    for epoch in range(args.pre_epoch, args.num_epoch):
        model.train()
        train_loss_accum = 0.0
        train_pbar = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{args.num_epoch} Training')
        for samples in train_pbar:

            optimizer.zero_grad()
            try:
                loss = model(samples)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                avg_train_loss = train_loss_accum / len(train_data_loader)
                if args.use_wandb:
                    wandb.log({"train_loss": loss.item()})
            except:
                pass

        model.eval()
        val_CIDEr = 0.0

        reference_list = []
        hypothesis_list = []

        val_pbar = tqdm(val_data_loader, desc=f'Epoch {epoch+1}/{args.num_epoch} Validation')
        with torch.no_grad():
            for samples in val_pbar:
                temp_res_text, anonymized = model(samples, True)
                cur_CIDEr_score = eval_cider(temp_res_text, anonymized)
                val_CIDEr += sum(cur_CIDEr_score)/len(cur_CIDEr_score)
                val_pbar.set_postfix({"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})
                
                reference_list.extend(anonymized)
                hypothesis_list.extend(temp_res_text)

        avg_val_CIDEr = val_CIDEr / len(val_data_loader)

        reference_dict = {i: [s] for i, s in enumerate(reference_list)}
        hypothesis_dict = {i: [s] for i, s in enumerate(hypothesis_list)}
        scores_this_epoch = calculate_metrics_of_set(reference_dict, hypothesis_dict)
        print(f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation scores: B1:{scores_this_epoch['BLEU-1']:.3f}|B4:{scores_this_epoch['BLEU-4']:.3f}|M:{scores_this_epoch['METEOR']:.3f}|R:{scores_this_epoch['ROUGE-L']:.3f}|C:{scores_this_epoch['CIDER']:.3f}")
        
        if args.model_save and epoch % args.model_save_every == 0:
            file_path = f"{args.model_output_dir}/model_save_{epoch+1}.pth"
            save_matchvoice_model(model, optimizer, file_path)

        if avg_val_CIDEr > max_val_CIDEr:
            max_val_CIDEr = avg_val_CIDEr
            file_path = f"{args.model_output_dir}/model_save_best_val_CIDEr.pth"
            save_matchvoice_model(model, optimizer, file_path)

def save_matchvoice_model(model, optimizer, file_path):
    device = model.device
    state_dict = model.cpu().state_dict()
    state_dict_without_llama = {}
    for key, value in state_dict.items():
        if "llama_model.model.layers" not in key:
            state_dict_without_llama[key] = value
    torch.save(state_dict_without_llama, file_path)
    model.to(device)
    # print("C:", model.device)
    for state in optimizer.state.values():
        if 'exp_avg' in state:
            state['exp_avg'] = state['exp_avg'].to(device)
        if 'exp_avg_sq' in state:
            state['exp_avg_sq'] = state['exp_avg_sq'].to(device)

if __name__ == "__main__":


    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description="Train a model with soccer dataset.")
    parser.add_argument("--npy_dir", type=str, default="FOLDER_FOR_NPY_FILES")
    parser.add_argument("--from_matchvision", type=bool, default=True)
    parser.add_argument("--load_from_npy", type=bool, default=True)
    parser.add_argument("--need_temporal", type=str, default="yes")
    parser.add_argument("--tokenizer_name", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--train_num_workers", type=int, default=20)

    parser.add_argument("--train_matchtime", type=bool, default=True)
    parser.add_argument("--train_soccerreplay", type=bool, default=False)
    # parser.add_argument("--valid_matchtime", type=bool, default=True)
    # parser.add_argument("--valid_soccerreplay", type=bool, default=True)

    parser.add_argument("--matchtime_train_json", type=str, default="./train_data/json/MatchTime/classification_train.json")
    parser.add_argument("--matchtime_valid_json", type=str, default="./train_data/json/MatchTime/classification_valid.json")
    parser.add_argument("--matchtime_video_folder", type=str, default="FOLDER_OF_VIDEO_CLIPS_OF_MATCHTIME")
    parser.add_argument("--soccerreplay1988_train_json", type=str, default="./train_data/json/SoccerReplay-1988/classification_train.json")
    parser.add_argument("--soccerreplay1988_valid_json", type=str, default="./train_data/json/SoccerReplay-1988/classification_valid.json")
    parser.add_argument("--soccerreplay1988_video_folder", type=str, default="FOLDER_OF_VIDEO_CLIPS_OF_SOCCERREPLAY_1988")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--model_save", type=bool, default=True)
    parser.add_argument("--model_save_every", type=int, default=1)
    parser.add_argument("--model_output_dir", type=str, default="FOLDER_TO_SAVE_MODELS")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_name", type=str, default="commentary_siglip")

    # If continue training from any epoch
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
    parser.add_argument("--pre_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str, default="TO_CONTINUE_TRAINING_FROM_THIS_CHECKPOINT")

    args = parser.parse_args()
    train(args)
