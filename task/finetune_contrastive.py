import sys
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from model.MatchVision import VisionTimesformer, TextEncoder
from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch, random, argparse
from transformers import AdamW
from torch.nn import DataParallel
import torch.nn.functional as F
from optimizer.optimizer_utls import optimizer_contrastive
import os
from utils.contrastive import create_label_from_comment, create_label_from_type
from model.MatchVision_contrastive import MatchVision_contrastive_model
import importlib.util
import wandb


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config 

def main():
    ############## Configs ################
    parser = argparse.ArgumentParser(description="Load a Python config file.")
    parser.add_argument('config_path', type=str, help='The path to the Python config file')
    args = parser.parse_args()
    config = load_config(args.config_path)
    # dataset
    config_dataset = config["dataset"]
    config_train_dataset = config_dataset["train"]
    config_valid_dataset = config_dataset["valid"]
    # train setting
    config_training_settings = config["training_settings"]
    num_epochs = config_training_settings["epoch"]
    device_ids = config_training_settings["device_ids"]
    open_text = config_training_settings["open_text"]
    loss_type = config_training_settings["loss_type"]
    encoder_type = config_training_settings["encoder_type"]
    load_checkpoint = config_training_settings["load_checkpoint"]
    checkpoint_path = config_training_settings["checkpoint_path"]
    # logging
    logging_info = config["logs"]
    save_check_point = logging_info["save_check_point"]
    save_every = logging_info["save_every"]
    check_point_base_dir = logging_info["check_point_base_dir"]
    os.makedirs(check_point_base_dir, exist_ok=True)
    wandb_configs = logging_info["wandb_configs"]

    random.seed(42)
    torch.manual_seed(42)

    ############################### DATASET
    
    if config_train_dataset["balanced_or_not"] != "balanced":
        train_DatasetType = VideoCaptionDataset
    else:
        train_DatasetType = VideoCaptionDataset_Balanced
    if config_valid_dataset["balanced_or_not"] != "balanced":
        valid_DatasetType = VideoCaptionDataset
    else:
        valid_DatasetType = VideoCaptionDataset_Balanced
    train_dataset = train_DatasetType(
        json_file=config_train_dataset['json'],
        video_base_dir=config_train_dataset['video_base'],
        sample=config_train_dataset["sample"],
        sample_num=config_train_dataset["sample_num"],
        require_text=True
    )

    valid_dataset = valid_DatasetType(
        json_file=config_valid_dataset['json'],
        video_base_dir=config_valid_dataset['video_base'],
        sample=config_valid_dataset["sample"],
        require_text=True
    )


    train_data_loader = DataLoader(train_dataset, batch_size=config_train_dataset["batch_size"], num_workers=config_train_dataset["num_workers"], drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config_valid_dataset["batch_size"], num_workers=config_valid_dataset["num_workers"], drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)

    print("======== Dataset Loaded ========")

    ##############################

    devices = [torch.device(f'cuda:{i}') for i in device_ids]
    model = MatchVision_contrastive_model(loss_type=loss_type, encoder_type=encoder_type)
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = {key.replace("module.siglip_model.", "visual_encoder."): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(new_state_dict)
    model = model.to(devices[0])  # 将模型首先放到第一个设备上
    model = DataParallel(model, device_ids=device_ids)  # 使用指定的GPU列表

    print("======== Model Loaded ========")
    
    ##############################

    optimizer = optimizer_contrastive(model, open_text=open_text)

    if wandb_configs["use_wandb"]:
        wandb.init(project=wandb_configs["project"], entity=wandb_configs["entity"])
        wandb.config = {
            "remark": wandb_configs["remark"]
        }

    for epoch in range(num_epochs):
        # 训练阶段
        if config_train_dataset["balanced_or_not"] == "balanced":
            train_dataset.shuffle_indices()
        train_losses = []
        train_progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        for batch_idx, (frames, caption_tensor, path, caption_text, comments_text) in train_progress_bar:
            frames = frames.to(devices[0])
            target_label = create_label_from_comment(caption_text).to(devices[0])
            target_label_type = create_label_from_type(caption_text).to(devices[0])
            loss = model(frames, comments_text, target_label)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if wandb_configs["use_wandb"]:
                wandb.log({"batch_train_loss": loss.item()})
            train_losses.append(loss.item())
            train_progress_bar.set_postfix(loss=loss.item())
           
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}')

        top1_accs_com, top3_accs_com, top5_accs_com = [], [], []
        top1_accs_type, top3_accs_type, top5_accs_type = [], [], []
        valid_progress_bar = tqdm(enumerate(valid_data_loader), total=len(valid_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        model.eval()
        with torch.no_grad():
            for batch_idx, (frames, caption_tensor, path, caption_text, comments_text) in valid_progress_bar:
                frames = frames.to(devices[0])
                target_label = create_label_from_comment(caption_text).to(devices[0])
                target_label_type = create_label_from_type(caption_text).to(devices[0])
                similarity_matrix = model.module.sim_mat(frames, comments_text)
                accuracy_comment, accuracy_type = model.module.calculate_top_k_accuracy(similarity_matrix, target_label, target_label_type)
                valid_progress_bar.set_postfix(com=f"{accuracy_comment[0]}|{accuracy_comment[1]}|{accuracy_comment[2]}", type=f"{accuracy_type[0]}|{accuracy_type[1]}|{accuracy_type[2]}")

                top1_accs_com.append(accuracy_comment[0])
                top3_accs_com.append(accuracy_comment[1])
                top5_accs_com.append(accuracy_comment[2])
                top1_accs_type.append(accuracy_type[0])
                top3_accs_type.append(accuracy_type[1])
                top5_accs_type.append(accuracy_type[2])

        avg_top1_accs_com = sum(top1_accs_com) / len(top1_accs_com) * 100
        avg_top3_accs_com = sum(top3_accs_com) / len(top3_accs_com) * 100
        avg_top5_accs_com = sum(top5_accs_com) / len(top5_accs_com) * 100
        avg_top1_accs_type = sum(top1_accs_type) / len(top1_accs_type) * 100
        avg_top3_accs_type = sum(top3_accs_type) / len(top3_accs_type) * 100
        avg_top5_accs_type = sum(top5_accs_type) / len(top5_accs_type) * 100
        print(f'Validation Commentary Accuracies for Epoch {epoch+1}: Top-1: {avg_top1_accs_com:.2f}%, Top-3: {avg_top3_accs_com:.2f}%, Top-5: {avg_top5_accs_com:.2f}%')
        print(f'Validation Type Accuracies for Epoch {epoch+1}: Top-1: {avg_top1_accs_type:.2f}%, Top-3: {avg_top3_accs_type:.2f}%, Top-5: {avg_top5_accs_type:.2f}%')
        
        if save_check_point:
            if epoch % save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': {k: v.cpu() for k, v in model.module.visual_encoder.state_dict().items()},
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(check_point_base_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f'Checkpoint saved at epoch {epoch+1}')

if __name__ == "__main__":
    main()