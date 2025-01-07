import sys, os
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
import torch
from torch.utils.data import DataLoader, random_split
from model.MatchVision_classifier import MatchVision_Classifier
import torch.optim as optim
from optimizer.optimizer_utls import optimizer_sn_v2_pretrain
from tqdm import tqdm
from torch.nn import DataParallel
import wandb
import importlib.util
import argparse
import random

def topk_accuracy(predictions, targets, topk=(1, 3, 5)):
    batch_size = targets.size(0)
    res = {}
    expanded_targets = targets.view(-1, 1).expand(batch_size, predictions.size(1))
    correct = (predictions == expanded_targets)
    for k in topk:
        correct_k = correct[:, :k].any(dim=1).float().mean().item()
        res[f'top{k}'] = correct_k * 100  # 转换为百分比
    return res

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
    open_siglip = config_training_settings["open_siglip"]
    classifier_transformer_type = config_training_settings["classifier_transformer_type"]
    encoder_type = config_training_settings["encoder_type"]
    use_transformer = config_training_settings["use_transformer"]
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

    ############## Dataset ################
    soccer_video_dataset_type_train = None
    if config_train_dataset["balanced_or_not"] == "balanced":
        soccer_video_dataset_type_train = VideoCaptionDataset_Balanced
    else:
        soccer_video_dataset_type_train = VideoCaptionDataset

    soccer_video_dataset_type_valid = None
    if config_valid_dataset["balanced_or_not"] == "balanced":
        soccer_video_dataset_type_valid = VideoCaptionDataset_Balanced
    else:
        soccer_video_dataset_type_valid = VideoCaptionDataset

    train_dataset = soccer_video_dataset_type_train(
        json_file=config_train_dataset["json"],
        video_base_dir=config_train_dataset["video_base"],
        sample=config_train_dataset["sample"],
        keywords=config_train_dataset["keywords"],
        sample_num=config_train_dataset["sample_num"],
    )

    valid_dataset = soccer_video_dataset_type_valid(
        json_file=config_valid_dataset["json"],
        video_base_dir=config_valid_dataset["video_base"],
        sample=config_valid_dataset["sample"],
        keywords=config_valid_dataset["keywords"],
    )

    ############## Configs ################
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    classifier = MatchVision_Classifier(
        keywords=config_train_dataset["keywords"],
        classifier_transformer_type=classifier_transformer_type,
        vision_encoder_type=encoder_type,
        use_transformer=use_transformer
        ).train()
    
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
        classifier.load_state_dict(new_state_dict)
    
    classifier = classifier.to(devices[0]) 
    classifier = DataParallel(classifier, device_ids=device_ids)

    optimizer = optimizer_sn_v2_pretrain(classifier, classifier_transformer_type, encoder_type, use_transformer, open_siglip)

    if wandb_configs["use_wandb"]:
        wandb.init(project=wandb_configs["project"], entity=wandb_configs["entity"])
        wandb.config = {
            "remark": wandb_configs["remark"]
        }

    ############## Train ################

    for epoch in range(num_epochs):
        if config_train_dataset["balanced_or_not"] == "balanced":
            train_dataset.shuffle_indices()
        train_data_loader = DataLoader(train_dataset, batch_size=config_train_dataset["batch_size"], num_workers=config_train_dataset["num_workers"], drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)
        
        train_losses = []
        train_progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        classifier.train()
        for batch_idx, (frames, caption_tensor) in train_progress_bar:
            frames, caption_tensor = frames.to(devices[0]), caption_tensor.to(devices[0])
            loss, logits = classifier(frames, caption_tensor)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            top_indices = classifier.module.get_types(logits)
            topk_acc = topk_accuracy(top_indices, caption_tensor, topk=(1, 3, 5)) 
            if wandb_configs["use_wandb"]:
                wandb.log({"batch_train_loss": loss.item(),"top1-accuracy": topk_acc['top1'],"top3-accuracy": topk_acc['top3'],"top5-accuracy": topk_acc['top5']})
            train_progress_bar.set_postfix(loss=loss.item(), top1=topk_acc['top1'], top3=topk_acc['top3'], top5=topk_acc['top5'])
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}')

        ############## Validation ################

        if config_valid_dataset["balanced_or_not"] == "balanced":
            valid_dataset.shuffle_indices()
        valid_data_loader = DataLoader(valid_dataset, batch_size=config_valid_dataset["batch_size"], num_workers=config_valid_dataset["num_workers"], drop_last=True, shuffle=False, pin_memory=True, persistent_workers=True)
        top1_accs, top3_accs, top5_accs = [], [], []
        valid_progress_bar = tqdm(enumerate(valid_data_loader), total=len(valid_data_loader), desc=f"Validation Epoch {epoch+1}/{num_epochs}")
        classifier.eval()
        with torch.no_grad():
            for batch_idx, (frames, caption_tensor) in valid_progress_bar:

                frames, caption_tensor = frames.to(devices[0]), caption_tensor.to(devices[0])
                logits = classifier.module.get_logits(frames)
                top_indices = classifier.module.get_types(logits)
                topk_acc = topk_accuracy(top_indices, caption_tensor, topk=(1, 3, 5))  
                top1_accs.append(topk_acc['top1'])
                top3_accs.append(topk_acc['top3'])
                top5_accs.append(topk_acc['top5'])
                
                valid_progress_bar.set_postfix(top1=topk_acc['top1'], top3=topk_acc['top3'], top5=topk_acc['top5'])

        avg_top1_acc = sum(top1_accs) / len(top1_accs)
        avg_top3_acc = sum(top3_accs) / len(top3_accs)
        avg_top5_acc = sum(top5_accs) / len(top5_accs)
        print(f'Validation Accuracies for Epoch {epoch+1}: Top-1: {avg_top1_acc:.2f}%, Top-3: {avg_top3_acc:.2f}%, Top-5: {avg_top5_acc:.2f}%')
    
        if save_check_point:
            if epoch % save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': {k: v.cpu() for k, v in classifier.state_dict().items()},
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(check_point_base_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f'Checkpoint saved at epoch {epoch+1}')

if __name__ == "__main__":
    main()


