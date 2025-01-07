import torch

def create_label_from_comment(caption_text, special_categories = {
            "end of half game", "off side", "start of half game",
            "ball possession", "substitution"
        }):
    N = len(caption_text)
    tensor = torch.eye(N) * 2 - 1
    for i in range(N):
        for j in range(i + 1, N):
            if caption_text[i] == caption_text[j] and caption_text[i] in special_categories:
                tensor[i, j] = 1
                tensor[j, i] = 1
    return tensor

def create_label_from_type(caption_text):
    N = len(caption_text)
    tensor = torch.eye(N) * 2 - 1
    for i in range(N):
        for j in range(i + 1, N):
            if caption_text[i] == caption_text[j]:
                tensor[i, j] = 1
                tensor[j, i] = 1
    return tensor
