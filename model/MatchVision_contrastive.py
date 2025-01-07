import sys
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from model.MatchVision import VisionTimesformer, TextEncoder
from torch import nn
import torch.nn.functional as F
import torch

class MatchVision_contrastive_model(nn.Module):
    def __init__(self,
                 loss_type="siglip_loss", # siglip_loss / infonce_loss
                 encoder_type="spatial_and_temporal"
                ):
            super(MatchVision_contrastive_model, self).__init__()

            # create modules.
            self.visual_encoder = VisionTimesformer(encoder_type)
            self.text_encoder = TextEncoder()
            self.loss_type = loss_type
            if self.loss_type == "siglip_loss":
                self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
                self.logits_bias = nn.Parameter(torch.tensor(-10.0))
            elif self.loss_type == "infonce_loss":
                pass

    def forward(self, video_frame, comments_text, target_label):
        similarity_matrix = self.sim_mat(video_frame, comments_text)
        if self.loss_type == "siglip_loss":
            loss = self.compute_siglip_loss(similarity_matrix, target_label)
        elif self.loss_type == "infonce_loss":
            loss = self.compute_infonce_loss(similarity_matrix, target_label)
        # print("AAA", similarity_matrix.shape)
        return loss

    def encode_visual(self, video_frame):
        visual_features = self.visual_encoder(video_frame).mean(dim=1)
        return visual_features
    
    def encode_textual(self, comments_text):
        textual_features = self.text_encoder(comments_text)[0]
        return textual_features

    def sim_mat(self, video_frame, comments_text):
        visual_embeds = F.normalize(self.encode_visual(video_frame), dim=1)
        textual_embeds = F.normalize(self.encode_textual(comments_text), dim=1)
        similarity_matrix = torch.matmul(textual_embeds, visual_embeds.t())
        # print("BBB", similarity_matrix.shape)
        return similarity_matrix

    def compute_siglip_loss(self, similarity_matrix, target_label):
        # similarity_matrix = self.sim_mat(video_frame, comments_text)
        # cosine similarity as logits
        logits_per_text = similarity_matrix * self.logit_scale.exp() + self.logits_bias
        logits_per_image = logits_per_text.t()
        loss = - F.logsigmoid(target_label * logits_per_image).sum() / target_label.shape[0]
        # print(loss)
        return loss

    def compute_infonce_loss(self, similarity_matrix, target_label, temperature=0.3):
        positive_samples = target_label == 1  # Get mask for positive samples
        negative_samples = target_label == -1  # Get mask for negative samples
        pos_similarities = similarity_matrix[positive_samples.t()].view(-1)
        neg_similarities = similarity_matrix[negative_samples.t()]

        # Calculate the InfoNCE loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_similarities))) + torch.mean(F.softplus(-neg_similarities))
        return loss


    def calculate_top_k_accuracy(self, similarity_matrix, target_label, target_label_type):
        # Step 3: Prepare accuracy counts for top-1, top-3, top-5
        batch_size = similarity_matrix.size(0)
        accuracy_comment = self.calculate_top_k_accuracy_helper(batch_size, similarity_matrix, target_label)
        accuracy_type = self.calculate_top_k_accuracy_helper(batch_size, similarity_matrix, target_label_type)
        return accuracy_comment, accuracy_type

    def calculate_top_k_accuracy_helper(self, batch_size, similarity_matrix, target_label):
        top_1_correct = 0
        top_3_correct = 0
        top_5_correct = 0
        for i in range(batch_size):
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            if target_label[i, sorted_indices[0]] == 1:
                top_1_correct += 1
            if any(target_label[i, sorted_indices[:3]] == 1):
                top_3_correct += 1
            if any(target_label[i, sorted_indices[:5]] == 1):
                top_5_correct += 1

        # Calculate the accuracy for each top k
        top_1_accuracy = top_1_correct / batch_size
        top_3_accuracy = top_3_correct / batch_size
        top_5_accuracy = top_5_correct / batch_size
        accuracy = (top_1_accuracy, top_3_accuracy, top_5_accuracy)
        return accuracy