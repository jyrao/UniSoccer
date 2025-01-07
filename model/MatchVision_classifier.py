import sys
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
import torch
import torch.nn as nn
# from model.MatchVision_from_siglip import VisionTimesformer
from model.MatchVision import VisionTimesformer
import torch.nn.functional as F
from einops import rearrange

class MatchVision_Classifier(nn.Module):
    def __init__(self, keywords=None, feature_dim=768, num_transformer_layers=2,
                  transformer_heads=8, classifier_transformer_type = "avg_pool",
                 vision_encoder_type = "spatial_and_temporal", use_transformer = True,
                 model_name = "google/siglip-base-patch16-224"):
        super(MatchVision_Classifier, self).__init__()
        
        if keywords is None:
            self.keywords = [
                'corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed', 
                'red card', 'second yellow card', 'substitution', 'start of game(half)', 
                'end of game(half)', 'yellow card', 'throw in', 'free kick', 
                'saved by goal-keeper', 'shot off target', 'clearance', 'lead to corner', 
                'off-side', 'var', 'foul (no card)', 'statistics and summary', 
                'ball possession', 'ball out of play'
            ]
        else:
            self.keywords = keywords
        
        self.siglip_model = VisionTimesformer(patch_size=16, model_name=model_name, width=768, layers=12, heads=12, output_dim=feature_dim, input_resolution=224,encoder_type=vision_encoder_type)
        self.classifier_ln1 = nn.LayerNorm(feature_dim)
        self.classifier_ln2 = nn.LayerNorm(feature_dim)
        self.classifier_transformer_type = classifier_transformer_type
        self.use_transformer = use_transformer

        if self.classifier_transformer_type == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        if self.use_transformer:
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=transformer_heads,
                dim_feedforward=feature_dim * 4,
                dropout=0.,
                activation='relu'
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer,
                num_layers=num_transformer_layers
            )

        self.classifier = nn.Linear(feature_dim, len(self.keywords))

    def forward(self, x, targets):
        logits = self.get_logits(x)
        # print(logits.shape)
        loss = F.cross_entropy(logits, targets)  # Calculate the Cross Entropy Loss here
        return loss, logits
    
    def get_logits(self, x):
        B, _, _, _, _ = x.shape
        x = self.siglip_model(x)
        x = self.classifier_ln1(x)

        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_transformer:
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)
            if self.classifier_transformer_type == "cls_token":
                x = x[0, :, :]  # cls_token是第一个元素
            elif self.classifier_transformer_type == "avg_pool":
                x = x.mean(dim=0)
        else:
            x = x.mean(dim=1)

        x = self.classifier_ln2(x)
        logits = self.classifier(x)
        return logits
    
    def get_types(self, logits):
        _, top_indices = torch.topk(logits, k=5, dim=1, largest=True, sorted=True)
        return top_indices

    def get_feature_with_cls(self, x):
        B, _, _, _, _ = x.shape
        x = self.siglip_model(x)
        x = self.classifier_ln1(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = rearrange(x, "t b m -> b t m")
        return x
    
    def get_feature_without_cls(self, x):
        B, _, _, _, _ = x.shape
        x = self.siglip_model(x)
        x = self.classifier_ln1(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = self.classifier_ln2(x)
        x = rearrange(x, "t b m -> b t m")
        return x
    
    def get_feature_before_transformer(self, x):
        B, _, _, _, _ = x.shape
        x = self.siglip_model(x)
        x = self.classifier_ln1(x)
        return x
