from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import einops
import contextlib
import sys, os
sys.path.append('PATH_TO_FOLDER_OF_THIS_PROJECT')
from model.matchvoice_Qformer import BertConfig, BertLMHeadModel
from model.MatchVision import VisionTimesformer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from typing import List
import pickle as pkl
import sys
import io
from peft import get_peft_model, LoraConfig

def process_output_tokens(predict_model, tokens):
    output_texts = []
    for output_token in tokens:
        output_text = predict_model.tokenizer.decode(output_token)
        end_token_index = output_text.find('<|end_of_text|>')
        if end_token_index != -1:
            output_text = output_text[:end_token_index]
        output_texts.append(output_text)
    return output_texts

class RestrictTokenGenerationLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_id_list: List[int]):
        super().__init__()
        self.allowed_token_id_list = allowed_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float('inf'))
        for allowed_id in self.allowed_token_id_list:
            mask[:, allowed_id] = scores[:, allowed_id]
        return mask

class matchvoice_model_all_blocks(nn.Module):
    def __init__(self,
                 # Visual Encoder
                 load_checkpoint = True,
                 visual_encoder_checkpoint = "PATH_TO_VISUAL_BACKBONE_CHECKPOINT",
                 # LLM part
                 llm_ckpt = "Meta-Llama-3-8B-Instruct",
                 tokenizer_ckpt = "Meta-Llama-3-8B-Instruct",
                 # Q-former part
                 max_frame_pos = 128,
                 window = 30,
                 num_query_tokens = 32,
                 num_video_query_token = 32,
                 num_features = 512,
                #  device = "cuda:0",
                 inference = False,
                 file_path = './words_world/merge.pkl',
                 need_temporal = True,
                 encoder_type = "spatial_and_temporal",
                 open_visual_encoder = False,
                 open_llm_decoder = False,
                 llm_lora_rank = 16,
                 llm_lora_dropout = 0.05,
                 **kwargs,
                 ):
        super().__init__()
        if len(kwargs):
            print(f'kwargs not used: {kwargs}')
        # self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        self.tokenizer.add_tokens(["[PLAYER]","[TEAM]","[COACH]","[REFEREE]","([TEAM])"], special_tokens=True)
        self.open_visual_encoder = open_visual_encoder
        if self.open_visual_encoder:
            print("======== Visual encoder is opened!")
        self.open_llm_decoder = open_llm_decoder
        # print("AA", open_llm_decoder)
        self.llama_model = AutoModelForCausalLM.from_pretrained(llm_ckpt, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.tokenizer))
        if self.open_llm_decoder == True:
            lora_config = LoraConfig(
                r=llm_lora_rank, 
                lora_alpha=llm_lora_rank*2, 
                target_modules=["q_proj", "v_proj"], 
                lora_dropout=llm_lora_dropout,
                bias="none" 
            )
            self.llama_model = get_peft_model(self.llama_model, lora_config)
            print(f"===== LLaMA open with LoRA rank = {llm_lora_rank}")


        self.ln_vision = LayerNorm(num_features)
        self.num_query_tokens = num_query_tokens,
        self.num_video_query_token = num_video_query_token
        self.inference = inference
        self.need_temporal = need_temporal

        # Initialize visual encoder
        self.visual_encoder = VisionTimesformer(encoder_type=encoder_type)
        if load_checkpoint:
            checkpoint = torch.load(visual_encoder_checkpoint, map_location="cpu")
            new_state_dict = {key.replace("module.siglip_model.", ""): value for key, value in checkpoint['state_dict'].items()}
            self.visual_encoder.load_state_dict(new_state_dict, strict=False)

        # Initialize video Q-former
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,
                                                                             vision_width=num_features,
                                                                             num_hidden_layers =2)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # llama projection
        self.llama_proj = nn.Linear(
            self.video_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        # video frame positional embedding
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, num_features)
        self.window = window

        # ################## move to device ##################

        # LLaMA model parameters
        if not self.open_llm_decoder:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            pass

        # Visual encoder parameters
        if not self.open_visual_encoder:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
        else:
            pass
        self.video_frame_position_embedding = self.video_frame_position_embedding

        with open(file_path, 'rb') as file:
            self.token_ids_list = pkl.load(file)
        self.token_ids_list.append(128000)
        self.token_ids_list.append(128001)
        self.processor = RestrictTokenGenerationLogitsProcessor(allowed_token_id_list=self.token_ids_list)
        self.logits_prosessors = LogitsProcessorList()
        self.logits_prosessors.append(self.processor)

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    
    def maybe_autocast(self, embedding_cat, dtype=torch.float16):
        enable_autocast = embedding_cat.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples, validating=False):
        video_frames = samples['frames']
        targets = samples['labels']
        atts_llama = samples['attention_mask']
        inputs_ids = samples['input_ids']
        caption_text = samples['caption_text']
        video_path = samples['video_path']

        video_features = self.visual_encoder(video_frames)

        batch_size = None
        time_length = None
        try:
            batch_size, time_length, _ = video_features.size()
        except:
            batch_size, time_length, _, _ = video_features.size()

        if len(video_features.size()) != 4:
            video_features = video_features.unsqueeze(-2)
        video_features = self.ln_vision(video_features)
        video_features = einops.rearrange(video_features, 'b t n f -> (b t) n f', b=batch_size, t=time_length)

        # print("ei")
        if self.need_temporal == "yes":
            # print("temporal enrolled")
            position_ids = torch.arange(time_length, dtype=torch.long, device=video_features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f',b=batch_size,t=time_length)
        
        if self.need_temporal == "yes":
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(frame_hidden_state)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(frame_hidden_state.device)

        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = self.llama_proj(video_hidden)
        if self.inference:
            return self.generate_text(inputs_llama)

        if validating:
            temp_res_text = self.generate_text(inputs_llama)
            return temp_res_text, caption_text, video_path
        
        visual_label = torch.full((batch_size, self.num_video_query_token), -100, dtype=targets.dtype).to(inputs_llama.device)
        concat_targets = torch.cat((visual_label, targets), dim=1).to(inputs_llama.device)
        temp_input_ids = inputs_ids.clone().to(inputs_llama.device)
        if self.open_llm_decoder == True:
            targets_embeds = self.llama_model.base_model.model.model.embed_tokens(temp_input_ids)
        else:
            targets_embeds = self.llama_model.model.embed_tokens(temp_input_ids)
        embedding_cat = torch.cat((inputs_llama, targets_embeds), dim=1)
        mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_llama.dtype).to(inputs_llama.device)
        mask = torch.concat((mask_prefix, atts_llama), dim=1).to(inputs_llama.device)
    
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        with self.maybe_autocast(embedding_cat):
            outputs = self.llama_model(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                return_dict=True,
                labels=concat_targets,
            )
        sys.stdout = original_stdout
        loss = outputs.loss
        return loss
    
    def generate_text(self, inputs_llama):
        if self.open_llm_decoder == True:
            start_embeds = self.llama_model.base_model.model.model.embed_tokens(torch.tensor([128000]).to(inputs_llama.device))
        else:
            start_embeds = self.llama_model.model.embed_tokens(torch.tensor([128000]).to(inputs_llama.device))
        inputs_llama_with_s = torch.cat([inputs_llama, start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(dtype=torch.bfloat16)
        temp_res_tokens = self.llama_model.generate(
            logits_processor=self.logits_prosessors,
            renormalize_logits=True,
            inputs_embeds=inputs_llama_with_s,
            max_new_tokens=128,
            num_beams=5,
            do_sample=True,
            min_length=5,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
        res_text = process_output_tokens(self, temp_res_tokens)
        return res_text

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)