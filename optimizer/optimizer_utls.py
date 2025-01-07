import torch.optim as optim

def optimizer_sn_v2_pretrain(classifier, classifier_transformer_type="avg_pool", encoder_type="spatial_and_temporal", use_transformer=True, open_siglip=True):
    
    all_params = []

    # Siglip and projection
    no_decay_params = {'params': [param for name, param in classifier.module.siglip_model.named_parameters() if name in classifier.module.siglip_model.no_weight_decay()], 'lr': 1e-4}
    all_params.append(no_decay_params)
    
    if open_siglip:
        siglip_params = {'params': [param for name, param in classifier.module.siglip_model.named_parameters() if ('temporal_norm1' not in name and 'temporal_attn' not in name and 'temporal_fc' not in name and 'temporal_alpha_attn' not in name and name != "proj" and name not in classifier.module.siglip_model.no_weight_decay())], 'weight_decay': 1e-4, 'lr': 5e-5}
        all_params.append(siglip_params)

    # proj_params = {'params': classifier.module.siglip_model.proj, 'weight_decay': 1e-4, 'lr': 1e-4}
    # all_params.append(proj_params)
    
    # Encoder
    if encoder_type == "spatial_and_temporal":
        temporal_params = {'params': [param for name, param in classifier.module.siglip_model.named_parameters() if ('temporal_norm1' in name or 'temporal_attn' in name or 'temporal_fc' in name or 'temporal_alpha_attn' in name)], 'weight_decay': 1e-4, 'lr': 1e-4}
        all_params.append(temporal_params)
    elif encoder_type == "spatial_only":
        pass
    
    # Classifier
    classifier_ln1_params = {'params': classifier.module.classifier_ln1.parameters(), 'lr': 1e-4}
    classifier_ln2_params = {'params': classifier.module.classifier_ln2.parameters(), 'lr': 1e-4}
    all_params.append(classifier_ln1_params)
    all_params.append(classifier_ln2_params)
    
    if use_transformer:
        transformer_encoder_params = {'params': classifier.module.transformer_encoder.parameters(), 'weight_decay': 1e-4, 'lr': 1e-4}
        all_params.append(transformer_encoder_params)

    classifier_params = {'params': classifier.module.classifier.parameters(), 'weight_decay': 1e-4, 'lr': 2e-4}
    all_params.append(classifier_params)

    if classifier_transformer_type == "cls_token":
        cls_params = {'params': classifier.module.cls_token, 'weight_decay': 1e-4, 'lr': 1e-4}
        all_params.append(cls_params)

    optimizer = optim.AdamW(all_params)
    return optimizer

def optimizer_contrastive(contrastive_model, encoder_type="spatial_and_temporal", open_visual=True, open_text=True):
    
    all_params = []

    # Siglip and projection
    no_decay_params = {'params': [param for name, param in contrastive_model.module.visual_encoder.named_parameters() if name in contrastive_model.module.visual_encoder.no_weight_decay()], 'lr': 1e-4}
    all_params.append(no_decay_params)
    
    if open_visual:
        siglip_params = {'params': [param for name, param in contrastive_model.module.visual_encoder.named_parameters() if ('temporal_norm1' not in name and 'temporal_attn' not in name and 'temporal_fc' not in name and 'temporal_alpha_attn' not in name and name != "proj" and name not in contrastive_model.module.visual_encoder.no_weight_decay())], 'weight_decay': 1e-4, 'lr': 5e-5}
        all_params.append(siglip_params)

    # Encoder
    if encoder_type == "spatial_and_temporal":
        temporal_params = {'params': [param for name, param in contrastive_model.module.visual_encoder.named_parameters() if ('temporal_norm1' in name or 'temporal_attn' in name or 'temporal_fc' in name or 'temporal_alpha_attn' in name)], 'weight_decay': 1e-4, 'lr': 1e-4}
        all_params.append(temporal_params)
    elif encoder_type == "spatial_only":
        pass

    if open_text:
        text_params = {'params': [param for name, param in contrastive_model.module.text_encoder.named_parameters()], 'weight_decay': 1e-4, 'lr': 5e-5}
        all_params.append(text_params)
    
    optimizer = optim.AdamW(all_params)
    return optimizer

def optimizer_commentary_new_benchmark(commentary_model, encoder_type="spatial_and_temporal", open_visual=True, open_text=True):
    
    all_params = []
    
    ################ llama_model #################

    if open_text:
        siglip_params = {'params': [param for name, param in commentary_model.module.llama_model.named_parameters()], 'weight_decay': 1e-4, 'lr': 5e-5}
        all_params.append(siglip_params)
    
    ################### Bridge ###################

    video_qformer_params = {'params': [param for name, param in commentary_model.module.video_Qformer.named_parameters()], 'weight_decay': 1e-4, 'lr': 1e-4}
    all_params.append(video_qformer_params)

    video_frame_position_embedding_params = {'params': [param for name, param in commentary_model.module.video_frame_position_embedding.named_parameters()], 'weight_decay': 1e-4, 'lr': 1e-4}
    all_params.append(video_frame_position_embedding_params)

    llama_proj_params = {'params': [param for name, param in commentary_model.module.llama_proj.named_parameters()], 'weight_decay': 1e-4, 'lr': 1e-4}
    all_params.append(llama_proj_params)

    ############### visual_encoder ###############

    # Siglip and projection
    no_decay_params = {'params': [param for name, param in commentary_model.module.visual_encoder.named_parameters() if name in commentary_model.module.visual_encoder.no_weight_decay()], 'lr': 1e-4}
    all_params.append(no_decay_params)
    
    if open_visual:
        siglip_params = {'params': [param for name, param in commentary_model.module.visual_encoder.named_parameters() if ('temporal_norm1' not in name and 'temporal_attn' not in name and 'temporal_fc' not in name and 'temporal_alpha_attn' not in name and name != "proj" and name not in commentary_model.module.visual_encoder.no_weight_decay())], 'weight_decay': 1e-4, 'lr': 5e-5}
        all_params.append(siglip_params)

    # Encoder
    if encoder_type == "spatial_and_temporal":
        temporal_params = {'params': [param for name, param in commentary_model.module.visual_encoder.named_parameters() if ('temporal_norm1' in name or 'temporal_attn' in name or 'temporal_fc' in name or 'temporal_alpha_attn' in name)], 'weight_decay': 1e-4, 'lr': 1e-4}
        all_params.append(temporal_params)
    elif encoder_type == "spatial_only":
        pass
    
    optimizer = optim.AdamW(all_params)
    return optimizer


