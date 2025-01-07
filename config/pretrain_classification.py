config = dict(
    # Dataset and dataloader settings 
    dataset = dict(
        train = dict(
            # Here you should add the path to the json file of the dataset
            json = ["./train_data/json/SoccerReplay-1988/classification_train.json",
                    "./train_data/json/MatchTime/classification_train.json",
                    "./train_data/json/SoccerNet-v2/classification_train.json"],
            # Here you should add the keywords of the dataset, now we used 24 classes as default, you can also reference the SoccerNet-v2 version of 17 classes. (https://arxiv.org/abs/2011.13367)
            keywords = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession", "throw in", "show added time", "shot off target", "start of half game", "substitution", "saved by goal-keeper", "red card", "lead to corner", "ball out of play", "off side", "goal", "penalty", "yellow card", "foul lead to penalty", "corner", "free kick", "foul with no card"],
            sample_num = [5000, 5000, 15000, 2500, 8000, 5000, 10000, 5000, 15000, 5000, 15000, 12000, 2500, 10000, 15000, 8000, 10000, 2500, 10000, 2500, 15000, 10000, 15000],
            video_base = ["PATH_TO_VIDEO_CLIPS_OF_SOCCERREPLAY_1988",
                          "PATH_TO_VIDEO_CLIPS_OF_MATCHTIME",
                          "PATH_TO_VIDEO_CLIPS_OF_SOCCERNET_V2"],
            batch_size = 48,
            num_workers = 24,
            # sample is [middle/rand]
            sample = 'middle',
            balanced_or_not = 'unbalanced' # balanced / unbalanced 
        ),
        valid = dict(
            json = ["./train_data/json/SoccerReplay-1988/classification_test.json",
                    "./train_data/json/MatchTime/classification_test.json",
                    "./train_data/json/SoccerNet-v2/classification_test.json"],
            keywords = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession", "throw in", "show added time", "shot off target", "start of half game", "substitution", "saved by goal-keeper", "red card", "lead to corner", "ball out of play", "off side", "goal", "penalty", "yellow card", "foul lead to penalty", "corner", "free kick", "foul with no card"],
            video_base = ["PATH_TO_VIDEO_CLIPS_OF_SOCCERREPLAY_1988",
                          "PATH_TO_VIDEO_CLIPS_OF_MATCHTIME",
                          "PATH_TO_VIDEO_CLIPS_OF_SOCCERNET_V2"],
            batch_size = 48,
            num_workers = 24,
            sample = 'middle',
            balanced_or_not = 'unbalanced' # balanced / unbalanced 
        )
    ),
    # Training settings
    training_settings = dict(
        epoch = 50,
        device_ids = [4, 5, 6, 7], # Choose the device you want to use

        open_siglip = True, # Here you should choose whether to open the parts of siglip model in the encoder
        classifier_transformer_type = "avg_pool", # You can chosse from ""avg_pool"/"cls_token", in our project, we train it with "avg_pool"
        encoder_type = "spatial_and_temporal",
        use_transformer = True,

        load_checkpoint = False,
        checkpoint_path = None
    ),

    logs = dict(
        save_check_point = True,
        save_every = 1,
        check_point_base_dir = "FOLDER_TO_SAVE_CHECKPOINTS",

        wandb_configs = dict(
            use_wandb = False, # Choose whether to use wandb
            project = "YOUR_PROJECT_NAME",
            entity = "YOUR_ENTITY_NAME",
            remark = "YOUR_PROJECT_NAME",
        )
    )
)