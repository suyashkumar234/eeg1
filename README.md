## Updated Best Result-
6-Fold Cross Subject Accuracies: [54.26799610894941, 58.40527723735408, 54.81517509727627, 57.45379377431906, 52.98212548638133, 53.16451848249027]
Mean Accuracy over 6 folds: 55.1815



Hyperparameter used-
    args.win_time = 2
    args.win_len = math.ceil(args.fs * args.win_time)
    args.overlap = 0.5
    args.window_lap = args.win_len * (1 - args.overlap)
    args.lambda_domain = 0.1
    args.lambda_contrastive = 0.8
    args.temperature = 0.07         # Sharper for strong contrastive
    args.optimizer = 'SGD'
    args.lr = 1e-2
    args.batch_size = 64            # Smaller batch = more diverse pairs
    args.weight_decay = 1e-3
    args.momentum = 0.9
    args.lam = 0.2
    args.lr_decayrate = 0.5
    args.max_epoch = 100
    args.patience = 10
    args.log_interval = 10
    args.use_domain_adversarial = True   # Set to False to disable domain adversarial
    args.use_contrastive = True  
