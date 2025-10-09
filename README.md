## Updated Best Result 
Val Accuracy- 6-Fold Cross Subject Accuracies: [59.36588035019456, 52.751094357976655, 53.213156614785994, 54.32879377431906, 55.52650778210116, 54.37439202334631]
Mean Accuracy over 6 folds: 54.9266

Current device is cuda:0
start time: task1_AAD_MM-AAD_trainer3_2025-10-09-04-20-03
============================================================================================================
Arguments =
	_action_groups: [<argparse._ArgumentGroup object at 0x7f7ad77c5880>, <argparse._ArgumentGroup object at 0x7f7aae6079d0>]
	_actions: [_HelpAction(option_strings=['-h', '--help'], dest='help', nargs=0, const=None, default='==SUPPRESS==', type=None, choices=None, required=False, help='show this help message and exit', metavar=None)]
	_defaults: {}
	_has_negative_number_optionals: []
	_mutually_exclusive_groups: []
	_negative_number_matcher: re.compile('^-\\d+$|^-\\d*\\.\\d+$')
	_option_string_actions: {'-h': _HelpAction(option_strings=['-h', '--help'], dest='help', nargs=0, const=None, default='==SUPPRESS==', type=None, choices=None, required=False, help='show this help message and exit', metavar=None), '--help': _HelpAction(option_strings=['-h', '--help'], dest='help', nargs=0, const=None, default='==SUPPRESS==', type=None, choices=None, required=False, help='show this help message and exit', metavar=None)}
	_optionals: <argparse._ArgumentGroup object at 0x7f7aae6079d0>
	_positionals: <argparse._ArgumentGroup object at 0x7f7ad77c5880>
	_registries: {'action': {None: <class 'argparse._StoreAction'>, 'store': <class 'argparse._StoreAction'>, 'store_const': <class 'argparse._StoreConstAction'>, 'store_true': <class 'argparse._StoreTrueAction'>, 'store_false': <class 'argparse._StoreFalseAction'>, 'append': <class 'argparse._AppendAction'>, 'append_const': <class 'argparse._AppendConstAction'>, 'count': <class 'argparse._CountAction'>, 'help': <class 'argparse._HelpAction'>, 'version': <class 'argparse._VersionAction'>, 'parsers': <class 'argparse._SubParsersAction'>, 'extend': <class 'argparse._ExtendAction'>}, 'type': {None: <function ArgumentParser.__init__.<locals>.identity at 0x7f7aae6278b0>}}
	_subparsers: None
	add_help: True
	allow_abbrev: True
	alpha: 0.99
	argument_default: None
	batch_size: 64
	conflict_handler: error
	data_path: /home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data/
	dataset: MM-AAD
	description: None
	eeg_channel: 32
	epilog: None
	exit_on_error: True
	fig_path: ./exps/cross-subject/DARNet/figures/
	formatter_class: <class 'argparse.HelpFormatter'>
	fromfile_prefix_chars: None
	fs: 128
	label_path: /home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label/
	lam: 0.2
	lambda_domain: 0.08
	log_interval: 10
	lr: 0.003
	lr_decayrate: 0.6
	max_epoch: 100
	model: DARNet
	model_save_path: ./exps/cross-subject/DARNet/baseline_task1_AAD_MM-AAD_trainer3_2025-10-09-04-20-03
	optimizer: RMSprop
	overlap: 0.5
	patience: 10
	prefix_chars: -
	prog: trainer3.py
	seed: 42
	start_time: task1_AAD_MM-AAD_trainer3_2025-10-09-04-20-03
	subject_number: 40
	trail_number: 20
	usage: None
	use_domain_adversarial: True
	weight_decay: 0.0005
	win_len: 192
	win_time: 1.5
	window_lap: 96.0
============================================================================================================

========== Fold 1 ==========
Train IDs: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Val IDs:   [1, 2, 3, 4, 5]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[1, 2, 3, 4, 5] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5470 | Valid Loss 1.1049 | Train Acc 70.0839| Valid Acc 48.8540
Validation acc increase (0.000000 --> 48.853964) in epoch (1).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.3875 | Valid Loss 1.6149 | Train Acc 81.9960| Valid Acc 49.9757
Validation acc increase (48.853964 --> 49.975681) in epoch (2).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.3162 | Valid Loss 1.5994 | Train Acc 86.1077| Valid Acc 53.3165
Validation acc increase (49.975681 --> 53.316513) in epoch (3).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2714 | Valid Loss 1.7982 | Train Acc 88.3475| Valid Acc 52.3833
TestSub: S[1, 2, 3, 4, 5] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.2475 | Valid Loss 1.8380 | Train Acc 89.6395| Valid Acc 52.7754
TestSub: S[1, 2, 3, 4, 5] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.2269 | Valid Loss 1.9754 | Train Acc 90.6414| Valid Acc 52.4714
TestSub: S[1, 2, 3, 4, 5] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.2171 | Valid Loss 1.8763 | Train Acc 91.1169| Valid Acc 52.7511
TestSub: S[1, 2, 3, 4, 5] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.2017 | Valid Loss 2.0149 | Train Acc 91.8951| Valid Acc 52.2252
TestSub: S[1, 2, 3, 4, 5] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1939 | Valid Loss 1.9463 | Train Acc 92.1960| Valid Acc 53.4563
Validation acc increase (53.316513 --> 53.456347) in epoch (9).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1853 | Valid Loss 1.9137 | Train Acc 92.7110| Valid Acc 52.6508
TestSub: S[1, 2, 3, 4, 5] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1756 | Valid Loss 1.8912 | Train Acc 93.0952| Valid Acc 53.1615
TestSub: S[1, 2, 3, 4, 5] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1687 | Valid Loss 1.9351 | Train Acc 93.4320| Valid Acc 54.5750
Validation acc increase (53.456347 --> 54.575024) in epoch (12).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1577 | Valid Loss 1.8213 | Train Acc 93.9123| Valid Acc 52.9517
TestSub: S[1, 2, 3, 4, 5] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1525 | Valid Loss 2.1395 | Train Acc 94.1561| Valid Acc 54.0491
TestSub: S[1, 2, 3, 4, 5] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1476 | Valid Loss 2.0119 | Train Acc 94.3385| Valid Acc 53.5384
TestSub: S[1, 2, 3, 4, 5] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.1422 | Valid Loss 2.0094 | Train Acc 94.5787| Valid Acc 52.2921
TestSub: S[1, 2, 3, 4, 5] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.1388 | Valid Loss 2.0269 | Train Acc 94.7434| Valid Acc 53.7847
TestSub: S[1, 2, 3, 4, 5] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.1345 | Valid Loss 2.0464 | Train Acc 94.9112| Valid Acc 53.4047
TestSub: S[1, 2, 3, 4, 5] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.1331 | Valid Loss 2.0140 | Train Acc 94.9739| Valid Acc 55.0401
Validation acc increase (54.575024 --> 55.040126) in epoch (19).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.1309 | Valid Loss 1.9279 | Train Acc 95.1082| Valid Acc 53.9792
TestSub: S[1, 2, 3, 4, 5] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.1563 | Valid Loss 1.7863 | Train Acc 93.9069| Valid Acc 56.0676
Validation acc increase (55.040126 --> 56.067607) in epoch (21).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.1405 | Valid Loss 1.6930 | Train Acc 94.6717| Valid Acc 56.4902
Validation acc increase (56.067607 --> 56.490151) in epoch (22).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.1308 | Valid Loss 1.8109 | Train Acc 95.0541| Valid Acc 54.5416
TestSub: S[1, 2, 3, 4, 5] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.1259 | Valid Loss 1.8091 | Train Acc 95.2766| Valid Acc 52.7602
TestSub: S[1, 2, 3, 4, 5] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.1225 | Valid Loss 1.6968 | Train Acc 95.4110| Valid Acc 55.2712
TestSub: S[1, 2, 3, 4, 5] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.1197 | Valid Loss 1.6780 | Train Acc 95.5472| Valid Acc 56.2652
TestSub: S[1, 2, 3, 4, 5] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.1173 | Valid Loss 1.7513 | Train Acc 95.6420| Valid Acc 56.6573
Validation acc increase (56.490151 --> 56.657344) in epoch (27).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.1185 | Valid Loss 1.8328 | Train Acc 95.6134| Valid Acc 56.2865
TestSub: S[1, 2, 3, 4, 5] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.1136 | Valid Loss 1.8854 | Train Acc 95.7715| Valid Acc 54.9884
TestSub: S[1, 2, 3, 4, 5] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.1137 | Valid Loss 1.8762 | Train Acc 95.8098| Valid Acc 54.6267
TestSub: S[1, 2, 3, 4, 5] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.1134 | Valid Loss 1.8823 | Train Acc 95.7806| Valid Acc 56.5054
TestSub: S[1, 2, 3, 4, 5] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.1093 | Valid Loss 1.8433 | Train Acc 95.9332| Valid Acc 55.6177
TestSub: S[1, 2, 3, 4, 5] Epoch 33 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0743 | Train Loss 0.1114 | Valid Loss 1.7353 | Train Acc 95.8633| Valid Acc 58.1742
Validation acc increase (56.657344 --> 58.174246) in epoch (33).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 34 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0748 | Train Loss 0.1107 | Valid Loss 1.6757 | Train Acc 95.9028| Valid Acc 57.5754
TestSub: S[1, 2, 3, 4, 5] Epoch 35 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0753 | Train Loss 0.1080 | Valid Loss 1.8539 | Train Acc 95.9959| Valid Acc 56.4324
TestSub: S[1, 2, 3, 4, 5] Epoch 36 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0757 | Train Loss 0.1070 | Valid Loss 1.9255 | Train Acc 96.0767| Valid Acc 52.6265
TestSub: S[1, 2, 3, 4, 5] Epoch 37 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0761 | Train Loss 0.1062 | Valid Loss 1.7704 | Train Acc 96.0871| Valid Acc 55.1891
TestSub: S[1, 2, 3, 4, 5] Epoch 38 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0765 | Train Loss 0.1054 | Valid Loss 1.9207 | Train Acc 96.1199| Valid Acc 56.3078
TestSub: S[1, 2, 3, 4, 5] Epoch 39 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0768 | Train Loss 0.1049 | Valid Loss 1.8845 | Train Acc 96.0962| Valid Acc 55.9187
TestSub: S[1, 2, 3, 4, 5] Epoch 40 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0771 | Train Loss 0.1046 | Valid Loss 1.8918 | Train Acc 96.1661| Valid Acc 54.5081
TestSub: S[1, 2, 3, 4, 5] Epoch 41 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0774 | Train Loss 0.1063 | Valid Loss 1.9208 | Train Acc 96.0767| Valid Acc 55.7059
TestSub: S[1, 2, 3, 4, 5] Epoch 42 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0776 | Train Loss 0.1025 | Valid Loss 2.1284 | Train Acc 96.1758| Valid Acc 54.3896
TestSub: S[1, 2, 3, 4, 5] Epoch 43 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0779 | Train Loss 0.1011 | Valid Loss 2.0181 | Train Acc 96.2658| Valid Acc 56.7394
TestSub: S[1, 2, 3, 4, 5] Epoch 44 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0781 | Train Loss 0.1005 | Valid Loss 1.9443 | Train Acc 96.2944| Valid Acc 55.6420
TestSub: S[1, 2, 3, 4, 5] Epoch 45 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0782 | Train Loss 0.0999 | Valid Loss 2.0256 | Train Acc 96.2609| Valid Acc 57.8490
TestSub: S[1, 2, 3, 4, 5] Epoch 46 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0784 | Train Loss 0.1012 | Valid Loss 1.8987 | Train Acc 96.2774| Valid Acc 57.4204
TestSub: S[1, 2, 3, 4, 5] Epoch 47 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0786 | Train Loss 0.0997 | Valid Loss 2.0068 | Train Acc 96.3461| Valid Acc 56.5661
TestSub: S[1, 2, 3, 4, 5] Epoch 48 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0787 | Train Loss 0.0979 | Valid Loss 1.9095 | Train Acc 96.4269| Valid Acc 57.1984
TestSub: S[1, 2, 3, 4, 5] Epoch 49 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0788 | Train Loss 0.0982 | Valid Loss 2.0536 | Train Acc 96.3838| Valid Acc 55.1070
TestSub: S[1, 2, 3, 4, 5] Epoch 50 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0789 | Train Loss 0.0977 | Valid Loss 2.0487 | Train Acc 96.4044| Valid Acc 56.8367
TestSub: S[1, 2, 3, 4, 5] Epoch 51 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0790 | Train Loss 0.0969 | Valid Loss 1.9226 | Train Acc 96.4531| Valid Acc 57.3626
TestSub: S[1, 2, 3, 4, 5] Epoch 52 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0791 | Train Loss 0.0959 | Valid Loss 1.9613 | Train Acc 96.5066| Valid Acc 55.3928
TestSub: S[1, 2, 3, 4, 5] Epoch 53 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0792 | Train Loss 0.0978 | Valid Loss 2.0359 | Train Acc 96.4245| Valid Acc 54.6297
TestSub: S[1, 2, 3, 4, 5] Epoch 54 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0940 | Valid Loss 2.1379 | Train Acc 96.5698| Valid Acc 54.7270
TestSub: S[1, 2, 3, 4, 5] Epoch 55 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0958 | Valid Loss 1.9627 | Train Acc 96.5169| Valid Acc 55.4444
TestSub: S[1, 2, 3, 4, 5] Epoch 56 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0794 | Train Loss 0.0953 | Valid Loss 2.2105 | Train Acc 96.4506| Valid Acc 54.5264
TestSub: S[1, 2, 3, 4, 5] Epoch 57 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0950 | Valid Loss 1.8948 | Train Acc 96.4810| Valid Acc 55.7819
TestSub: S[1, 2, 3, 4, 5] Epoch 58 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0932 | Valid Loss 2.0817 | Train Acc 96.5649| Valid Acc 55.5843
TestSub: S[1, 2, 3, 4, 5] Epoch 59 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0930 | Valid Loss 2.1085 | Train Acc 96.6093| Valid Acc 55.6572
TestSub: S[1, 2, 3, 4, 5] Epoch 60 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0922 | Valid Loss 1.9256 | Train Acc 96.6196| Valid Acc 55.8974
TestSub: S[1, 2, 3, 4, 5] Epoch 61 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0918 | Valid Loss 2.0271 | Train Acc 96.6318| Valid Acc 56.7425
TestSub: S[1, 2, 3, 4, 5] Epoch 62 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0937 | Valid Loss 1.9003 | Train Acc 96.6045| Valid Acc 55.7332
TestSub: S[1, 2, 3, 4, 5] Epoch 63 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0910 | Valid Loss 1.9840 | Train Acc 96.6707| Valid Acc 59.3659
Validation acc increase (58.174246 --> 59.365880) in epoch (63).  Saving model ...
TestSub: S[1, 2, 3, 4, 5] Epoch 64 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0919 | Valid Loss 1.9778 | Train Acc 96.6452| Valid Acc 57.5085
TestSub: S[1, 2, 3, 4, 5] Epoch 65 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0909 | Valid Loss 1.8571 | Train Acc 96.6610| Valid Acc 57.0586
TestSub: S[1, 2, 3, 4, 5] Epoch 66 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0907 | Valid Loss 1.9630 | Train Acc 96.6403| Valid Acc 57.1224
TestSub: S[1, 2, 3, 4, 5] Epoch 67 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0893 | Valid Loss 2.0552 | Train Acc 96.7200| Valid Acc 55.9369
TestSub: S[1, 2, 3, 4, 5] Epoch 68 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0889 | Valid Loss 1.9856 | Train Acc 96.7723| Valid Acc 55.5204
TestSub: S[1, 2, 3, 4, 5] Epoch 69 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0896 | Valid Loss 1.8972 | Train Acc 96.7157| Valid Acc 55.1708
TestSub: S[1, 2, 3, 4, 5] Epoch 70 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0909 | Valid Loss 2.0118 | Train Acc 96.6610| Valid Acc 56.3655
TestSub: S[1, 2, 3, 4, 5] Epoch 71 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0893 | Valid Loss 2.0775 | Train Acc 96.7254| Valid Acc 55.1404
TestSub: S[1, 2, 3, 4, 5] Epoch 72 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0880 | Valid Loss 1.9284 | Train Acc 96.7765| Valid Acc 55.2499
TestSub: S[1, 2, 3, 4, 5] Epoch 73 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0892 | Valid Loss 2.1950 | Train Acc 96.7285| Valid Acc 54.9884
TestSub: S[1, 2, 3, 4, 5] Epoch 74 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0873 | Valid Loss 2.0380 | Train Acc 96.7996| Valid Acc 55.5295
TestSub: S[1, 2, 3, 4, 5] Epoch 75 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0855 | Valid Loss 2.0647 | Train Acc 96.8665| Valid Acc 58.7518
TestSub: S[1, 2, 3, 4, 5] Epoch 76 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0861 | Valid Loss 1.9371 | Train Acc 96.8714| Valid Acc 57.0373
TestSub: S[1, 2, 3, 4, 5] Epoch 77 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0887 | Valid Loss 2.0046 | Train Acc 96.7419| Valid Acc 57.2015
TestSub: S[1, 2, 3, 4, 5] Epoch 78 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0855 | Valid Loss 2.0629 | Train Acc 96.9151| Valid Acc 58.9403
TestSub: S[1, 2, 3, 4, 5] Epoch 79 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0867 | Valid Loss 2.0556 | Train Acc 96.8464| Valid Acc 56.9097
TestSub: S[1, 2, 3, 4, 5] Epoch 80 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0893 | Valid Loss 2.0746 | Train Acc 96.7455| Valid Acc 56.4111
TestSub: S[1, 2, 3, 4, 5] Epoch 81 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0869 | Valid Loss 2.0528 | Train Acc 96.8811| Valid Acc 56.9948
TestSub: S[1, 2, 3, 4, 5] Epoch 82 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0855 | Valid Loss 2.0193 | Train Acc 96.9261| Valid Acc 58.7792
TestSub: S[1, 2, 3, 4, 5] Epoch 83 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0857 | Valid Loss 2.1506 | Train Acc 96.9084| Valid Acc 58.1043
TestSub: S[1, 2, 3, 4, 5] Epoch 84 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0858 | Valid Loss 1.8744 | Train Acc 96.8519| Valid Acc 57.9523
TestSub: S[1, 2, 3, 4, 5] Epoch 85 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0848 | Valid Loss 2.0784 | Train Acc 96.8957| Valid Acc 55.0857
TestSub: S[1, 2, 3, 4, 5] Epoch 86 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0842 | Valid Loss 2.2697 | Train Acc 96.9607| Valid Acc 55.3684
TestSub: S[1, 2, 3, 4, 5] Epoch 87 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0849 | Valid Loss 2.2165 | Train Acc 96.9419| Valid Acc 55.5569
TestSub: S[1, 2, 3, 4, 5] Epoch 88 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0850 | Valid Loss 2.3385 | Train Acc 96.8744| Valid Acc 55.7849
TestSub: S[1, 2, 3, 4, 5] Epoch 89 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0855 | Valid Loss 2.2764 | Train Acc 96.8896| Valid Acc 57.7213
TestSub: S[1, 2, 3, 4, 5] Epoch 90 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0844 | Valid Loss 2.1968 | Train Acc 96.9176| Valid Acc 54.2406
TestSub: S[1, 2, 3, 4, 5] Epoch 91 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0841 | Valid Loss 2.3141 | Train Acc 96.9413| Valid Acc 54.6389
TestSub: S[1, 2, 3, 4, 5] Epoch 92 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0841 | Valid Loss 2.0718 | Train Acc 96.9972| Valid Acc 56.2348
TestSub: S[1, 2, 3, 4, 5] Epoch 93 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0848 | Valid Loss 2.0967 | Train Acc 96.9042| Valid Acc 54.6206
Early stopping triggered after 93 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[1, 2, 3, 4, 5] |Best epoch:63 | Test Loss:1.9840 | Best Acc 59.3659 | Savemodel Acc 59.3659
--------------------------------------------------

========== Fold 2 ==========
Train IDs: [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Val IDs:   [6, 7, 8, 9, 10]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[6, 7, 8, 9, 10] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5626 | Valid Loss 1.0758 | Train Acc 67.7018| Valid Acc 51.8999
Validation acc increase (0.000000 --> 51.899927) in epoch (1).  Saving model ...
TestSub: S[6, 7, 8, 9, 10] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.4098 | Valid Loss 1.4866 | Train Acc 79.9112| Valid Acc 52.7511
Validation acc increase (51.899927 --> 52.751094) in epoch (2).  Saving model ...
TestSub: S[6, 7, 8, 9, 10] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.3109 | Valid Loss 1.8372 | Train Acc 86.5181| Valid Acc 49.4467
TestSub: S[6, 7, 8, 9, 10] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2563 | Valid Loss 2.0541 | Train Acc 89.3482| Valid Acc 49.6991
TestSub: S[6, 7, 8, 9, 10] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.2267 | Valid Loss 2.1282 | Train Acc 90.6761| Valid Acc 48.1457
TestSub: S[6, 7, 8, 9, 10] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.2142 | Valid Loss 2.0869 | Train Acc 91.3643| Valid Acc 49.9514
TestSub: S[6, 7, 8, 9, 10] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.1994 | Valid Loss 2.2255 | Train Acc 92.0264| Valid Acc 49.2887
TestSub: S[6, 7, 8, 9, 10] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.1879 | Valid Loss 2.4132 | Train Acc 92.4538| Valid Acc 49.6869
TestSub: S[6, 7, 8, 9, 10] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1811 | Valid Loss 1.8688 | Train Acc 92.8557| Valid Acc 49.0607
TestSub: S[6, 7, 8, 9, 10] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1740 | Valid Loss 2.3417 | Train Acc 93.1475| Valid Acc 48.2581
TestSub: S[6, 7, 8, 9, 10] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1677 | Valid Loss 2.3843 | Train Acc 93.4345| Valid Acc 50.0274
TestSub: S[6, 7, 8, 9, 10] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1599 | Valid Loss 2.3719 | Train Acc 93.7342| Valid Acc 47.1972
TestSub: S[6, 7, 8, 9, 10] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1550 | Valid Loss 2.7563 | Train Acc 93.9354| Valid Acc 48.1973
TestSub: S[6, 7, 8, 9, 10] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1492 | Valid Loss 2.3419 | Train Acc 94.2181| Valid Acc 50.4286
TestSub: S[6, 7, 8, 9, 10] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1472 | Valid Loss 2.3847 | Train Acc 94.3045| Valid Acc 49.2643
TestSub: S[6, 7, 8, 9, 10] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.1461 | Valid Loss 2.2076 | Train Acc 94.3470| Valid Acc 50.0426
TestSub: S[6, 7, 8, 9, 10] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.1396 | Valid Loss 2.3004 | Train Acc 94.6486| Valid Acc 47.8934
TestSub: S[6, 7, 8, 9, 10] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.1399 | Valid Loss 2.3827 | Train Acc 94.6091| Valid Acc 47.2428
TestSub: S[6, 7, 8, 9, 10] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.1360 | Valid Loss 2.4158 | Train Acc 94.7459| Valid Acc 47.7657
TestSub: S[6, 7, 8, 9, 10] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.1339 | Valid Loss 2.7410 | Train Acc 94.8577| Valid Acc 50.8238
TestSub: S[6, 7, 8, 9, 10] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.1348 | Valid Loss 2.5969 | Train Acc 94.8176| Valid Acc 49.2370
TestSub: S[6, 7, 8, 9, 10] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.1304 | Valid Loss 2.7078 | Train Acc 94.9945| Valid Acc 50.6870
TestSub: S[6, 7, 8, 9, 10] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.1296 | Valid Loss 2.6704 | Train Acc 95.0626| Valid Acc 49.1701
TestSub: S[6, 7, 8, 9, 10] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.1298 | Valid Loss 2.5461 | Train Acc 95.0134| Valid Acc 49.2157
TestSub: S[6, 7, 8, 9, 10] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.1270 | Valid Loss 2.3144 | Train Acc 95.2250| Valid Acc 50.8420
TestSub: S[6, 7, 8, 9, 10] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.1263 | Valid Loss 2.3673 | Train Acc 95.2505| Valid Acc 51.7662
TestSub: S[6, 7, 8, 9, 10] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.1253 | Valid Loss 2.5788 | Train Acc 95.2572| Valid Acc 48.7749
TestSub: S[6, 7, 8, 9, 10] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.1232 | Valid Loss 2.5854 | Train Acc 95.3362| Valid Acc 49.2522
TestSub: S[6, 7, 8, 9, 10] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.1231 | Valid Loss 2.5871 | Train Acc 95.3642| Valid Acc 51.9729
TestSub: S[6, 7, 8, 9, 10] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.1228 | Valid Loss 2.4615 | Train Acc 95.2857| Valid Acc 49.5623
TestSub: S[6, 7, 8, 9, 10] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.1208 | Valid Loss 2.5666 | Train Acc 95.4608| Valid Acc 51.1430
TestSub: S[6, 7, 8, 9, 10] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.1214 | Valid Loss 2.2857 | Train Acc 95.4712| Valid Acc 49.0546
Early stopping triggered after 32 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[6, 7, 8, 9, 10] |Best epoch:2 | Test Loss:1.4866 | Best Acc 52.7511 | Savemodel Acc 52.7511
--------------------------------------------------

========== Fold 3 ==========
Train IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Val IDs:   [11, 12, 13, 14, 15]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[11, 12, 13, 14, 15] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5469 | Valid Loss 1.2671 | Train Acc 70.0340| Valid Acc 49.3069
Validation acc increase (0.000000 --> 49.306907) in epoch (1).  Saving model ...
TestSub: S[11, 12, 13, 14, 15] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.3660 | Valid Loss 1.2819 | Train Acc 83.5567| Valid Acc 50.1125
Validation acc increase (49.306907 --> 50.112476) in epoch (2).  Saving model ...
TestSub: S[11, 12, 13, 14, 15] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.3026 | Valid Loss 1.6884 | Train Acc 87.0015| Valid Acc 51.7996
Validation acc increase (50.112476 --> 51.799611) in epoch (3).  Saving model ...
TestSub: S[11, 12, 13, 14, 15] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2693 | Valid Loss 1.6688 | Train Acc 88.6637| Valid Acc 52.2374
Validation acc increase (51.799611 --> 52.237354) in epoch (4).  Saving model ...
TestSub: S[11, 12, 13, 14, 15] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.2413 | Valid Loss 1.7212 | Train Acc 89.9830| Valid Acc 53.2132
Validation acc increase (52.237354 --> 53.213157) in epoch (5).  Saving model ...
TestSub: S[11, 12, 13, 14, 15] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.2185 | Valid Loss 1.6777 | Train Acc 91.2232| Valid Acc 51.1217
TestSub: S[11, 12, 13, 14, 15] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.2122 | Valid Loss 1.7638 | Train Acc 91.4287| Valid Acc 51.5838
TestSub: S[11, 12, 13, 14, 15] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.2035 | Valid Loss 1.7601 | Train Acc 91.9370| Valid Acc 51.0062
TestSub: S[11, 12, 13, 14, 15] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1939 | Valid Loss 1.7832 | Train Acc 92.3158| Valid Acc 50.1824
TestSub: S[11, 12, 13, 14, 15] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1870 | Valid Loss 1.9571 | Train Acc 92.6234| Valid Acc 50.3861
TestSub: S[11, 12, 13, 14, 15] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1823 | Valid Loss 1.7860 | Train Acc 92.8715| Valid Acc 49.5319
TestSub: S[11, 12, 13, 14, 15] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1818 | Valid Loss 1.7010 | Train Acc 92.8459| Valid Acc 51.9394
TestSub: S[11, 12, 13, 14, 15] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1804 | Valid Loss 1.8187 | Train Acc 93.0107| Valid Acc 50.0456
TestSub: S[11, 12, 13, 14, 15] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1764 | Valid Loss 1.7325 | Train Acc 93.1031| Valid Acc 51.9364
TestSub: S[11, 12, 13, 14, 15] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1798 | Valid Loss 1.8192 | Train Acc 93.0715| Valid Acc 49.5258
TestSub: S[11, 12, 13, 14, 15] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.1741 | Valid Loss 1.7304 | Train Acc 93.2466| Valid Acc 51.5047
TestSub: S[11, 12, 13, 14, 15] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.1658 | Valid Loss 1.8069 | Train Acc 93.5980| Valid Acc 49.5136
TestSub: S[11, 12, 13, 14, 15] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.1655 | Valid Loss 1.8854 | Train Acc 93.6266| Valid Acc 47.8508
TestSub: S[11, 12, 13, 14, 15] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.1669 | Valid Loss 1.9911 | Train Acc 93.5688| Valid Acc 49.6474
TestSub: S[11, 12, 13, 14, 15] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.1636 | Valid Loss 1.7831 | Train Acc 93.7220| Valid Acc 49.8024
TestSub: S[11, 12, 13, 14, 15] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.1618 | Valid Loss 1.7974 | Train Acc 93.7810| Valid Acc 48.7567
TestSub: S[11, 12, 13, 14, 15] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.1577 | Valid Loss 1.7986 | Train Acc 93.9786| Valid Acc 50.0213
TestSub: S[11, 12, 13, 14, 15] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.1596 | Valid Loss 1.9082 | Train Acc 93.8795| Valid Acc 47.1304
TestSub: S[11, 12, 13, 14, 15] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.1633 | Valid Loss 1.7832 | Train Acc 93.7567| Valid Acc 48.2429
TestSub: S[11, 12, 13, 14, 15] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.1802 | Valid Loss 1.8126 | Train Acc 92.9493| Valid Acc 45.8202
TestSub: S[11, 12, 13, 14, 15] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.1707 | Valid Loss 1.7887 | Train Acc 93.4223| Valid Acc 47.4799
TestSub: S[11, 12, 13, 14, 15] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.1655 | Valid Loss 1.7583 | Train Acc 93.6594| Valid Acc 49.4224
TestSub: S[11, 12, 13, 14, 15] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.1621 | Valid Loss 1.6993 | Train Acc 93.7719| Valid Acc 49.7994
TestSub: S[11, 12, 13, 14, 15] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.1609 | Valid Loss 1.7754 | Train Acc 93.8412| Valid Acc 51.7327
TestSub: S[11, 12, 13, 14, 15] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.1548 | Valid Loss 1.7137 | Train Acc 94.1057| Valid Acc 50.6201
TestSub: S[11, 12, 13, 14, 15] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.1538 | Valid Loss 1.7969 | Train Acc 94.1124| Valid Acc 50.4651
TestSub: S[11, 12, 13, 14, 15] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.1522 | Valid Loss 1.8640 | Train Acc 94.1762| Valid Acc 49.7538
TestSub: S[11, 12, 13, 14, 15] Epoch 33 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0743 | Train Loss 0.1485 | Valid Loss 1.7918 | Train Acc 94.3002| Valid Acc 53.0247
TestSub: S[11, 12, 13, 14, 15] Epoch 34 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0748 | Train Loss 0.1482 | Valid Loss 1.6640 | Train Acc 94.3647| Valid Acc 50.6931
TestSub: S[11, 12, 13, 14, 15] Epoch 35 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0753 | Train Loss 0.1474 | Valid Loss 1.8161 | Train Acc 94.4206| Valid Acc 51.9121
Early stopping triggered after 35 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[11, 12, 13, 14, 15] |Best epoch:5 | Test Loss:1.7212 | Best Acc 53.2132 | Savemodel Acc 53.2132
--------------------------------------------------

========== Fold 4 ==========
Train IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Val IDs:   [16, 17, 18, 19, 20]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[16, 17, 18, 19, 20] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5337 | Valid Loss 1.1017 | Train Acc 70.7034| Valid Acc 51.6385
Validation acc increase (0.000000 --> 51.638497) in epoch (1).  Saving model ...
TestSub: S[16, 17, 18, 19, 20] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.3495 | Valid Loss 1.4098 | Train Acc 84.1288| Valid Acc 49.1671
TestSub: S[16, 17, 18, 19, 20] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.2745 | Valid Loss 1.4651 | Train Acc 88.3062| Valid Acc 52.7207
Validation acc increase (51.638497 --> 52.720696) in epoch (3).  Saving model ...
TestSub: S[16, 17, 18, 19, 20] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2305 | Valid Loss 2.0554 | Train Acc 90.4973| Valid Acc 51.8209
TestSub: S[16, 17, 18, 19, 20] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.1995 | Valid Loss 1.9258 | Train Acc 91.8890| Valid Acc 50.9211
TestSub: S[16, 17, 18, 19, 20] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.1725 | Valid Loss 2.0855 | Train Acc 93.2022| Valid Acc 49.6535
TestSub: S[16, 17, 18, 19, 20] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.1520 | Valid Loss 1.8889 | Train Acc 94.1026| Valid Acc 52.3134
TestSub: S[16, 17, 18, 19, 20] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.1389 | Valid Loss 2.2099 | Train Acc 94.6650| Valid Acc 49.2309
TestSub: S[16, 17, 18, 19, 20] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1275 | Valid Loss 1.9248 | Train Acc 95.1192| Valid Acc 51.3223
TestSub: S[16, 17, 18, 19, 20] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1204 | Valid Loss 2.1964 | Train Acc 95.4159| Valid Acc 51.9516
TestSub: S[16, 17, 18, 19, 20] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1124 | Valid Loss 2.2971 | Train Acc 95.7375| Valid Acc 52.8514
Validation acc increase (52.720696 --> 52.851411) in epoch (11).  Saving model ...
TestSub: S[16, 17, 18, 19, 20] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1100 | Valid Loss 2.0809 | Train Acc 95.8536| Valid Acc 51.6902
TestSub: S[16, 17, 18, 19, 20] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1066 | Valid Loss 2.1683 | Train Acc 95.9703| Valid Acc 51.2251
TestSub: S[16, 17, 18, 19, 20] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1039 | Valid Loss 2.5494 | Train Acc 96.1503| Valid Acc 49.3555
TestSub: S[16, 17, 18, 19, 20] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1007 | Valid Loss 2.2382 | Train Acc 96.2044| Valid Acc 50.0821
TestSub: S[16, 17, 18, 19, 20] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.0991 | Valid Loss 2.3172 | Train Acc 96.3679| Valid Acc 52.0732
TestSub: S[16, 17, 18, 19, 20] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.0958 | Valid Loss 2.1107 | Train Acc 96.4287| Valid Acc 48.9421
TestSub: S[16, 17, 18, 19, 20] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.0938 | Valid Loss 2.2688 | Train Acc 96.5382| Valid Acc 50.3891
TestSub: S[16, 17, 18, 19, 20] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.0979 | Valid Loss 1.8928 | Train Acc 96.3679| Valid Acc 53.3196
Validation acc increase (52.851411 --> 53.319553) in epoch (19).  Saving model ...
TestSub: S[16, 17, 18, 19, 20] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.1095 | Valid Loss 2.0366 | Train Acc 95.9351| Valid Acc 51.4227
TestSub: S[16, 17, 18, 19, 20] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.1019 | Valid Loss 2.2422 | Train Acc 96.1904| Valid Acc 50.9576
TestSub: S[16, 17, 18, 19, 20] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.0983 | Valid Loss 2.1085 | Train Acc 96.3959| Valid Acc 49.9818
TestSub: S[16, 17, 18, 19, 20] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.0957 | Valid Loss 2.0795 | Train Acc 96.4670| Valid Acc 52.5961
TestSub: S[16, 17, 18, 19, 20] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.0932 | Valid Loss 2.1803 | Train Acc 96.5339| Valid Acc 52.4319
TestSub: S[16, 17, 18, 19, 20] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.0905 | Valid Loss 2.3036 | Train Acc 96.6956| Valid Acc 53.0338
TestSub: S[16, 17, 18, 19, 20] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.0898 | Valid Loss 2.1940 | Train Acc 96.6877| Valid Acc 50.4225
TestSub: S[16, 17, 18, 19, 20] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.0893 | Valid Loss 2.1903 | Train Acc 96.7741| Valid Acc 51.9972
TestSub: S[16, 17, 18, 19, 20] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.0877 | Valid Loss 2.2828 | Train Acc 96.8306| Valid Acc 54.3288
Validation acc increase (53.319553 --> 54.328794) in epoch (28).  Saving model ...
TestSub: S[16, 17, 18, 19, 20] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.0877 | Valid Loss 2.2163 | Train Acc 96.7893| Valid Acc 49.9331
TestSub: S[16, 17, 18, 19, 20] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.0870 | Valid Loss 2.0877 | Train Acc 96.8245| Valid Acc 51.4622
TestSub: S[16, 17, 18, 19, 20] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.0850 | Valid Loss 2.2123 | Train Acc 96.9103| Valid Acc 53.1676
TestSub: S[16, 17, 18, 19, 20] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.0829 | Valid Loss 2.3830 | Train Acc 97.0361| Valid Acc 51.8513
TestSub: S[16, 17, 18, 19, 20] Epoch 33 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0743 | Train Loss 0.0835 | Valid Loss 2.2487 | Train Acc 96.9729| Valid Acc 52.7481
TestSub: S[16, 17, 18, 19, 20] Epoch 34 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0748 | Train Loss 0.0826 | Valid Loss 2.1820 | Train Acc 96.9808| Valid Acc 52.7146
TestSub: S[16, 17, 18, 19, 20] Epoch 35 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0753 | Train Loss 0.0826 | Valid Loss 2.1787 | Train Acc 97.0154| Valid Acc 52.8940
TestSub: S[16, 17, 18, 19, 20] Epoch 36 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0757 | Train Loss 0.0803 | Valid Loss 2.0762 | Train Acc 97.0884| Valid Acc 51.6294
TestSub: S[16, 17, 18, 19, 20] Epoch 37 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0761 | Train Loss 0.0812 | Valid Loss 2.4264 | Train Acc 97.0544| Valid Acc 49.5592
TestSub: S[16, 17, 18, 19, 20] Epoch 38 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0765 | Train Loss 0.0807 | Valid Loss 2.3605 | Train Acc 97.0908| Valid Acc 50.8968
TestSub: S[16, 17, 18, 19, 20] Epoch 39 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0768 | Train Loss 0.0799 | Valid Loss 2.4140 | Train Acc 97.0945| Valid Acc 51.4804
TestSub: S[16, 17, 18, 19, 20] Epoch 40 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0771 | Train Loss 0.0806 | Valid Loss 2.3531 | Train Acc 97.1152| Valid Acc 50.2189
TestSub: S[16, 17, 18, 19, 20] Epoch 41 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0774 | Train Loss 0.0788 | Valid Loss 2.2899 | Train Acc 97.1936| Valid Acc 52.3954
TestSub: S[16, 17, 18, 19, 20] Epoch 42 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0776 | Train Loss 0.0781 | Valid Loss 2.1684 | Train Acc 97.2307| Valid Acc 51.2403
TestSub: S[16, 17, 18, 19, 20] Epoch 43 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0779 | Train Loss 0.0785 | Valid Loss 2.2372 | Train Acc 97.1723| Valid Acc 51.6385
TestSub: S[16, 17, 18, 19, 20] Epoch 44 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0781 | Train Loss 0.0773 | Valid Loss 2.2363 | Train Acc 97.2282| Valid Acc 52.2708
TestSub: S[16, 17, 18, 19, 20] Epoch 45 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0782 | Train Loss 0.0781 | Valid Loss 2.3368 | Train Acc 97.1960| Valid Acc 48.8570
TestSub: S[16, 17, 18, 19, 20] Epoch 46 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0784 | Train Loss 0.0785 | Valid Loss 2.5506 | Train Acc 97.1389| Valid Acc 51.6659
TestSub: S[16, 17, 18, 19, 20] Epoch 47 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0786 | Train Loss 0.0768 | Valid Loss 2.3891 | Train Acc 97.2471| Valid Acc 49.5015
TestSub: S[16, 17, 18, 19, 20] Epoch 48 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0787 | Train Loss 0.0775 | Valid Loss 2.4507 | Train Acc 97.2471| Valid Acc 51.9455
TestSub: S[16, 17, 18, 19, 20] Epoch 49 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0788 | Train Loss 0.0759 | Valid Loss 2.2320 | Train Acc 97.2319| Valid Acc 51.3254
TestSub: S[16, 17, 18, 19, 20] Epoch 50 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0789 | Train Loss 0.0762 | Valid Loss 2.2680 | Train Acc 97.2550| Valid Acc 51.3132
TestSub: S[16, 17, 18, 19, 20] Epoch 51 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0790 | Train Loss 0.0768 | Valid Loss 2.2673 | Train Acc 97.2684| Valid Acc 52.4137
TestSub: S[16, 17, 18, 19, 20] Epoch 52 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0791 | Train Loss 0.0754 | Valid Loss 2.2396 | Train Acc 97.2696| Valid Acc 53.0186
TestSub: S[16, 17, 18, 19, 20] Epoch 53 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0792 | Train Loss 0.0765 | Valid Loss 2.1546 | Train Acc 97.2459| Valid Acc 52.9274
TestSub: S[16, 17, 18, 19, 20] Epoch 54 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0749 | Valid Loss 2.3998 | Train Acc 97.3243| Valid Acc 49.7355
TestSub: S[16, 17, 18, 19, 20] Epoch 55 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0746 | Valid Loss 2.1125 | Train Acc 97.3340| Valid Acc 52.7572
TestSub: S[16, 17, 18, 19, 20] Epoch 56 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0794 | Train Loss 0.0736 | Valid Loss 2.2888 | Train Acc 97.3279| Valid Acc 51.0731
TestSub: S[16, 17, 18, 19, 20] Epoch 57 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0745 | Valid Loss 2.1678 | Train Acc 97.3340| Valid Acc 51.7753
TestSub: S[16, 17, 18, 19, 20] Epoch 58 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0750 | Valid Loss 2.4210 | Train Acc 97.3152| Valid Acc 50.7995
Early stopping triggered after 58 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[16, 17, 18, 19, 20] |Best epoch:28 | Test Loss:2.2828 | Best Acc 54.3288 | Savemodel Acc 54.3288
--------------------------------------------------

========== Fold 5 ==========
Train IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 30]
Val IDs:   [21, 22, 23, 24, 25]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[21, 22, 23, 24, 25] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5414 | Valid Loss 1.4232 | Train Acc 69.4303| Valid Acc 50.9545
Validation acc increase (0.000000 --> 50.954523) in epoch (1).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.3845 | Valid Loss 1.3022 | Train Acc 81.1631| Valid Acc 49.4832
TestSub: S[21, 22, 23, 24, 25] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.3122 | Valid Loss 1.4425 | Train Acc 86.0694| Valid Acc 50.8542
TestSub: S[21, 22, 23, 24, 25] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2490 | Valid Loss 1.6347 | Train Acc 89.4011| Valid Acc 48.2977
TestSub: S[21, 22, 23, 24, 25] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.2054 | Valid Loss 1.8706 | Train Acc 91.6178| Valid Acc 46.0755
TestSub: S[21, 22, 23, 24, 25] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.1716 | Valid Loss 2.0982 | Train Acc 93.2740| Valid Acc 48.5682
TestSub: S[21, 22, 23, 24, 25] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.1485 | Valid Loss 1.9586 | Train Acc 94.2802| Valid Acc 51.3801
Validation acc increase (50.954523 --> 51.380107) in epoch (7).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.1362 | Valid Loss 2.0842 | Train Acc 94.7805| Valid Acc 49.7355
TestSub: S[21, 22, 23, 24, 25] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1271 | Valid Loss 1.9231 | Train Acc 95.2274| Valid Acc 52.7602
Validation acc increase (51.380107 --> 52.760214) in epoch (9).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1194 | Valid Loss 2.3784 | Train Acc 95.5843| Valid Acc 49.3981
TestSub: S[21, 22, 23, 24, 25] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1159 | Valid Loss 2.1689 | Train Acc 95.6901| Valid Acc 50.0152
TestSub: S[21, 22, 23, 24, 25] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1133 | Valid Loss 2.3810 | Train Acc 95.7977| Valid Acc 52.0002
TestSub: S[21, 22, 23, 24, 25] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1104 | Valid Loss 2.0102 | Train Acc 95.9576| Valid Acc 51.1187
TestSub: S[21, 22, 23, 24, 25] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1053 | Valid Loss 2.0018 | Train Acc 96.1351| Valid Acc 51.5534
TestSub: S[21, 22, 23, 24, 25] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1066 | Valid Loss 2.0403 | Train Acc 96.0761| Valid Acc 48.7749
TestSub: S[21, 22, 23, 24, 25] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.1023 | Valid Loss 2.2365 | Train Acc 96.2713| Valid Acc 48.1821
TestSub: S[21, 22, 23, 24, 25] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.1022 | Valid Loss 2.0387 | Train Acc 96.2859| Valid Acc 51.3041
TestSub: S[21, 22, 23, 24, 25] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.0999 | Valid Loss 2.1902 | Train Acc 96.3904| Valid Acc 47.5438
TestSub: S[21, 22, 23, 24, 25] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.1008 | Valid Loss 2.2395 | Train Acc 96.3041| Valid Acc 50.0851
TestSub: S[21, 22, 23, 24, 25] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.0996 | Valid Loss 2.3533 | Train Acc 96.3947| Valid Acc 51.5929
TestSub: S[21, 22, 23, 24, 25] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.0968 | Valid Loss 2.1494 | Train Acc 96.5254| Valid Acc 50.7144
TestSub: S[21, 22, 23, 24, 25] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.0963 | Valid Loss 2.1230 | Train Acc 96.5272| Valid Acc 50.0821
TestSub: S[21, 22, 23, 24, 25] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.0951 | Valid Loss 2.2252 | Train Acc 96.5169| Valid Acc 49.9483
TestSub: S[21, 22, 23, 24, 25] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.0964 | Valid Loss 2.4576 | Train Acc 96.5406| Valid Acc 49.9909
TestSub: S[21, 22, 23, 24, 25] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.0962 | Valid Loss 2.3503 | Train Acc 96.5303| Valid Acc 49.0090
TestSub: S[21, 22, 23, 24, 25] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.0962 | Valid Loss 2.2356 | Train Acc 96.5576| Valid Acc 50.1854
TestSub: S[21, 22, 23, 24, 25] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.0941 | Valid Loss 2.1701 | Train Acc 96.5643| Valid Acc 51.0032
TestSub: S[21, 22, 23, 24, 25] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.0945 | Valid Loss 2.3101 | Train Acc 96.5905| Valid Acc 49.6291
TestSub: S[21, 22, 23, 24, 25] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.0927 | Valid Loss 2.5183 | Train Acc 96.6458| Valid Acc 49.6504
TestSub: S[21, 22, 23, 24, 25] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.0935 | Valid Loss 2.4283 | Train Acc 96.6160| Valid Acc 50.4377
TestSub: S[21, 22, 23, 24, 25] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.0921 | Valid Loss 2.3425 | Train Acc 96.6750| Valid Acc 49.8936
TestSub: S[21, 22, 23, 24, 25] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.0912 | Valid Loss 2.1184 | Train Acc 96.7339| Valid Acc 52.4988
TestSub: S[21, 22, 23, 24, 25] Epoch 33 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0743 | Train Loss 0.0915 | Valid Loss 2.3573 | Train Acc 96.7291| Valid Acc 50.8876
TestSub: S[21, 22, 23, 24, 25] Epoch 34 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0748 | Train Loss 0.0918 | Valid Loss 2.2379 | Train Acc 96.6397| Valid Acc 51.8847
TestSub: S[21, 22, 23, 24, 25] Epoch 35 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0753 | Train Loss 0.0893 | Valid Loss 2.3400 | Train Acc 96.7680| Valid Acc 53.2557
Validation acc increase (52.760214 --> 53.255715) in epoch (35).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 36 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0757 | Train Loss 0.0894 | Valid Loss 2.2297 | Train Acc 96.8014| Valid Acc 52.0519
TestSub: S[21, 22, 23, 24, 25] Epoch 37 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0761 | Train Loss 0.0886 | Valid Loss 2.2726 | Train Acc 96.7881| Valid Acc 51.2129
TestSub: S[21, 22, 23, 24, 25] Epoch 38 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0765 | Train Loss 0.0869 | Valid Loss 2.1362 | Train Acc 96.8470| Valid Acc 51.8513
TestSub: S[21, 22, 23, 24, 25] Epoch 39 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0768 | Train Loss 0.0891 | Valid Loss 2.5178 | Train Acc 96.8245| Valid Acc 49.4072
TestSub: S[21, 22, 23, 24, 25] Epoch 40 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0771 | Train Loss 0.0880 | Valid Loss 2.3750 | Train Acc 96.8027| Valid Acc 52.1401
TestSub: S[21, 22, 23, 24, 25] Epoch 41 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0774 | Train Loss 0.0864 | Valid Loss 2.2531 | Train Acc 96.8671| Valid Acc 51.7267
TestSub: S[21, 22, 23, 24, 25] Epoch 42 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0776 | Train Loss 0.0874 | Valid Loss 2.1261 | Train Acc 96.8543| Valid Acc 53.9336
Validation acc increase (53.255715 --> 53.933609) in epoch (42).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 43 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0779 | Train Loss 0.0859 | Valid Loss 2.2552 | Train Acc 96.8920| Valid Acc 53.3621
TestSub: S[21, 22, 23, 24, 25] Epoch 44 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0781 | Train Loss 0.0851 | Valid Loss 2.4300 | Train Acc 96.9559| Valid Acc 51.3467
TestSub: S[21, 22, 23, 24, 25] Epoch 45 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0782 | Train Loss 0.0866 | Valid Loss 2.1650 | Train Acc 96.9078| Valid Acc 53.1280
TestSub: S[21, 22, 23, 24, 25] Epoch 46 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0784 | Train Loss 0.0851 | Valid Loss 2.2703 | Train Acc 97.0112| Valid Acc 50.0547
TestSub: S[21, 22, 23, 24, 25] Epoch 47 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0786 | Train Loss 0.0834 | Valid Loss 2.2577 | Train Acc 97.0063| Valid Acc 51.7966
TestSub: S[21, 22, 23, 24, 25] Epoch 48 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0787 | Train Loss 0.0836 | Valid Loss 2.0702 | Train Acc 97.0221| Valid Acc 52.4228
TestSub: S[21, 22, 23, 24, 25] Epoch 49 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0788 | Train Loss 0.0841 | Valid Loss 2.2013 | Train Acc 97.0124| Valid Acc 53.9458
Validation acc increase (53.933609 --> 53.945768) in epoch (49).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 50 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0789 | Train Loss 0.0836 | Valid Loss 2.0557 | Train Acc 97.0392| Valid Acc 51.7996
TestSub: S[21, 22, 23, 24, 25] Epoch 51 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0790 | Train Loss 0.0821 | Valid Loss 2.3955 | Train Acc 97.0464| Valid Acc 52.9943
TestSub: S[21, 22, 23, 24, 25] Epoch 52 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0791 | Train Loss 0.0836 | Valid Loss 2.3215 | Train Acc 97.0045| Valid Acc 52.4410
TestSub: S[21, 22, 23, 24, 25] Epoch 53 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0792 | Train Loss 0.0826 | Valid Loss 2.2071 | Train Acc 97.0446| Valid Acc 54.4929
Validation acc increase (53.945768 --> 54.492947) in epoch (53).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 54 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0824 | Valid Loss 2.2114 | Train Acc 97.0708| Valid Acc 54.2619
TestSub: S[21, 22, 23, 24, 25] Epoch 55 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0801 | Valid Loss 2.1559 | Train Acc 97.1370| Valid Acc 53.2436
TestSub: S[21, 22, 23, 24, 25] Epoch 56 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0794 | Train Loss 0.0811 | Valid Loss 2.2401 | Train Acc 97.0951| Valid Acc 52.6873
TestSub: S[21, 22, 23, 24, 25] Epoch 57 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0806 | Valid Loss 2.3054 | Train Acc 97.1389| Valid Acc 51.1673
TestSub: S[21, 22, 23, 24, 25] Epoch 58 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0824 | Valid Loss 2.2287 | Train Acc 97.1261| Valid Acc 53.5141
TestSub: S[21, 22, 23, 24, 25] Epoch 59 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0811 | Valid Loss 2.2667 | Train Acc 97.1182| Valid Acc 51.7601
TestSub: S[21, 22, 23, 24, 25] Epoch 60 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0814 | Valid Loss 2.1576 | Train Acc 97.0349| Valid Acc 52.2799
TestSub: S[21, 22, 23, 24, 25] Epoch 61 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0796 | Valid Loss 1.9578 | Train Acc 97.1954| Valid Acc 52.9456
TestSub: S[21, 22, 23, 24, 25] Epoch 62 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0810 | Valid Loss 2.2158 | Train Acc 97.1437| Valid Acc 51.0609
TestSub: S[21, 22, 23, 24, 25] Epoch 63 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0794 | Valid Loss 2.1613 | Train Acc 97.1759| Valid Acc 52.1401
TestSub: S[21, 22, 23, 24, 25] Epoch 64 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0782 | Valid Loss 2.2863 | Train Acc 97.2276| Valid Acc 51.5625
TestSub: S[21, 22, 23, 24, 25] Epoch 65 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0802 | Valid Loss 2.0659 | Train Acc 97.2088| Valid Acc 55.5265
Validation acc increase (54.492947 --> 55.526508) in epoch (65).  Saving model ...
TestSub: S[21, 22, 23, 24, 25] Epoch 66 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0785 | Valid Loss 2.0752 | Train Acc 97.2313| Valid Acc 53.4411
TestSub: S[21, 22, 23, 24, 25] Epoch 67 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0789 | Valid Loss 2.1956 | Train Acc 97.1972| Valid Acc 53.1524
TestSub: S[21, 22, 23, 24, 25] Epoch 68 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0785 | Valid Loss 2.2858 | Train Acc 97.2203| Valid Acc 50.4104
TestSub: S[21, 22, 23, 24, 25] Epoch 69 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0790 | Valid Loss 2.2184 | Train Acc 97.1887| Valid Acc 52.1310
TestSub: S[21, 22, 23, 24, 25] Epoch 70 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0792 | Valid Loss 1.9571 | Train Acc 97.1626| Valid Acc 51.4409
TestSub: S[21, 22, 23, 24, 25] Epoch 71 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0784 | Valid Loss 2.4337 | Train Acc 97.2203| Valid Acc 51.2524
TestSub: S[21, 22, 23, 24, 25] Epoch 72 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0774 | Valid Loss 2.1384 | Train Acc 97.2854| Valid Acc 51.6780
TestSub: S[21, 22, 23, 24, 25] Epoch 73 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0784 | Valid Loss 2.1437 | Train Acc 97.2270| Valid Acc 50.6262
TestSub: S[21, 22, 23, 24, 25] Epoch 74 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0779 | Valid Loss 2.2256 | Train Acc 97.2367| Valid Acc 49.3069
TestSub: S[21, 22, 23, 24, 25] Epoch 75 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0787 | Valid Loss 2.3522 | Train Acc 97.2167| Valid Acc 50.2736
TestSub: S[21, 22, 23, 24, 25] Epoch 76 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0775 | Valid Loss 2.2967 | Train Acc 97.2793| Valid Acc 51.4257
TestSub: S[21, 22, 23, 24, 25] Epoch 77 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0758 | Valid Loss 2.3955 | Train Acc 97.3589| Valid Acc 50.1338
TestSub: S[21, 22, 23, 24, 25] Epoch 78 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0761 | Valid Loss 2.2435 | Train Acc 97.3845| Valid Acc 49.5714
TestSub: S[21, 22, 23, 24, 25] Epoch 79 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0778 | Valid Loss 2.2111 | Train Acc 97.2647| Valid Acc 52.0033
TestSub: S[21, 22, 23, 24, 25] Epoch 80 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0781 | Valid Loss 2.4927 | Train Acc 97.2380| Valid Acc 49.7690
TestSub: S[21, 22, 23, 24, 25] Epoch 81 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0756 | Valid Loss 2.3740 | Train Acc 97.3529| Valid Acc 50.3800
TestSub: S[21, 22, 23, 24, 25] Epoch 82 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0757 | Valid Loss 2.2217 | Train Acc 97.3328| Valid Acc 52.6082
TestSub: S[21, 22, 23, 24, 25] Epoch 83 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0754 | Valid Loss 2.3135 | Train Acc 97.3267| Valid Acc 52.4623
TestSub: S[21, 22, 23, 24, 25] Epoch 84 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0762 | Valid Loss 2.3104 | Train Acc 97.3310| Valid Acc 51.6811
TestSub: S[21, 22, 23, 24, 25] Epoch 85 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0759 | Valid Loss 2.3105 | Train Acc 97.3085| Valid Acc 54.2528
TestSub: S[21, 22, 23, 24, 25] Epoch 86 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0759 | Valid Loss 2.1740 | Train Acc 97.3237| Valid Acc 53.9883
TestSub: S[21, 22, 23, 24, 25] Epoch 87 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0775 | Valid Loss 2.0800 | Train Acc 97.2495| Valid Acc 53.3743
TestSub: S[21, 22, 23, 24, 25] Epoch 88 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0753 | Valid Loss 2.2747 | Train Acc 97.3577| Valid Acc 50.9484
TestSub: S[21, 22, 23, 24, 25] Epoch 89 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0761 | Valid Loss 2.1618 | Train Acc 97.3200| Valid Acc 52.2830
TestSub: S[21, 22, 23, 24, 25] Epoch 90 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0770 | Valid Loss 2.4666 | Train Acc 97.3164| Valid Acc 49.4285
TestSub: S[21, 22, 23, 24, 25] Epoch 91 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0766 | Valid Loss 2.3933 | Train Acc 97.3067| Valid Acc 49.1701
TestSub: S[21, 22, 23, 24, 25] Epoch 92 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0754 | Valid Loss 2.3618 | Train Acc 97.3255| Valid Acc 50.2250
TestSub: S[21, 22, 23, 24, 25] Epoch 93 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0766 | Valid Loss 2.2518 | Train Acc 97.2963| Valid Acc 51.3284
TestSub: S[21, 22, 23, 24, 25] Epoch 94 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0738 | Valid Loss 2.3813 | Train Acc 97.4331| Valid Acc 52.9000
TestSub: S[21, 22, 23, 24, 25] Epoch 95 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0800 | Train Loss 0.0753 | Valid Loss 2.0134 | Train Acc 97.3504| Valid Acc 52.1249
Early stopping triggered after 95 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[21, 22, 23, 24, 25] |Best epoch:65 | Test Loss:2.0659 | Best Acc 55.5265 | Savemodel Acc 55.5265
--------------------------------------------------

========== Fold 6 ==========
Train IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
Val IDs:   [26, 27, 28, 29, 30]
train_data_shape(164500, 32, 128),val_data_shape(32900, 32, 128)
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
DARNet                                   --
├─TokenEmbedding: 1-1                    --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  576
│    │    └─BatchNorm2d: 3-2             128
│    │    └─GELU: 3-3                    --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-4                  32,784
│    │    └─BatchNorm2d: 3-5             32
│    │    └─GELU: 3-6                    --
│    └─PositionalEmbedding: 2-3          --
├─Flatten: 1-2                           --
├─Linear: 1-3                            18
├─AttnRefine: 1-4                        --
│    └─MyAttention: 2-4                  --
│    │    └─Attention: 3-7               800
│    └─Refine: 2-5                       --
│    │    └─Conv1d: 3-8                  784
│    │    └─BatchNorm1d: 3-9             32
│    │    └─ELU: 3-10                    --
│    │    └─MaxPool1d: 3-11              --
│    └─AdaptiveAvgPool1d: 2-6            --
│    └─Linear: 2-7                       68
│    └─Flatten: 2-8                      --
├─AttnRefine: 1-5                        --
│    └─MyAttention: 2-9                  --
│    │    └─Attention: 3-12              800
│    └─Refine: 2-10                      --
│    │    └─Conv1d: 3-13                 784
│    │    └─BatchNorm1d: 3-14            32
│    │    └─ELU: 3-15                    --
│    │    └─MaxPool1d: 3-16              --
│    └─AdaptiveAvgPool1d: 2-11           --
│    └─Linear: 2-12                      68
│    └─Flatten: 2-13                     --
├─GradientReversalLayer: 1-6             --
├─SubjectDiscriminator: 1-7              --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-17                 2,304
│    │    └─ReLU: 3-18                   --
│    │    └─Dropout: 3-19                --
│    │    └─Linear: 3-20                 32,896
│    │    └─ReLU: 3-21                   --
│    │    └─Dropout: 3-22                --
│    │    └─Linear: 3-23                 3,870
=================================================================
Total params: 75,976
Trainable params: 75,976
Non-trainable params: 0
=================================================================
TestSub: S[26, 27, 28, 29, 30] Epoch  1 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0040 | Train Loss 0.5133 | Valid Loss 1.5097 | Train Acc 72.7857| Valid Acc 48.5773
Validation acc increase (0.000000 --> 48.577335) in epoch (1).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch  2 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0080 | Train Loss 0.3328 | Valid Loss 1.7891 | Train Acc 85.3028| Valid Acc 50.2918
Validation acc increase (48.577335 --> 50.291829) in epoch (2).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch  3 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0119 | Train Loss 0.2643 | Valid Loss 1.9076 | Train Acc 88.9944| Valid Acc 49.2643
TestSub: S[26, 27, 28, 29, 30] Epoch  4 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0158 | Train Loss 0.2158 | Valid Loss 1.9997 | Train Acc 91.3704| Valid Acc 46.6075
TestSub: S[26, 27, 28, 29, 30] Epoch  5 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0196 | Train Loss 0.1892 | Valid Loss 2.1148 | Train Acc 92.6283| Valid Acc 48.9269
TestSub: S[26, 27, 28, 29, 30] Epoch  6 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0233 | Train Loss 0.1729 | Valid Loss 2.3350 | Train Acc 93.2679| Valid Acc 50.6293
Validation acc increase (50.291829 --> 50.629256) in epoch (6).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch  7 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0269 | Train Loss 0.1555 | Valid Loss 2.2374 | Train Acc 94.0291| Valid Acc 48.7141
TestSub: S[26, 27, 28, 29, 30] Epoch  8 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0304 | Train Loss 0.1671 | Valid Loss 1.7907 | Train Acc 93.5269| Valid Acc 50.9089
Validation acc increase (50.629256 --> 50.908925) in epoch (8).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch  9 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0338 | Train Loss 0.1618 | Valid Loss 2.0952 | Train Acc 93.7482| Valid Acc 49.5045
TestSub: S[26, 27, 28, 29, 30] Epoch 10 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0370 | Train Loss 0.1298 | Valid Loss 2.2988 | Train Acc 95.0772| Valid Acc 50.6901
TestSub: S[26, 27, 28, 29, 30] Epoch 11 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0400 | Train Loss 0.1182 | Valid Loss 2.4053 | Train Acc 95.6043| Valid Acc 49.2643
TestSub: S[26, 27, 28, 29, 30] Epoch 12 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0430 | Train Loss 0.1130 | Valid Loss 2.1890 | Train Acc 95.8001| Valid Acc 50.9363
Validation acc increase (50.908925 --> 50.936284) in epoch (12).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch 13 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0457 | Train Loss 0.1156 | Valid Loss 1.9567 | Train Acc 95.7198| Valid Acc 52.8605
Validation acc increase (50.936284 --> 52.860530) in epoch (13).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch 14 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0483 | Train Loss 0.1202 | Valid Loss 2.4696 | Train Acc 95.5344| Valid Acc 49.8358
TestSub: S[26, 27, 28, 29, 30] Epoch 15 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0508 | Train Loss 0.1103 | Valid Loss 2.6032 | Train Acc 95.8834| Valid Acc 47.7170
TestSub: S[26, 27, 28, 29, 30] Epoch 16 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0531 | Train Loss 0.1071 | Valid Loss 2.4558 | Train Acc 96.0378| Valid Acc 51.0822
TestSub: S[26, 27, 28, 29, 30] Epoch 17 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0553 | Train Loss 0.1032 | Valid Loss 2.2909 | Train Acc 96.2184| Valid Acc 51.2889
TestSub: S[26, 27, 28, 29, 30] Epoch 18 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0573 | Train Loss 0.1010 | Valid Loss 2.4452 | Train Acc 96.3333| Valid Acc 50.6293
TestSub: S[26, 27, 28, 29, 30] Epoch 19 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0592 | Train Loss 0.0972 | Valid Loss 2.2714 | Train Acc 96.4762| Valid Acc 49.7933
TestSub: S[26, 27, 28, 29, 30] Epoch 20 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0609 | Train Loss 0.0962 | Valid Loss 2.9129 | Train Acc 96.4920| Valid Acc 52.3438
TestSub: S[26, 27, 28, 29, 30] Epoch 21 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0625 | Train Loss 0.0954 | Valid Loss 2.3540 | Train Acc 96.5108| Valid Acc 51.9303
TestSub: S[26, 27, 28, 29, 30] Epoch 22 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0640 | Train Loss 0.0930 | Valid Loss 2.2377 | Train Acc 96.6440| Valid Acc 52.3255
TestSub: S[26, 27, 28, 29, 30] Epoch 23 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0654 | Train Loss 0.0922 | Valid Loss 2.2328 | Train Acc 96.6117| Valid Acc 53.4290
Validation acc increase (52.860530 --> 53.428988) in epoch (23).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch 24 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0667 | Train Loss 0.0917 | Valid Loss 2.3271 | Train Acc 96.6580| Valid Acc 49.2157
TestSub: S[26, 27, 28, 29, 30] Epoch 25 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0679 | Train Loss 0.0908 | Valid Loss 2.2462 | Train Acc 96.7029| Valid Acc 50.1368
TestSub: S[26, 27, 28, 29, 30] Epoch 26 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0689 | Train Loss 0.0886 | Valid Loss 2.1327 | Train Acc 96.8008| Valid Acc 52.6690
TestSub: S[26, 27, 28, 29, 30] Epoch 27 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0699 | Train Loss 0.0895 | Valid Loss 2.3734 | Train Acc 96.7406| Valid Acc 50.7448
TestSub: S[26, 27, 28, 29, 30] Epoch 28 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0708 | Train Loss 0.0868 | Valid Loss 2.7094 | Train Acc 96.8586| Valid Acc 49.5167
TestSub: S[26, 27, 28, 29, 30] Epoch 29 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0717 | Train Loss 0.0875 | Valid Loss 2.3952 | Train Acc 96.8866| Valid Acc 50.8329
TestSub: S[26, 27, 28, 29, 30] Epoch 30 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0724 | Train Loss 0.0855 | Valid Loss 2.2791 | Train Acc 96.8999| Valid Acc 49.5775
TestSub: S[26, 27, 28, 29, 30] Epoch 31 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0731 | Train Loss 0.0848 | Valid Loss 2.1784 | Train Acc 96.9358| Valid Acc 50.8329
TestSub: S[26, 27, 28, 29, 30] Epoch 32 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0737 | Train Loss 0.0850 | Valid Loss 2.3115 | Train Acc 96.8993| Valid Acc 51.0457
TestSub: S[26, 27, 28, 29, 30] Epoch 33 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0743 | Train Loss 0.0837 | Valid Loss 2.4804 | Train Acc 96.9990| Valid Acc 52.2343
TestSub: S[26, 27, 28, 29, 30] Epoch 34 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0748 | Train Loss 0.0845 | Valid Loss 2.3582 | Train Acc 96.9802| Valid Acc 51.3892
TestSub: S[26, 27, 28, 29, 30] Epoch 35 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0753 | Train Loss 0.0861 | Valid Loss 2.5392 | Train Acc 96.9176| Valid Acc 49.8267
TestSub: S[26, 27, 28, 29, 30] Epoch 36 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0757 | Train Loss 0.0836 | Valid Loss 2.3712 | Train Acc 97.0246| Valid Acc 49.9149
TestSub: S[26, 27, 28, 29, 30] Epoch 37 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0761 | Train Loss 0.0841 | Valid Loss 2.3277 | Train Acc 96.8999| Valid Acc 49.5531
TestSub: S[26, 27, 28, 29, 30] Epoch 38 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0765 | Train Loss 0.0828 | Valid Loss 2.3256 | Train Acc 96.9832| Valid Acc 52.2617
TestSub: S[26, 27, 28, 29, 30] Epoch 39 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0768 | Train Loss 0.0822 | Valid Loss 2.2930 | Train Acc 96.9936| Valid Acc 52.1066
TestSub: S[26, 27, 28, 29, 30] Epoch 40 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0771 | Train Loss 0.0811 | Valid Loss 2.5404 | Train Acc 97.0550| Valid Acc 52.8484
TestSub: S[26, 27, 28, 29, 30] Epoch 41 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0774 | Train Loss 0.0819 | Valid Loss 2.4197 | Train Acc 97.0471| Valid Acc 51.1460
TestSub: S[26, 27, 28, 29, 30] Epoch 42 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0776 | Train Loss 0.0805 | Valid Loss 2.4194 | Train Acc 97.1164| Valid Acc 52.7541
TestSub: S[26, 27, 28, 29, 30] Epoch 43 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0779 | Train Loss 0.0828 | Valid Loss 2.4220 | Train Acc 97.0197| Valid Acc 54.3744
Validation acc increase (53.428988 --> 54.374392) in epoch (43).  Saving model ...
TestSub: S[26, 27, 28, 29, 30] Epoch 44 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0781 | Train Loss 0.0814 | Valid Loss 2.3525 | Train Acc 97.0568| Valid Acc 50.8816
TestSub: S[26, 27, 28, 29, 30] Epoch 45 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0782 | Train Loss 0.0793 | Valid Loss 2.4044 | Train Acc 97.1389| Valid Acc 53.5810
TestSub: S[26, 27, 28, 29, 30] Epoch 46 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0784 | Train Loss 0.0795 | Valid Loss 2.3267 | Train Acc 97.1334| Valid Acc 52.9000
TestSub: S[26, 27, 28, 29, 30] Epoch 47 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0786 | Train Loss 0.0790 | Valid Loss 2.3497 | Train Acc 97.1832| Valid Acc 50.6718
TestSub: S[26, 27, 28, 29, 30] Epoch 48 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0787 | Train Loss 0.0801 | Valid Loss 2.2966 | Train Acc 97.1237| Valid Acc 50.9272
TestSub: S[26, 27, 28, 29, 30] Epoch 49 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0788 | Train Loss 0.0786 | Valid Loss 2.2934 | Train Acc 97.1705| Valid Acc 53.3986
TestSub: S[26, 27, 28, 29, 30] Epoch 50 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0789 | Train Loss 0.0800 | Valid Loss 2.3100 | Train Acc 97.1109| Valid Acc 51.7206
TestSub: S[26, 27, 28, 29, 30] Epoch 51 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0790 | Train Loss 0.0791 | Valid Loss 2.2844 | Train Acc 97.1607| Valid Acc 51.5686
TestSub: S[26, 27, 28, 29, 30] Epoch 52 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0791 | Train Loss 0.0789 | Valid Loss 2.2671 | Train Acc 97.1766| Valid Acc 52.8818
TestSub: S[26, 27, 28, 29, 30] Epoch 53 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0792 | Train Loss 0.0780 | Valid Loss 2.2495 | Train Acc 97.2076| Valid Acc 51.0609
TestSub: S[26, 27, 28, 29, 30] Epoch 54 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0781 | Valid Loss 2.6717 | Train Acc 97.2264| Valid Acc 51.6811
TestSub: S[26, 27, 28, 29, 30] Epoch 55 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0793 | Train Loss 0.0785 | Valid Loss 2.3269 | Train Acc 97.1790| Valid Acc 53.5688
TestSub: S[26, 27, 28, 29, 30] Epoch 56 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0794 | Train Loss 0.0763 | Valid Loss 2.3683 | Train Acc 97.3000| Valid Acc 52.0489
TestSub: S[26, 27, 28, 29, 30] Epoch 57 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0778 | Valid Loss 2.2661 | Train Acc 97.1972| Valid Acc 51.4318
TestSub: S[26, 27, 28, 29, 30] Epoch 58 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0795 | Train Loss 0.0778 | Valid Loss 2.1405 | Train Acc 97.2021| Valid Acc 52.7329
TestSub: S[26, 27, 28, 29, 30] Epoch 59 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0786 | Valid Loss 2.2265 | Train Acc 97.1863| Valid Acc 53.3104
TestSub: S[26, 27, 28, 29, 30] Epoch 60 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0768 | Valid Loss 2.5852 | Train Acc 97.2538| Valid Acc 50.4499
TestSub: S[26, 27, 28, 29, 30] Epoch 61 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0796 | Train Loss 0.0770 | Valid Loss 2.2965 | Train Acc 97.2495| Valid Acc 51.6294
TestSub: S[26, 27, 28, 29, 30] Epoch 62 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0777 | Valid Loss 2.4830 | Train Acc 97.2513| Valid Acc 50.5289
TestSub: S[26, 27, 28, 29, 30] Epoch 63 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0765 | Valid Loss 2.4099 | Train Acc 97.2659| Valid Acc 52.9061
TestSub: S[26, 27, 28, 29, 30] Epoch 64 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0797 | Train Loss 0.0780 | Valid Loss 2.4354 | Train Acc 97.2149| Valid Acc 51.9030
TestSub: S[26, 27, 28, 29, 30] Epoch 65 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0764 | Valid Loss 2.3209 | Train Acc 97.2817| Valid Acc 51.5139
TestSub: S[26, 27, 28, 29, 30] Epoch 66 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0772 | Valid Loss 2.2750 | Train Acc 97.2349| Valid Acc 53.1098
TestSub: S[26, 27, 28, 29, 30] Epoch 67 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0853 | Valid Loss 2.3635 | Train Acc 96.9261| Valid Acc 49.1276
TestSub: S[26, 27, 28, 29, 30] Epoch 68 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0936 | Valid Loss 2.4840 | Train Acc 96.6166| Valid Acc 47.3401
TestSub: S[26, 27, 28, 29, 30] Epoch 69 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0798 | Train Loss 0.0864 | Valid Loss 2.2767 | Train Acc 96.8993| Valid Acc 52.2890
TestSub: S[26, 27, 28, 29, 30] Epoch 70 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0835 | Valid Loss 2.4744 | Train Acc 97.0051| Valid Acc 48.5530
TestSub: S[26, 27, 28, 29, 30] Epoch 71 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0844 | Valid Loss 2.3377 | Train Acc 96.9407| Valid Acc 52.4076
TestSub: S[26, 27, 28, 29, 30] Epoch 72 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0824 | Valid Loss 2.6309 | Train Acc 97.0434| Valid Acc 49.5683
TestSub: S[26, 27, 28, 29, 30] Epoch 73 Finsh | Now_lr 0.0030/0.0030 | Lambda 0.0799 | Train Loss 0.0831 | Valid Loss 2.6520 | Train Acc 97.0398| Valid Acc 50.0973
Early stopping triggered after 73 epochs (30 epochs without improvement)
--------------------------------------------------
Test_Subject :S[26, 27, 28, 29, 30] |Best epoch:43 | Test Loss:2.4220 | Best Acc 54.3744 | Savemodel Acc 54.3744
--------------------------------------------------
lr:0.003 -> bs:64
6-Fold Cross Subject Accuracies: [59.36588035019456, 52.751094357976655, 53.213156614785994, 54.32879377431906, 55.52650778210116, 54.37439202334631]
Mean Accuracy over 6 folds: 54.9266
============================================================================================================
end time: 25-10-09-12:48:12
