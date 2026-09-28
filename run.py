import argparse
import os
import torch

from exp.exp_rul import Exp_RUL

from utils.print_args import print_args, print_hyperparameters
from utils.tools import load_content
import random
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='psdi_kd')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, zero_shot_forecast, few_shot_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer', help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=768, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48, help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # hyperparameters
    parser.add_argument('--vlm_type', type=str, default='CLIP', help='VLM model type, e.g. CLIP, BLIP2, etc.')
    parser.add_argument('--image_size', type=int, default=224, help='image size for time series to image')
    parser.add_argument('--memory_bank_size', type=int, default=20, help='memory bank size')
    parser.add_argument('--patch_memory_size', type=int, default=100, help='patch memory bank size')
    parser.add_argument('--periodicity', type=int, default=96)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--three_channel_image', type=str2bool, default=True, help='use three channel image')
    parser.add_argument('--finetune_vlm', type=str2bool, default=False, help='finetune VLM model')
    parser.add_argument('--learnable_image', type=str2bool, default=True, help='learnable image')
    parser.add_argument('--save_images', type=str2bool, default=False, help='save images')
    parser.add_argument('--use_cross_attention', type=str2bool, default=True, help='use cross attention to fuse image and text embeddings in customVLM')
    parser.add_argument('--w_out_visual', type=str2bool, default=False, help='without visual part')
    parser.add_argument('--w_out_text', type=str2bool, default=False, help='without text part')
    parser.add_argument('--w_out_query', type=str2bool, default=False, help='without query part')
    parser.add_argument('--visualize_embeddings', type=str2bool, default=False, help='visualize embeddings')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2024, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding', type=int, default=8, help='padding')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--align_const', type=float, default=0.4)

    # zero-shot forecasting
    parser.add_argument('--target_data', type=str, default='ETTh2', help='target dataset type')
    parser.add_argument('--target_root_path', type=str, default='./data/ETT/', help='root path of the target data file')
    parser.add_argument('--target_data_path', type=str, default='ETTh2.csv', help='target data file')

    # few-shot forecasting
    parser.add_argument('--percent', type=float, default=1, help='proportion of in-distribution downstream dataset')

    # ablation experiment related parameters
    parser.add_argument('--ablate_vision_modality', type=str2bool, default=False, help='ablation vision modality (use zero tensor instead of vision features)')
    parser.add_argument('--ablate_temporal_modality', type=str2bool, default=False, help='ablation temporal modality (use zero tensor instead of temporal features)')

    # MAE pre-training model path parameter
    parser.add_argument('--mae_pretrained_path', type=str, default='facebook/vit-mae-base', help='Hugging Face MAE pre-training model ID or local path')
    # use CLS token parameter
    parser.add_argument('--use_cls_token', type=str2bool, default=False, help='use CLS token instead of average pooling for MAE features')
    # MAE size parameter
    parser.add_argument('--mae_size', type=str, default='base', help='MAE model size, options: [base, large, huge]')

    # knowledge distillation related parameters
    parser.add_argument('--use_distillation', type=str2bool, default=False, help='whether to use knowledge distillation')
    parser.add_argument('--teacher_vlm_type', type=str, default='clip', help='teacher VLM type, options: [clip, blip2, vilt, custom]')
    parser.add_argument('--student_vision_type', type=str, default='tiny_vit', help='student vision backbone type, options: [tiny_vit, mid_vit, vit, efficientnet, mobilenet]')
    parser.add_argument('--teacher_hidden_size', type=int, default=512, help='hidden size of teacher model')
    parser.add_argument('--student_hidden_size', type=int, default=128, help='hidden size of student model')


    parser.add_argument('--feature_w', type=float, default=0.01, help='feature distillation loss weight')
    parser.add_argument('--fcst_w', type=float, default=1.0, help='forecast loss weight')
    parser.add_argument('--recon_w', type=float, default=0.5, help='reconstruction loss weight')
    parser.add_argument('--att_w', type=float, default=0.01, help='attention distillation loss weight')
    
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='mask ratio for MAE')

    # distillation training mode
    parser.add_argument('--distillation_mode', type=str, default='two_stage', 
                    help='distillation mode: two_stage (two-stage training) or joint (joint training)')

    # teacher model pretraining related parameters
    parser.add_argument('--teacher_pretrain_epochs', type=int, default=10, 
                    help='teacher pretraining epochs (only used in two_stage mode)')

    # adaptive distillation parameters
    parser.add_argument('--init_temperature', type=float, default=4.0, 
                    help='initial temperature for distillation')
    parser.add_argument('--enable_adaptive_weights', type=str2bool, default=True, 
                    help='enable adaptive weights learning')
    parser.add_argument('--distill_lr_ratio', type=float, default=0.1, 
                    help='distillation parameter learning rate ratio to main model learning rate')
    
    # feature alignment related parameters  
    parser.add_argument('--num_alignment_scales', type=int, default=3, 
                    help='number of scales for multi-scale feature alignment')
    parser.add_argument('--feature_alignment_dropout', type=float, default=0.1, 
                    help='dropout rate for feature alignment layer')
    
    # loss balancer parameters
    parser.add_argument('--loss_momentum', type=float, default=0.9, 
                    help='momentum parameter for loss balancer')
    parser.add_argument('--weight_regularization', type=float, default=0.001,
                    help='weight regularization coefficient')

    # RUL prediction parameters
    parser.add_argument('--rul_phase', type=int, default=1,
                    help='RUL phase: 1=student-only, 2=+distillation, 3=+cross-attention')
    parser.add_argument('--pca_patch_len', type=int, default=64,
                    help='patch length for PCA temporal encoder (Phase 3)')
    parser.add_argument('--pca_stride', type=int, default=32,
                    help='stride for PCA temporal encoder (Phase 3)')
    parser.add_argument('--pca_padding', type=int, default=32,
                    help='padding for PCA temporal encoder (Phase 3)')
    parser.add_argument('--phase2_ckpt_path', type=str, default='',
                    help='Phase 2 student checkpoint path for Phase 3 resume')
    parser.add_argument('--phase2_teacher_ckpt_path', type=str, default='',
                    help='Phase 2 teacher checkpoint path for Phase 3 resume')
    parser.add_argument('--skip_teacher_stage_in_phase3', type=str2bool, default=True,
                    help='Skip teacher retraining and load teacher checkpoint in Phase 3')
    parser.add_argument('--phase3_from_scratch', type=str2bool, default=False,
                    help='Run full Phase 3 pipeline from scratch (no Phase 2 checkpoint load)')
    parser.add_argument('--phase3_qkv_mode', type=str, default='vision_q_phase_kv',
                    choices=['phase_q_vision_kv', 'vision_q_phase_kv'],
                    help='Phase-3 cross-attention direction: phase_q_vision_kv or vision_q_phase_kv')
    parser.add_argument('--phase3_residual_fusion', type=str2bool, default=False,
                    help='Fuse Phase-3 cross-modal feature with Phase-2 vision feature')
    parser.add_argument('--phase3_fusion_alpha_init', type=float, default=0.7,
                    help='Initial alpha for Phase-3 residual fusion (alpha*phase3 + (1-alpha)*vision)')

    # Delta RUL parameters
    parser.add_argument('--window_size', type=int, default=10,
                    help='Number of consecutive frames in delta RUL window (K)')
    parser.add_argument('--use_delta_only', type=str2bool, default=True,
                    help='Use P_delta (end - start) for physics input')
    parser.add_argument('--dataset_name', type=str, default='xjtu',
                    help='Dataset name for FPT/EOF lookup: xjtu or phm')
    parser.add_argument('--degradation_only', type=str2bool, default=False,
                    help='Only use degradation-phase windows (from FPT onwards)')
    parser.add_argument('--delta_factor', type=float, default=3.0,
                    help='Safety multiplier for delta_max computation')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay for AdamW optimizer')
    parser.add_argument('--use_degradation_weighted_fcst', type=str2bool, default=False,
                    help='Use degradation-aware weights for forecast loss')
    parser.add_argument('--degradation_lambda', type=float, default=2.0,
                    help='Weight strength for degradation-aware forecast loss')
    parser.add_argument('--degradation_gamma', type=float, default=2.0,
                    help='Exponent for degradation-aware forecast loss')
    parser.add_argument('--temporal_mono_w', type=float, default=0.0,
                    help='Weight for monotonic temporal regularization')
    parser.add_argument('--temporal_slope_w', type=float, default=0.0,
                    help='Weight for slope consistency temporal regularization')
    parser.add_argument('--learn_temporal_weights', type=str2bool, default=False,
                    help='Learn temporal mono/slope regularization weights during training')

    # CNN baseline parameters
    parser.add_argument('--cnn_type', type=str, default=None,
                    choices=[None, 'simple', 'vgg', 'resnet18', 'resnet34', 'efficientnet_b0'],
                    help='CNN baseline type for Phase-1 standalone training. '
                         'None (default) uses the existing teacher/student pipeline.')
    parser.add_argument('--finetune_cnn_backbone', type=str2bool, default=True,
                    help='Finetune pretrained CNN backbone (ResNet/EfficientNet). '
                         'Set False to freeze backbone and only train RUL head.')

    # K-Window Temporal Aggregation
    parser.add_argument('--k_window_size', type=int, default=1,
                    help='Sliding window size K (1=single image, default)')
    parser.add_argument('--temporal_conv_layers', type=int, default=2,
                    help='Number of temporal conv layers in aggregator')
    parser.add_argument('--temporal_conv_kernel', type=int, default=3,
                    help='Kernel size for temporal convolution')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    # set gpu id
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name in ('rul', 'delta_rul'):
        args.content = None
    else:
        args.content = load_content(args)

    # print arguments
    print_args(args)
    print_hyperparameters(args)

    if args.task_name == 'rul':
        Exp = Exp_RUL
    else:
        Exp = Exp_RUL

if args.is_training:
    for ii in range(args.itr):
        exp = Exp(args)

        # === Setting string construction ===
        if args.task_name == 'rul':
            # RUL-specific setting string
            setting = 'rul_{}_phase{}_dm{}_{}'.format(
                args.model_id, args.rul_phase, args.d_model, ii)
            if args.rul_phase >= 2:
                setting += '_dst_{}'.format(args.teacher_vlm_type)
        elif args.task_name == 'delta_rul':
            setting = 'delta_rul_{}_K{}_dm{}_{}'.format(
                args.model_id, args.window_size, args.d_model, ii)
            if getattr(args, 'use_distillation', False) and args.use_distillation:
                setting += '_dst_{}'.format(args.teacher_vlm_type)
        else:
            # Original setting string for forecasting tasks
            base_setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}'.format(
                args.task_name,
                args.vlm_type,  # Note: This is usually the student's type in distillation mode
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.percent,
            )

            # If using distillation, add distillation-related parameters and replace '.' with '^'
            if args.use_distillation:
                distill_setting = '_dst{}_tvlm{}_svt{}_fw{}_ftw{}_rw{}_aw{}_mr{}'.format(
                    args.distillation_mode,  # distillation mode
                    args.teacher_vlm_type,
                    args.student_vision_type,
                    str(args.feature_w).replace('.', '^'),
                    str(args.fcst_w).replace('.', '^'),
                    str(args.recon_w).replace('.', '^'),
                    str(args.att_w).replace('.', '^'),
                    str(args.mask_ratio).replace('.', '^')
                )
                setting = base_setting + distill_setting + '_{}'.format(ii) # Add iteration number
            else:
                # If not using distillation, add a marker
                setting = base_setting + '_nodst' + '_{}'.format(ii) # Add iteration number

            # ablation experiment identifier
            if args.ablate_vision_modality:
                setting = setting + '_ablv'  # ablate vision
            if args.ablate_temporal_modality:
                setting = setting + '_ablt'  # ablate temporal
        # === END setting string ===

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    exp = Exp(args)

    # === Setting string construction ===
    if args.task_name == 'rul':
        setting = 'rul_{}_phase{}_dm{}_{}'.format(
            args.model_id, args.rul_phase, args.d_model, ii)
        if args.rul_phase >= 2:
            setting += '_dst_{}'.format(args.teacher_vlm_type)
    else:
        # Base setting part
        base_setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}'.format(
            args.task_name,
            args.vlm_type, # Note: This is usually the student's type in distillation mode
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.percent,
        )

        # If using distillation, add distillation-related parameters and replace '.' with '^'
        if args.use_distillation:
            distill_setting = '_dst{}_tvlm{}_svt{}_fw{}_ftw{}_rw{}_aw{}_mr{}'.format(
                args.distillation_mode,  # distillation mode
                args.teacher_vlm_type,
                args.student_vision_type,
                str(args.feature_w).replace('.', '^'),
                str(args.fcst_w).replace('.', '^'),
                str(args.recon_w).replace('.', '^'),
                str(args.att_w).replace('.', '^'),
                str(args.mask_ratio).replace('.', '^')
            )
            setting = base_setting + distill_setting + '_{}'.format(ii) # Add iteration number
        else:
            # If not using distillation, add a marker
            setting = base_setting + '_nodst' + '_{}'.format(ii) # Add iteration number

        # ablation experiment identifier
        if args.ablate_vision_modality:
            setting = setting + '_ablv'  # ablate vision
        if args.ablate_temporal_modality:
            setting = setting + '_ablt'  # ablate temporal
    # === MODIFICATION END ===

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
