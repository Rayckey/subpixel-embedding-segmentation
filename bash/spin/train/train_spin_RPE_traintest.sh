export CUDA_VISIBLE_DEVICES=0

python3 src/train_spin.py \
--train_multimodal_scan_paths training/RPE-vscans-train-images.txt \
--train_ground_truth_path training/RPE-vscans-train-masks.txt \
--val_multimodal_scan_paths validation/RPE-vscans-val-images.txt \
--val_ground_truth_path validation/RPE-vscans-val-masks.txt \
--n_batch 12 \
--n_chunk 1 \
--n_height 512 \
--n_width 200 \
--dataset_normalization standard \
--dataset_means 47.080208 \
--dataset_stddevs 22.534702 \
--encoder_type_subpixel_embedding resnet5_subpixel_embedding \
--n_filters_encoder_subpixel_embedding 16 16 16 \
--decoder_type_subpixel_embedding subpixel \
--n_filter_decoder_subpixel_embedding 16 \
--output_channels_subpixel_embedding 8 \
--output_func_subpixel_embedding linear \
--encoder_type_segmentation resnet18 \
--n_filters_encoder_segmentation 32 64 128 196 196 \
--resolutions_subpixel_guidance 0 1 \
--n_filters_subpixel_guidance 8 8 \
--n_convolutions_subpixel_guidance 1 1 \
--decoder_type_segmentation subpixel_guidance learnable_downsampler \
--n_filters_decoder_segmentation 196 128 64 32 16 16 \
--n_filters_learnable_downsampler 16 16 \
--kernel_sizes_learnable_downsampler 3 3 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--use_batch_norm \
--learning_rates 1e-3 1e-4 1e-5 \
--learning_schedule 200 700 800 \
--positive_class_sample_rates 0.95 \
--positive_class_sample_schedule -1 \
--positive_class_size_thresholds 0 \
--augmentation_probabilities 1.00 0.50 \
--augmentation_schedule 700 800 \
--augmentation_flip_type horizontal \
--augmentation_rotate 45 \
--augmentation_noise_type gaussian \
--augmentation_noise_spread 1e-2 \
--augmentation_resize_and_pad 1.0 1.1 \
--w_weight_decay_subpixel_embedding 0.0 \
--loss_func_segmentation cross_entropy weight_decay \
--w_weight_decay_segmentation 0.0 \
--w_positive_class 5.0 \
--n_summary 500 \
--n_checkpoint 500 \
--checkpoint_path \
trained_spin_models/RPE/spin_traintest_1024x400_wpos1 \
--device gpu \
--n_thread 8
