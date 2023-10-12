export CUDA_VISIBLE_DEVICES=0

python3 src/run_spin.py \
--multimodal_scan_paths training/RPE-vscans-train-images.txt \
--ground_truth_path training/RPE-vscans-train-masks.txt \
--dataset_normalization standard \
--dataset_means 47.119747 \
--dataset_stddevs 22.61012 \
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
--checkpoint_path \
trained_spin_models/RPE/spin_traintest_512x200_wpos4 \
--restore_path \
trained_spin_models/RPE/spin_traintest_512x200_wpos4/model-2000.pth \
--device gpu \
--n_thread 8 \
--do_visualize_predictions
