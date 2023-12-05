import argparse
import RPE_constants as settings
import torch, os
from data_utils import *
import data_utils
import datasets
from log_utils import log, save_RPE_prediction_img
from transforms import Transforms
from spin_model import SPiNModel
from eval_utils import testRPE

parser = argparse.ArgumentParser()



def run(single_input_path,
        # Input settings
        n_chunk=settings.N_CHUNK,
        # Normalization setting
        dataset_normalization='standard',
        dataset_means=[settings.RPE_MEAN],
        dataset_stddevs=[settings.RPE_SD],
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING,
        n_filters_encoder_subpixel_embedding=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING,
        decoder_type_subpixel_embedding=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING,
        n_filter_decoder_subpixel_embedding=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING,
        output_channels_subpixel_embedding=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING,
        output_func_subpixel_embedding=settings.OUTPUT_FUNC,
        # Segmentation network settings
        encoder_type_segmentation=settings.ENCODER_TYPE_SEGMENTATION,
        n_filters_encoder_segmentation=settings.N_FILTERS_ENCODER_SEGMENTATION,
        resolutions_subpixel_guidance=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE,
        n_filters_subpixel_guidance=settings.N_FILTERS_SUBPIXEL_GUIDANCE,
        n_convolutions_subpixel_guidance=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE,
        decoder_type_segmentation=settings.DECODER_TYPE_SEGMENTATION,
        n_filters_decoder_segmentation=settings.N_FILTERS_DECODER_SEGMENTATION,
        n_filters_learnable_downsampler=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER,
        kernel_sizes_learnable_downsampler=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER,
        # Weights settings
        weight_initializer=settings.WEIGHT_INITIALIZER,
        activation_func=settings.ACTIVATION_FUNC,
        use_batch_norm=settings.USE_BATCH_NORM,
        # Test time augmentation
        augmentation_flip_type=['none'],
        # Checkpoint settings
        checkpoint_path=settings.CHECKPOINT_PATH,
        restore_path=settings.RESTORE_PATH,
        do_visualize_predictions=True,
        # Hardware settings
        device=settings.DEVICE,
        n_thread=settings.N_THREAD):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if do_visualize_predictions:
        # Get input modality names for ....:
        visual_path = single_input_path + ".out.png"


    # Set up dataloader
     # Determine which data type to use
    scan_type = 'MRI'
    # Training dataloader
    if scan_type == 'MRI':
        dataloader = torch.utils.data.DataLoader(
            datasets.SPiNMRISingleDataset(
                scan_path=single_input_path,
                shape=(None, None)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        input_channels = n_chunk
        validate = None
        save_prediction_img = save_RPE_prediction_img

    transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization)

    # Obtain indices of scans with small lesions
    small_lesion_idxs = None

    # Build subpixel network (SPiN)
    model = SPiNModel(
        input_channels=input_channels,
        encoder_type_subpixel_embedding=encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=decoder_type_subpixel_embedding,
        output_channels_subpixel_embedding=output_channels_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=n_filter_decoder_subpixel_embedding,
        output_func_subpixel_embedding=output_func_subpixel_embedding,
        encoder_type_segmentation=encoder_type_segmentation,
        n_filters_encoder_segmentation=n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=n_convolutions_subpixel_guidance,
        decoder_type_segmentation=decoder_type_segmentation,
        n_filters_decoder_segmentation=n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
        device=device)

    _, parameters_subpixel_embedding, parameters_segmentation = model.parameters()

    if restore_path is not None:

        assert os.path.isfile(restore_path), \
            'Cannot find retore path: {}'.format(restore_path)

        model.restore_model(restore_path)

    with torch.no_grad():

        model.eval()
        best_results = None

        # Run without ground truth, will only save results

        validate = testRPE

        validate(
            model=model,
            dataloader=dataloader,
            transforms=transforms,
            log_path='',
            save_prediction_img=save_prediction_img,
            step=0,
            dataset_means=dataset_means,
            n_chunk=n_chunk,
            visual_path= visual_path)

    return best_results


# --single_input_path training/RPE-vscans-train-images.txt \
# --ground_truth_path training/RPE-vscans-train-masks.txt \
# --dataset_normalization standard \
# --dataset_means 47.119747 \
# --dataset_stddevs 22.61012 \
# --encoder_type_subpixel_embedding resnet5_subpixel_embedding \
# --n_filters_encoder_subpixel_embedding 16 16 16 \
# --decoder_type_subpixel_embedding subpixel \
# --n_filter_decoder_subpixel_embedding 16 \
# --output_channels_subpixel_embedding 8 \
# --output_func_subpixel_embedding linear \
# --encoder_type_segmentation resnet18 \
# --n_filters_encoder_segmentation 32 64 128 196 196 \
# --resolutions_subpixel_guidance 0 1 \
# --n_filters_subpixel_guidance 8 8 \
# --n_convolutions_subpixel_guidance 1 1 \
# --decoder_type_segmentation subpixel_guidance learnable_downsampler \
# --n_filters_decoder_segmentation 196 128 64 32 16 16 \
# --n_filters_learnable_downsampler 16 16 \
# --kernel_sizes_learnable_downsampler 3 3 \
# --weight_initializer kaiming_uniform \
# --activation_func leaky_relu \
# --use_batch_norm \
# --checkpoint_path \
# trained_spin_models/RPE/spin_traintest_512x200_wpos4 \
# --restore_path \
# trained_spin_models/RPE/spin_traintest_512x200_wpos4/model-2000.pth \
# --device gpu \
# --n_thread 8 \
# --do_visualize_predictions

# validation input filepaths
parser.add_argument('--single_input_path',
    default='testing\VSCAN_0012-071.png', nargs='+', type=str, help='Paths to list of MRI scan paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground-truth annotation paths')
# Input settings
parser.add_argument('--n_chunk',
    type=int, default=settings.N_CHUNK, help='Number of chunks or channels to process at a time')
# Normalization settings
parser.add_argument('--dataset_normalization',
    type=str, default='standard', help='Type of normalization: none, standard')
parser.add_argument('--dataset_means',
    nargs='+', type=float, default=[settings.RPE_MEAN], help='List of mean values of each modality in the dataset. Used after log preprocessing')
parser.add_argument('--dataset_stddevs',
    nargs='+', type=float, default=[settings.RPE_SD], help='List of standard deviations of each modality in the dataset. Used after log preprocessing')
# Subpixel embedding network settings
parser.add_argument('--encoder_type_subpixel_embedding',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SUBPIXEL_EMBEDDING, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING)
parser.add_argument('--n_filters_encoder_subpixel_embedding',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SUBPIXEL_EMBEDDING, help='Number of filters to use in each encoder block')
parser.add_argument('--decoder_type_subpixel_embedding',
    nargs='+', type=str, default=settings.DECODER_TYPE_SUBPIXEL_EMBEDDING, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SUBPIXEL_EMBEDDING)
parser.add_argument('--n_filter_decoder_subpixel_embedding',
    type=int, default=settings.N_FILTER_DECODER_SUBPIXEL_EMBEDDING, help='Number of filters to use in each decoder block')
parser.add_argument('--output_channels_subpixel_embedding',
    type=int, default=settings.OUTPUT_CHANNELS_SUBPIXEL_EMBEDDING, help='Number of filters to use in output')
parser.add_argument('--output_func_subpixel_embedding',
    type=str, default=settings.OUTPUT_FUNC, help='Output func: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
# Segmentation network settings
parser.add_argument('--encoder_type_segmentation',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SEGMENTATION, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_encoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SEGMENTATION, help='Number of filters to use in each encoder block')
parser.add_argument('--resolutions_subpixel_guidance',
    nargs='+', type=int, default=settings.RESOLUTIONS_SUBPIXEL_GUIDANCE, help='Exponent of scales for subpixel guidance modules')
parser.add_argument('--n_filters_subpixel_guidance',
    nargs='+', type=int, default=settings.N_FILTERS_SUBPIXEL_GUIDANCE, help='Number of filters for each module of SPG')
parser.add_argument('--n_convolutions_subpixel_guidance',
    nargs='+', type=int, default=settings.N_CONVOLUTIONS_SUBPIXEL_GUIDANCE, help='Number of convolutions for each S2D output to undergo')
parser.add_argument('--decoder_type_segmentation',
    nargs='+', type=str, default=settings.DECODER_TYPE_SEGMENTATION, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SEGMENTATION)
parser.add_argument('--n_filters_decoder_segmentation',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER_SEGMENTATION, help='Number of filters to use in each decoder block')
parser.add_argument('--n_filters_learnable_downsampler',
    nargs='+', type=int, default=settings.N_FILTERS_LEARNABLE_DOWNSAMPLER, help='Number of filters to use in learnable downsampler')
parser.add_argument('--kernel_sizes_learnable_downsampler',
    nargs='+', type=int, default=settings.KERNEL_SIZES_LEARNABLE_DOWNSAMPLER, help='Number of filters to use in learnable downsampler')
# Weights settings
parser.add_argument('--weight_initializer',
    type=str, default=settings.WEIGHT_INITIALIZER, help='Weight initializers: %s' % settings.WEIGHT_INITIALIZER_AVAILABLE)
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation func: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
parser.add_argument('--use_batch_norm',
    default=True ,help='yes')
# Test time augmentations
parser.add_argument('--augmentation_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type for augmentation: horizontal_test, vertical_test, both_test')
# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default='', help='Path to save checkpoints')
parser.add_argument('--restore_path',
    default='trained_spin_models\RPE\spin_traintest_512x200_wpos5\model-1000.pth', type=str,  help='Path to restore segmentation model to resume training')
parser.add_argument('--do_visualize_predictions',
    default=True, help='If true, visualize and store predictions as png.')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')

args = parser.parse_args()

if __name__ == '__main__':

    if args.ground_truth_path == '':
        args.ground_truth_path = None

    if args.restore_path == '':
        args.restore_path = None


    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(single_input_path=args.single_input_path,
        # Input settings
        n_chunk=args.n_chunk,
        # Normalization setting
        dataset_normalization=args.dataset_normalization,
        dataset_means=args.dataset_means,
        dataset_stddevs=args.dataset_stddevs,
        # Subpixel embedding network settings
        encoder_type_subpixel_embedding=args.encoder_type_subpixel_embedding,
        n_filters_encoder_subpixel_embedding=args.n_filters_encoder_subpixel_embedding,
        decoder_type_subpixel_embedding=args.decoder_type_subpixel_embedding,
        n_filter_decoder_subpixel_embedding=args.n_filter_decoder_subpixel_embedding,
        output_channels_subpixel_embedding=args.output_channels_subpixel_embedding,
        output_func_subpixel_embedding=args.output_func_subpixel_embedding,
        # Segmentation network settings
        encoder_type_segmentation=args.encoder_type_segmentation,
        n_filters_encoder_segmentation=args.n_filters_encoder_segmentation,
        resolutions_subpixel_guidance=args.resolutions_subpixel_guidance,
        n_filters_subpixel_guidance=args.n_filters_subpixel_guidance,
        n_convolutions_subpixel_guidance=args.n_convolutions_subpixel_guidance,
        decoder_type_segmentation=args.decoder_type_segmentation,
        n_filters_decoder_segmentation=args.n_filters_decoder_segmentation,
        n_filters_learnable_downsampler=args.n_filters_learnable_downsampler,
        kernel_sizes_learnable_downsampler=args.kernel_sizes_learnable_downsampler,
        # Weights settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        use_batch_norm=args.use_batch_norm,
        # Test time augmentation
        augmentation_flip_type=args.augmentation_flip_type,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        do_visualize_predictions=args.do_visualize_predictions,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
