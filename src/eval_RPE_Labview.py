import argparse
import RPE_constants as settings
import torch, os
from data_utils import *
import data_utils
import datasets
from log_utils import log, save_RPE_prediction_img
from transforms import Transforms
from spin_model import SPiNModel
from eval_utils import testRPE_singleinput
from PIL import Image 
import time

model = None


def create_model(
        # Input settings
        n_chunk=settings.N_CHUNK,
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
        restore_path='D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\trained_spin_models\RPE\spin_traintest_512x200_wpos5\model-1000.pth',
        # Hardware settings
        device='cpu'):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    input_channels = n_chunk


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


    if restore_path is not None:

        assert os.path.isfile(restore_path), \
            'Cannot find retore path: {}'.format(restore_path)

        model.restore_model(restore_path)

    return model

def initialize_global_model():
    global model
    if model is None:
        model = create_model()


def evaluate(input_array = None,
            single_input_path='D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\VSCAN_0012-071.png',
            do_visualize_predictions=False,
            dataset_normalization='standard',
            dataset_means=[settings.RPE_MEAN],
            dataset_stddevs=[settings.RPE_SD],
            n_chunk=settings.N_CHUNK,
            ):
    global model

    if do_visualize_predictions:
            # Get input modality names for ....:
        visual_path = single_input_path[:-4] + ".out.png"
    else:
        visual_path = ''

    #image = Image.open(single_input_path)
    #scan = np.array(image)
    if input_array is None:
        image = Image.open(single_input_path) # np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\41060326.txt') # 
    else:
        image = input_array
    image = np.array(image, dtype = np.float32)
    image = (np.clip(image, 20,80) - 20) / 60 * 255
    scan = np.expand_dims(image, axis=0)
    scan = np.expand_dims(scan, axis=-1).astype(np.float32)
    
    # Scan shape: D x H x W x C -> C x D x H x W
    scan = np.transpose(scan, (3, 0, 1, 2))        

    validate = None
    save_prediction_img = save_RPE_prediction_img

    transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization)
    with torch.no_grad():

        model.eval()
        best_results = None

        # Run without ground truth, will only save results

        validate = testRPE_singleinput

        best_results, results_indices = validate(
            model=model,
            scan=scan,
            transforms=transforms,
            log_path='',
            save_prediction_img=save_prediction_img,
            step=0,
            dataset_means=dataset_means,
            n_chunk=n_chunk,
            visual_path= visual_path)
    return np.column_stack(results_indices).astype(np.float32)



def trials():
    global model
    # initialize_global_model()
    image = np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\VSCAN_0012-071.txt')
    t1 = time.time()*1000
    for i in range(1):
        model = create_model()
        evaluate(image, 'D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\testing\vscan_0012_071.png', False)

    t2 = time.time()*1000
    print(t2-t1)
    return t2-t1

if __name__ == '__main__':
    #initialize_global_model()
    #evaluate()
    trials()