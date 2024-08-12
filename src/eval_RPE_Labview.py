import argparse
import torch, os
from data_utils import *
import datasets
from log_utils import log, save_multi_MRI_prediction_img
from transforms import Transforms
from spin_model import SPiNModel
from eval_utils import testMulti
from PIL import Image, ImageOps
import time

import global_constants as settings
from spin_main import run
import data_utils

model = None


def create_model(
        multimodal_scan_paths="validation/multi-vscans-val-images.txt",
        ground_truth_path="validation/multi-vscans-val-masks.txt",
        n_batch=4,
        n_chunk=1,
        n_height=1024,
        n_width=400,
        dataset_normalization="standard",
        dataset_means=[47.034603],
        dataset_stddevs=[22.447832],
        encoder_type_subpixel_embedding="resnet5_subpixel_embedding",
        n_filters_encoder_subpixel_embedding=[16, 16, 16],
        decoder_type_subpixel_embedding="subpixel",
        n_filter_decoder_subpixel_embedding=16,
        output_channels_subpixel_embedding=8,
        output_func_subpixel_embedding="linear",
        encoder_type_segmentation="resnet18",
        n_filters_encoder_segmentation=[32, 64, 128, 196, 196],
        resolutions_subpixel_guidance=[0, 1],
        n_filters_subpixel_guidance=[8, 8],
        n_convolutions_subpixel_guidance=[1, 1],
        decoder_type_segmentation=["subpixel_guidance", "learnable_downsampler"],
        n_filters_decoder_segmentation=[196, 128, 64, 32, 16, 16],
        n_filters_learnable_downsampler=[16, 16],
        kernel_sizes_learnable_downsampler=[3, 3],
        weight_initializer="kaiming_uniform",
        activation_func="leaky_relu",
        use_batch_norm=True,
        augmentation_flip_type="horizontal",
        checkpoint_path="trained_spin_models/multi/spin_traintest_1024x400_wpos4",
        restore_path="D:\Yasamin\ImageProcessing\subpixel-embedding-segmentation/trained_models/model-33500.pth",
        do_visualize_predictions=True,
        device="gpu",
        n_thread=8

        ):

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


def evaluate(model = None,
            input_array = None,
            single_input_path='D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\VSCAN_0012-071.png',
            do_visualize_predictions=False,
            dataset_normalization='standard',
            dataset_means=[settings.ATLAS_MEAN],
            dataset_stddevs=[settings.ATLAS_SD],
            n_chunk=settings.N_CHUNK,
            ):
    # global model

    if model is None:
        model = create_model()

    if do_visualize_predictions:
            # Get input modality names for ....:
        visual_path = single_input_path[:-4] + ".out.png"
    else:
        visual_path = ''

    if input_array is None:
        image = Image.open(single_input_path) # np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\41060326.txt') # 
        image = ImageOps.grayscale(image)
    else:
        image = input_array
    image = np.array(image, dtype = np.float32)
    image = (np.clip(image, 20,80) - 20) / 60 * 255
    scan = np.expand_dims(image, axis=0)
    scan = np.expand_dims(scan, axis=-1).astype(np.float32)
    
    # Scan shape: D x H x W x C -> C x D x H x W
    scan = np.transpose(scan, (3, 0, 1, 2))        

    validate = None
    save_prediction_img = save_multi_MRI_prediction_img

    transforms = Transforms(
        dataset_means=dataset_means,
        dataset_stddevs=dataset_stddevs,
        dataset_normalization=dataset_normalization)
    with torch.no_grad():

        model.eval()
        best_results = None

        # Run without ground truth, will only save results

        validate = testMulti

        results = validate(
            model=model,
            scan=scan,
            transforms=transforms,
            save_prediction_img=save_prediction_img,
            n_chunk=n_chunk,
            dataset_means=dataset_means,           
            visual_paths= visual_path)
    rpe = np.vstack(np.where(results == 1))
    ilm = np.vstack(np.where(results == 2))
    # np.column_stack(results_indices).astype(np.float32)
    return rpe, ilm



def trials():
    global model
    # initialize_global_model()
    # image = np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\VSCAN_0012-071.txt')
    image = Image.open('D:\Yasamin\Ascan-Project-Git-Test\\ImageProcessing\\testing\\im203.png') # np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\41060326.txt') # 
    image = ImageOps.grayscale(image)
    t1 = time.time()*1000
    model = create_model()
    for i in range(10):              
        t2 = time.time()*1000
        evaluate(model, image, 'D:\Yasamin\Ascan-Project-Git-Test\\ImageProcessing\\testing\\im203.png', False)
        t3 = time.time()*1000
        print(t3-t2)
    # t2 = time.time()*1000
    # print(t2-t1)
    # print(t3-t2)
    return t2-t1

if __name__ == '__main__':
    #initialize_global_model()
    #evaluate()
    trials()