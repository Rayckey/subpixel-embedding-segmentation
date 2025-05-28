import cv2
import sys
# model_path = 'D:\Yasamin\ImageProcessing\subpixel-embedding-segmentation\src'
model_path = 'C:\\Users\\yforo\\Box\\my UCLA Files\\IRISS\\IRISS_Project_Git\\Image Processing\\subpixel-embedding-segmentation\\src'
sys.path.append(model_path)
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32MultiArray
import numpy as np
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
import matplotlib.pyplot as plt


model = None
# model_path = 'D:\Yasamin\ImageProcessing\subpixel-embedding-segmentation\src'
# sys.path.append(model_path)

# from eval_RPE_Labview import create_model, evaluate

class ImageSegmentationNode:
    def __init__(self):
        # Load or initialize your model here
        self.model = self.create_model()


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
            restore_path="C:\\Users\\yforo\\Box\\my UCLA Files\\IRISS\\IRISS_Project_Git\\Image Processing\\subpixel-embedding-segmentation\\trained_models\\Yas\\model-500.pth",
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


    def evaluate(self,
                input_array = None,
                single_input_path='D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\VSCAN_0012-071.png',
                do_visualize_predictions=False,
                dataset_normalization='standard',
                dataset_means=[settings.MULTI_MEAN],
                dataset_stddevs=[settings.MULTI_SD],
                n_chunk=settings.N_CHUNK,
                ):
        # global model

        if self.model is None:
            self.model = self.create_model()

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
        image[0:400,:] = 0
        # image = (np.clip(image, 20,80) - 20) / 60 * 255
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

            self.model.eval()
            best_results = None

            # Run without ground truth, will only save results

            validate = testMulti

            results = validate(
                model=self.model,
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


if __name__ == '__main__':
    model = ImageSegmentationNode()
    dir = 'C:\\Users\\yforo\\Box\\my UCLA Files\\IRISS\\IRISS_Project_Git\\SRI Project\\Data\\Experiments\\2024-09-04\\'
    folder = 'Eye1Data2_png\\'
    file = 'BSCAN_58309264.png'
    path = dir + folder +file
    image = Image.open(path) # np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\41060326.txt') # 
    image = ImageOps.grayscale(image)
    # image = np.array(image, dtype = np.float32)
    # image[0:400,:] =0
    
    rpe, ilm = model.evaluate(image)

    rgb_image = np.stack([image] * 3, axis=-1)
    for coord in rpe.T:
        rgb_image[coord[0], coord[1]] = [0, 255, 0]

    for coord in ilm.T:
        rgb_image[coord[0], coord[1]] = [0, 0, 255]

    plt.imshow(rgb_image)
    plt.show()

