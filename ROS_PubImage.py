import roslibpy
import sys
model_path = 'D:\Yasamin\ImageProcessing\subpixel-embedding-segmentation\src'
sys.path.append(model_path)
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32MultiArray
import numpy as np
from PIL import Image, ImageOps
import time
import pdb


class PublishImage:
    def __init__(self):
        # Load or initialize your model here
        self.ros = roslibpy.Ros(host='localhost', port=9090)
        self.ros.on_ready(self.on_ros_ready, run_in_thread=False)
        self.ros.run()

    def on_ros_ready(self):
        print('Connected to ROS')

        # Publishers for the two arrays of positive indices
        self.rpe_pub = roslibpy.Topic(self.ros, 'rpe_indices', 'std_msgs/Int32MultiArray')
        self.ilm_pub = roslibpy.Topic(self.ros, 'ilm_indices', 'std_msgs/Int32MultiArray')
        self.image_pub = roslibpy.Topic(self.ros, 'Labview2Python', 'sensor_msgs/Image')
    
    def run(self):
        try:
            while not self.ros.is_connected:
                print('Waiting for ROS connection...')
                time.sleep(1)
            print('ROS connection established. Node is running...')
            while self.ros.is_connected:
                node.image_pub.publish(msg)
                time.sleep(6)
                # pass

        except KeyboardInterrupt:
            print('Shutting down ROS node...')
            self.ros.terminate()

if __name__ == '__main__':
    image = Image.open('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\VSCAN_0012-071.png') # np.loadtxt('D:\Yasamin\Ascan-Project-Git-Test\ImageProcessing\\testing\\41060326.txt') # 
    # pdb.set_trace()
    node = PublishImage()
    image = ImageOps.grayscale(image)
    image_array = np.repeat(np.array(image, dtype='uint8'), 4, axis=1).flatten().tolist()
    msg = roslibpy.Message({'data':image_array, 'height':1024, 'width':400,'encoding':'mono8'})
    # node.image_pub.publish(msg)
    node.run()
