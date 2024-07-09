import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
import numpy as np
import sys

model_path = 'D:\Yasamin\ImageProcessing\subpixel-embedding-segmentation\src'
sys.path.append(model_path)

from eval_RPE_Labview import create_model, evaluate

class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node('image_segmentation_node', anonymous=True)
        
        # Load or initialize your model here
        self.model = create_model()
        
        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("Labview2Python", Image, self.image_callback)
        
        # Publishers for the two arrays of positive indices
        self.rpe_pub = rospy.Publisher("/rpe_indices", Int32MultiArray, queue_size=1)
        self.ilm_pub = rospy.Publisher("/ilm_indices", Int32MultiArray, queue_size=1)
    
    def image_callback(self, msg):
        print("Starting image listener...")

        width = msg.width
        height = msg.height
        img_data = np.array(list(msg.data)).reshape(height, width)
    
        # processed_img = np.flipud(img_data) # replace with image processing function
        rpe, ilm = evaluate(self.model, img_data)

        # Publish the results
        self.publish_indices(rpe, ilm)

    
    def publish_indices(self, class1_indices, class2_indices):
        # Publish the indices as Int32MultiArray messages
        class1_msg = Int32MultiArray(data=class1_indices.flatten().tolist())
        class2_msg = Int32MultiArray(data=class2_indices.flatten().tolist())
        
        self.class1_pub.publish(class1_msg)
        self.class2_pub.publish(class2_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = ImageSegmentationNode()
    node.run()
