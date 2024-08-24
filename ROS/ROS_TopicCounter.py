import roslibpy
import time

# Initialize counter
counter1 = 0
counter2 = 0

def callback(message):
    global counter1
    counter1 += 1
    print(f'Counter1: {counter1}, Counter2: {counter2}')
    # Publish the counter to the counter_topic
    counter_publisher.publish(roslibpy.Message({'data': counter1}))

def callback2(message):
    global counter2
    counter2 += 1
    print(f'Counter1: {counter1}, Counter2: {counter2}')
    # Publish the counter to the counter_topic
    counter_publisher.publish(roslibpy.Message({'data': counter2}))

# Connect to the ROS bridge server
client = roslibpy.Ros(host='localhost', port=9090)
client.run()

# Subscribe to the input topic
# Incoming image from labview or python to python
listener = roslibpy.Topic(client, '/Labview2Python', 'sensor_msgs/Image', queue_size=1, queue_length=1)
listener.subscribe(callback)

# Outgoing image from python to labview
listener2 = roslibpy.Topic(client, '/Python2Labview', 'sensor_msgs/Image', queue_size=1, queue_length=1)
listener2.subscribe(callback2)

# Publisher to send the counter
counter_publisher = roslibpy.Topic(client, '/counter_topic', 'std_msgs/Int32')

try:
    while client.is_connected:
        time.sleep(1)
        pass  # Keep the script running while connected to ROS

except KeyboardInterrupt:
    print('Disconnecting...')

finally:
    listener.unsubscribe()
    counter_publisher.unadvertise()
    client.terminate()
