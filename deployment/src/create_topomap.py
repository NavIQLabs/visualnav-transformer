import argparse
import os
import sys
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/train")
sys.path.append(f"{os.getcwd()}/src/diffusion_policy")
import shutil
import time
from utils import msg_to_pil
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
import threading
os.chdir(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")
class Args:
    pass

class TopoMapCreator(Node):
    def __init__(self):
        super().__init__("create_topomap")
        self.args = self.load_ros_params()
        self.obs_img = None
        self.publisher = self.create_publisher(Image, "/subgoals", 1)
        self.image_subscriber = self.create_subscription(
            Image, self.args.image_topic, self.callback_obs, 1)
        self.joy_subscriber = self.create_subscription(
            Joy, "joy", self.callback_joy, 1)
        self.init_directories()
        self.i = 0
        self.start_time = time.time()
        print(1 /self.args.dt)
        self.rate = self.create_rate(1 / self.args.dt)
        print("Registered with master node. Waiting for images...")

        thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        thread.start()

    def load_ros_params(self):
        self.declare_parameter("dt", 1.0)
        self.declare_parameter("dir", "topomap")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("topomap_images_dir", "../topomaps/images")
        
        args = Args()
        parameters = [ "dt", "dir", "image_topic", "topomap_images_dir"]        
        # Load each parameter and assign it to the args object
        for param_name in parameters:
            print(param_name)
            # Assuming all parameters have been declared before this function is called
            param = self.get_parameter(param_name)
            param_value = param.get_parameter_value()

            # Check the type of the parameter and convert if necessary
            if param_value.type == 2:
                value = param_value.integer_value
            elif param_value.type == 3:
                value = param_value.double_value
            else:
                value = param_value.string_value
            setattr(args, param_name, value)

        return args


    def init_directories(self):
        topomap_name_dir = os.path.join(self.args.topomap_images_dir, self.args.dir)
        if not os.path.isdir(topomap_name_dir):
            os.makedirs(topomap_name_dir)
        else:
            print(f"{topomap_name_dir} already exists. Removing previous images...")
            self.remove_files_in_dir(topomap_name_dir)

    @staticmethod
    def remove_files_in_dir(dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def callback_obs(self, msg):
        self.obs_img = msg_to_pil(msg)

    def callback_joy(self, msg):
        if msg.buttons[0]:
            self.destroy_node()
            rclpy.shutdown()

    def run(self):
        while rclpy.ok():
            if self.obs_img is not None:
                img_path = os.path.join(self.args.topomap_images_dir, self.args.dir, f"{self.i}.png")
                self.obs_img.save(img_path)
                print(img_path)
                print("Published image", self.i)
                self.i += 1
                self.start_time = time.time()
                self.obs_img = None
            if time.time() - self.start_time > 2 * self.args.dt:
                print(f"Topic {self.args.image_topic} not publishing anymore. Shutting down...")
                break
            self.rate.sleep()

        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init()
    topo_map_creator = TopoMapCreator()
    topo_map_creator.run()

if __name__ == "__main__":
    main()
