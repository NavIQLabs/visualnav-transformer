import argparse
import os
import shutil
import time
from utils import msg_to_pil
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
import threading

IMAGE_TOPIC = "/image_raw"
TOPOMAP_IMAGES_DIR = "../topomaps/images"

class TopoMapCreator(Node):
    def __init__(self, args):
        super().__init__("create_topomap")
        self.args = args
        self.obs_img = None
        self.publisher = self.create_publisher(Image, "/subgoals", 1)
        self.image_subscriber = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1)
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

    def init_directories(self):
        topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, self.args.dir)
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
                img_path = os.path.join(TOPOMAP_IMAGES_DIR, self.args.dir, f"{self.i}.png")
                self.obs_img.save(img_path)
                print(img_path)
                print("Published image", self.i)
                self.i += 1
                self.start_time = time.time()
                self.obs_img = None
            if time.time() - self.start_time > 2 * self.args.dt:
                print(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
                break
            self.rate.sleep()

        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic")
    parser.add_argument("--dir", "-d", default="topomap", type=str,
                        help="path to topological map images in ../topomaps/images directory (default: topomap)")
    parser.add_argument("--dt", "-t", default=1.0, type=float,
                        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)")
    args = parser.parse_args(args)

    rclpy.init()
    topo_map_creator = TopoMapCreator(args)
    topo_map_creator.run()

if __name__ == "__main__":
    main()
