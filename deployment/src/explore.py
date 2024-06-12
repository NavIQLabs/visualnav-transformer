import argparse
import os
import sys
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/train")
sys.path.append(f"{os.getcwd()}/src/diffusion_policy")
import threading
import numpy as np
import torch
from PIL import Image as PILImage
import cv2
import cv_bridge
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
os.chdir(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")

class Args:
    pass

class ExplorationNode(Node):
    def __init__(self):
        super().__init__("exploration_node")
        

        self.args = self.load_ros_params()
        self.model_config, self.model_paths = self.load_configurations()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_queue = []
        self.load_model()

        self.image_subscriber = self.create_subscription(Image, self.args.image_topic, self.callback_obs, 1)
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, self.args.waypoint_topic, 1)
        self.actions_publisher = self.create_publisher(Float32MultiArray, self.args.sampled_actions_topic, 1)
        self.actions_vis = self.create_publisher(MarkerArray, self.args.action_vis_topic, 1)
        self.rate = self.create_rate(self.args.frame_rate)

        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()
        
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info("Node initialized and waiting for image observations...")

    def load_ros_params(self):
        # Declare parameters 
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('waypoint_topic', '/waypoint')
        self.declare_parameter('sampled_actions_topic', '/sampled_actions')
        self.declare_parameter('action_vis_topic', '/actions_vis')
        self.declare_parameter('model', 'nomad')
        self.declare_parameter('model_config_path', '../config/models.yaml')
        self.declare_parameter('goal_node', -1)
        self.declare_parameter('robot_config_path', '../config/robot.yaml')
        self.declare_parameter('num_samples', 8)
        self.declare_parameter('waypoint', 2)
        self.declare_parameter('max_v', 0.2)  # Default value as example
        self.declare_parameter('max_w', 0.4)  # Default value as example
        self.declare_parameter('frame_rate', 4)  # Default value as example
        args = Args()
        
        # List all parameters we expect
        parameters = [ 'image_topic', 'waypoint_topic', 'sampled_actions_topic', "action_vis_topic",
            'model', 'model_config_path', 
            'goal_node', 'robot_config_path',
            'num_samples', 'waypoint', 'max_v', 'max_w', 'frame_rate'
        ]
        
        # Load each parameter and assign it to the args object
        for param_name in parameters:
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

    def load_configurations(self):
        with open(self.args.model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)
        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        return model_config, model_paths

    def load_model(self):
        ckpt_path = self.model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(ckpt_path):
            self.get_logger().info(f"Loading model from {ckpt_path}")
            self.model = load_model(ckpt_path, self.model_config, self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_config["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.model_config["context_size"] + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def run(self):
        while rclpy.ok():
            if len(self.context_queue) > self.model_config["context_size"]:
                self.process_images()
            self.rate.sleep()

    def process_images(self):
        obs_images = transform_images(self.context_queue, self.model_config["image_size"], center_crop=False)
        obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(self.device)
        fake_goal = torch.randn((1, 3, *self.model_config["image_size"])).to(self.device)
        mask = torch.ones(1).long().to(self.device)  # ignore the goal

        with torch.no_grad():
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            naction = self.infer_action(obs_cond)
            self.publish_actions(naction)

    def infer_action(self, obs_cond):
        # (B, obs_horizon * obs_dim)
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(self.args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

        naction = torch.randn((self.args.num_samples, self.model_config["len_traj_pred"], 2), device=self.device)
        self.noise_scheduler.set_timesteps(self.model_config["num_diffusion_iters"])
        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        return naction
    
    def trajectory_vis(self, naction):
        trajectories = np.cumsum(naction, axis=1) * 0.12
        marker_array = MarkerArray()

        for i, trajectory in enumerate(trajectories):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = i
            marker.scale.x = 0.05  # Line width
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red color
            
            # Set trajectory points
            for j, xy in enumerate(trajectory):
                p = Point()
                p.x = float(xy[0])
                p.y = float(xy[1])
                p.z = 0.0
                marker.points.append(p)
            
            marker_array.markers.append(marker)

        self.actions_vis.publish(marker_array)

    def publish_actions(self, naction):
        naction = to_numpy(get_action(naction))
        actions_msg = Float32MultiArray()
        actions_msg.data = np.concatenate((np.array([0]), naction.flatten())).tolist()
        self.actions_publisher.publish(actions_msg)
        self.get_logger().info("Published actions")

        self.trajectory_vis(naction)
        naction = naction[0] # change this based on heuristic

        chosen_waypoint = naction[self.args.waypoint]

        if self.model_config["normalize"]:
            chosen_waypoint *= (self.args.max_v / self.args.frame_rate)
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_publisher.publish(waypoint_msg)
        print("Published waypoint")

def main(args = None):
    rclpy.init()
    exploration_node = ExplorationNode()
    try:
        exploration_node.run()
    finally:
        exploration_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":

    main()