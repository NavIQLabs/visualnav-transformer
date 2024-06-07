import argparse
import os
import threading
import numpy as np
import torch
from PIL import Image as PILImage
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC

class ExplorationNode(Node):
    def __init__(self, args):
        super().__init__("exploration_node")
        self.args = args
        self.robot_config, self.model_config, self.model_paths = self.load_configurations()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_queue = []
        self.load_model()

        self.image_subscriber = self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 1)
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.actions_publisher = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.rate = self.create_rate(self.robot_config["frame_rate"])

        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()
        
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info("Node initialized and waiting for image observations...")

    def load_configurations(self):
        with open("../config/robot.yaml", "r") as f:
            robot_config = yaml.safe_load(f)

        with open("../config/models.yaml", "r") as f:
            model_paths = yaml.safe_load(f)
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        return robot_config, model_config, model_paths

    def load_model(self):
        ckpt_path = self.model_paths[args.model]["ckpt_path"]
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
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        naction = torch.randn((self.args.num_samples, self.model_config["len_traj_pred"], 2), device=self.device)
        self.noise_scheduler.set_timesteps(self.model_config["num_diffusion_iters"])
        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        return naction

    def publish_actions(self, naction):
        naction = to_numpy(get_action(naction))
        actions_msg = Float32MultiArray()
        actions_msg.data = np.concatenate((np.array([0]), naction.flatten())).tolist()
        self.actions_publisher.publish(actions_msg)
        self.get_logger().info("Published actions")

        naction = naction[0] # change this based on heuristic

        chosen_waypoint = naction[self.args.waypoint]

        if self.model_config["normalize"]:
            chosen_waypoint *= (self.robot_config["max_v"] / self.robot_config["frame_rate"])
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_publisher.publish(waypoint_msg)
        print("Published waypoint")

def main(args):
    rclpy.init()
    exploration_node = ExplorationNode(args)
    try:
        exploration_node.run()
    finally:
        exploration_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument("--model", default="nomad", help="Model name")
    parser.add_argument("--waypoint", type=int, default=2, help="Waypoint index")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples")
    args = parser.parse_args()
    main(args)