#!/usr/bin/env python3
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
import time
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from .utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

os.chdir(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")
class Args:
    pass

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        self.args = self.load_ros_params()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_queue = []
        self.model, self.model_params = self.load_model_and_params()
        self.topomap = self.load_topomap()
        self.goal_node = self.args.goal_node if self.args.goal_node != -1 else len(self.topomap) - 1
        self.closest_node = 0
        self.reached_goal = False

        self.create_subscription(Image, self.args.image_topic, self.image_callback, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, self.args.waypoint_topic, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, self.args.sampled_actions_topic, 1)
        self.goal_pub = self.create_publisher(Bool, self.args.reached_goal_topic, 1)
        self.actions_vis = self.create_publisher(MarkerArray, "/actions_vis", 1)
        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()

        self.rate = self.create_rate(self.args.frame_rate)

        self.get_logger().info("Using device: {}".format(self.device))
        self.get_logger().info("Registered with master node. Waiting for image observations...")
        self.navigation()

    def load_ros_params(self):
        # Declare parameters 
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('waypoint_topic', '/waypoint')
        self.declare_parameter('sampled_actions_topic', '/sampled_actions')
        self.declare_parameter('reached_goal_topic', '/topoplan/reached_goal')
        self.declare_parameter('model', 'nomad')
        self.declare_parameter('dir', 'topomap')
        self.declare_parameter('model_config_path', '../config/models.yaml')
        self.declare_parameter('topomap_images_dir', '../topomaps/images')
        self.declare_parameter('goal_node', -1)
        self.declare_parameter('robot_config_path', '../config/robot.yaml')
        self.declare_parameter('close_threshold', 3)
        self.declare_parameter('radius', 4)
        self.declare_parameter('num_samples', 8)
        self.declare_parameter('waypoint', 2)
        self.declare_parameter('max_v', 0.2)  # Default value as example
        self.declare_parameter('max_w', 0.4)  # Default value as example
        self.declare_parameter('frame_rate', 4)  # Default value as example
        args = Args()
        
        # List all parameters we expect
        parameters = [ 'image_topic', 'waypoint_topic', 'sampled_actions_topic', 'reached_goal_topic',
            'model', 'dir', 'model_config_path', 'topomap_images_dir', 
            'goal_node', 'robot_config_path', 'close_threshold', 'radius', 
            'num_samples', 'waypoint', 'max_v', 'max_w', 'frame_rate'
        ]
        
        # Load each parameter and assign it to the args object
        for param_name in parameters:
            # Assuming all parameters have been declared before this function is called
            param = self.get_parameter(param_name)
            param_value = param.get_parameter_value()

            # Check the type of the parameter and convert if necessary
            print(param_name,"-----", param_value.type)
            if param_value.type == 2:
                value = param_value.integer_value
            elif param_value.type == 3:
                value = param_value.double_value
            else:
                value = param_value.string_value
            setattr(args, param_name, value)

        return args
        
    def load_model_and_params(self):
        with open(self.args.model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

        ckpt_path = model_paths[self.args.model]["ckpt_path"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        model = load_model(ckpt_path, model_params, self.device).to(self.device)
        model.eval()

        return model, model_params
    

    def load_topomap(self):
        topomap_dir = os.path.join(self.args.topomap_images_dir, self.args.dir)
        topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0]))
        return [PILImage.open(os.path.join(topomap_dir, filename)) for filename in topomap_filenames]

    def image_callback(self, msg):
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.model_params["context_size"] + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def trajectory_vis(self, naction):
        trajectories = np.cumsum(naction, axis=1)* 0.12
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


    def navigation(self):
        if self.model_params["model_type"] == "nomad":
            num_diffusion_iters = self.model_params["num_diffusion_iters"]
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        
        # navigation loop
        while rclpy.ok():
            # EXPLORATION MODE
            chosen_waypoint = np.zeros(4)
            if len(self.context_queue) > self.model_params["context_size"]:
                if self.model_params["model_type"] == "nomad":
                    obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
                    obs_images = torch.split(obs_images, 3, dim=1)
                    obs_images = torch.cat(obs_images, dim=1) 
                    obs_images = obs_images.to(self.device)
                    mask = torch.zeros(1).long().to(self.device)  

                    start = max(self.closest_node - self.args.radius, 0)
                    end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                    goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(self.device) for g_img in self.topomap[start:end + 1]]
                    goal_image = torch.concat(goal_image, dim=0)

                    obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                    dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                    dists = to_numpy(dists.flatten())
                    min_idx = np.argmin(dists)
                    self.closest_node = min_idx + start
                    print("closest node:", self.closest_node)
                    sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
                    obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                    # infer action
                    with torch.no_grad():
                        # encoder vision features
                        if len(obs_cond.shape) == 2:
                            obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                        else:
                            obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                        
                        # initialize action from Gaussian noise
                        noisy_action = torch.randn(
                            (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
                        naction = noisy_action

                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        start_time = time.time()
                        for k in noise_scheduler.timesteps[:]:
                            # predict noise
                            noise_pred = self.model(
                                'noise_pred_net',
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )
                            # inverse diffusion step (remove noise)
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample
                        print("time elapsed:", time.time() - start_time)

                    naction = to_numpy(get_action(naction))
                    sampled_actions_msg = Float32MultiArray()
                    sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten())).tolist()
                    print("published sampled actions")
                    self.sampled_actions_pub.publish(sampled_actions_msg)

                    self.trajectory_vis(naction)
                    naction = naction[0] 
                    chosen_waypoint = naction[self.args.waypoint]
                elif (len(self.context_queue) > self.model_params["context_size"]):
                    start = max(self.closest_node - self.args.radius, 0)
                    end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                    distances = []
                    waypoints = []
                    batch_obs_imgs = []
                    batch_goal_data = []
                    for i, sg_img in enumerate(self.topomap[start: end + 1]):
                        transf_obs_img = transform_images(self.context_queue, self.model_params["image_size"])
                        goal_data = transform_images(sg_img, self.model_params["image_size"])
                        batch_obs_imgs.append(transf_obs_img)
                        batch_goal_data.append(goal_data)
                        
                    # predict distances and waypoints
                    batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
                    batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)

                    distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
                    distances = to_numpy(distances)
                    waypoints = to_numpy(waypoints)
                    # look for closest node
                    self.closest_node = np.argmin(distances)
                    # chose subgoal and output waypoints
                    if distances[self.closest_node] > self.args.close_threshold:
                        chosen_waypoint = waypoints[self.closest_node][self.args.waypoint]
                        sg_img = self.topomap[start + self.closest_node]
                    else:
                        chosen_waypoint = waypoints[min(
                            self.closest_node + 1, len(waypoints) - 1)][self.args.waypoint]
                        sg_img = self.topomap[start + min(self.closest_node + 1, len(waypoints) - 1)]     
            # RECOVERY MODE
            if self.model_params["normalize"]:
                chosen_waypoint[:2] *= (self.args.max_v / self.args.frame_rate)  
            waypoint_msg = Float32MultiArray()
            print("chosen waypoint:", chosen_waypoint)
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)
            reached_goal = bool(self.closest_node == self.goal_node)
            reached_goal_msg = Bool()
            reached_goal_msg.data = reached_goal
            self.goal_pub.publish(reached_goal_msg)
            if reached_goal:
                print("Reached goal! Stopping...")
            self.rate.sleep()


    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.reached_goal:
                self.get_logger().info("Reached goal! Stopping...")
                break
            self.rate.sleep()

def main(args=None):

    rclpy.init()
    node = ExplorationNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
