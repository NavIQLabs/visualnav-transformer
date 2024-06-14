import sys
import os
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")
sys.path.append(f"{os.getcwd()}/src/visualnav-transformer/train")
sys.path.append(f"{os.getcwd()}/src/diffusion_policy")
import numpy as np
import yaml
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from ros_data import ROSData
from utils import clip_angle
os.chdir(f"{os.getcwd()}/src/visualnav-transformer/deployment/src")


class Args:
    pass

class PDControllerNode(Node):
    def __init__(self):
        super().__init__("pd_controller")
        
        self.args = self.load_ros_params()

        # Initialize ROS data class for waypoints
        self.waypoint = ROSData(self, 1, name="waypoint")
        
        # Subscribers
        self.create_subscription(Float32MultiArray, self.args.waypoint_topic, self.callback_drive, 1)
        self.create_subscription(Bool, self.args.reached_goal_topic, self.callback_reached_goal, 1)
        
        # Publisher
        self.vel_pub = self.create_publisher(Twist, self.args.vel_navi_topic, 1)
        
        # PD Controller state
        self.reached_goal = False
        
        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()

        self.get_logger().info("PD Controller Node initialized successfully")

    def load_ros_params(self):

        args = Args()
        self.declare_parameter('waypoint_topic', '/waypoint')
        self.declare_parameter('reached_goal_topic', '/topoplan/reached_goal')
        self.declare_parameter('max_v', 0.2)
        self.declare_parameter('max_w', 0.4)
        self.declare_parameter('frame_rate', 10)
        self.declare_parameter('vel_navi_topic', '/cmd_vel_mux/input/navi')

        parameters = ['waypoint_topic', 'reached_goal_topic', 'max_v', 'max_w', 'frame_rate', 'vel_navi_topic']
        
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

    def pd_controller(self, waypoint_data):
        dx, dy = waypoint_data[:2]
        v = dx / self.args.frame_rate
        w = np.arctan2(dy, dx) / self.args.frame_rate
        v = np.clip(v, -self.args.max_v, self.args.max_v)
        w = np.clip(w, -self.args.max_w, self.args.max_w)
        return v/1, w * 2

    def callback_drive(self, msg):
        self.waypoint.set(msg.data)
        self.get_logger().info("Waypoint updated")

    def callback_reached_goal(self, msg):
        self.reached_goal = msg.data
        self.get_logger().info("Goal status updated: {}".format(self.reached_goal))

    def run(self):
        rate = self.create_rate(10)
        while rclpy.ok():
            if self.reached_goal:
                self.vel_pub.publish(Twist())  # Stop the robot
                self.get_logger().info("Reached goal! Stopping...")
                break
            if self.waypoint.is_valid(verbose=True):
                v, w = self.pd_controller(self.waypoint.get())
                vel_msg = Twist()
                vel_msg.linear.x = v
                vel_msg.angular.z = w
                self.vel_pub.publish(vel_msg)
                self.get_logger().info(f"Publishing velocity - Linear: {v}, Angular: {w}")
            rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
