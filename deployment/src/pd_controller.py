import numpy as np
import yaml
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC
from ros_data import ROSData
from utils import clip_angle

class PDControllerNode(Node):
    def __init__(self):
        super().__init__("pd_controller")
        # Load configuration
        self.load_config()
        
        # Initialize ROS data class for waypoints
        self.waypoint = ROSData(self, 1, name="waypoint")
        
        # Subscribers
        self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self.callback_drive, 1)
        self.create_subscription(Bool, REACHED_GOAL_TOPIC, self.callback_reached_goal, 1)
        
        # Publisher
        self.vel_pub = self.create_publisher(Twist, self.config['vel_navi_topic'], 1)
        
        # PD Controller state
        self.reached_goal = False
        
        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()

        self.get_logger().info("PD Controller Node initialized successfully")

    def load_config(self):
        config_path = "../config/robot.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def pd_controller(self, waypoint_data):
        dx, dy = waypoint_data[:2]
        v = dx / self.config['frame_rate']
        w = np.arctan2(dy, dx) / self.config['frame_rate']
        v = np.clip(v, -self.config['max_v'], self.config['max_v'])
        w = np.clip(w, -self.config['max_w'], self.config['max_w'])
        return v, w

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
