import yaml

# ROS
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from topic_names import JOY_BUMPER_TOPIC

vel_msg = Twist()
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = 0.4
MAX_W = 0.8
VEL_TOPIC = robot_config["vel_teleop_topic"]
JOY_CONFIG_PATH = "../config/joystick.yaml"
with open(JOY_CONFIG_PATH, "r") as f:
	joy_config = yaml.safe_load(f)
DEADMAN_SWITCH = joy_config["deadman_switch"] # button index
LIN_VEL_BUTTON = joy_config["lin_vel_button"]
ANG_VEL_BUTTON = joy_config["ang_vel_button"]
RATE = 9

vel_pub = None
button = None
bumper = False


def callback_joy(data: Joy):
	"""Callback function for the joystick subscriber"""
	global vel_msg, button, bumper, vel_pub
	button = data.buttons[DEADMAN_SWITCH] 
	bumper_button = data.buttons[DEADMAN_SWITCH - 1]
	if button is not None: # hold down the dead-man switch to teleop the robot
		vel_msg.linear.x = MAX_V * data.axes[LIN_VEL_BUTTON]
		vel_msg.angular.z = MAX_W * data.axes[ANG_VEL_BUTTON]	
	else:
		vel_msg = Twist()
		vel_pub.publish(vel_msg)
	if bumper_button is not None:
		bumper = bool(data.buttons[DEADMAN_SWITCH - 1])
	else:
		bumper = False



def main():
	rclpy.init()
	node = rclpy.create_node("Joy2Locobot")
	vel_pub = node.create_publisher(Twist, VEL_TOPIC, 1)
	bumper_pub = node.create_publisher(Bool, JOY_BUMPER_TOPIC, 1)
	joy_sub = node.create_subscription(Joy, "joy", callback_joy)
	rate = node.create_rate(RATE)	
	print("Registered with master node. Waiting for joystick input...")
	while rclpy.ok():
		if button:
			print(f"Teleoperating the robot:\n {vel_msg}")
			vel_pub.publish(vel_msg)
			rate.sleep()
		bumper_msg = Bool()
		bumper_msg.data = bumper
		bumper_pub.publish(bumper_msg)
		if bumper:
			print("Bumper pressed!")


if __name__ == "__main__":
	main()

