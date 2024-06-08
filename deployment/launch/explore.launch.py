import os
import launch
from ament_index_python.packages import get_package_share_directory

import launch_ros.actions
import yaml

def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        # Load the YAML file
        data = yaml.safe_load(file)
        return data

def generate_launch_description():
    package_dir = get_package_share_directory('visualnav_transformer')

    params_file = os.path.join(package_dir, 'params', 'navigation.yaml')
    topics_file = os.path.join(package_dir, 'params', 'topics.yaml')
    robot_file = os.path.join(package_dir, 'params', 'robot.yaml')

    topics = load_yaml_file(topics_file)
    robot_params = load_yaml_file(robot_file)
    
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='visualnav_transformer',
            executable='explore',
            name='explore_node',
            parameters=[params_file, topics, robot_params],
            output='screen'
        ),
        launch_ros.actions.Node(
            package='visualnav_transformer',
            executable='pd_controller',
            name='pd_controller_node',
            parameters=[params_file, topics, robot_params],
            output='screen'
        )
    ])

if __name__ == '__main__':
    generate_launch_description()