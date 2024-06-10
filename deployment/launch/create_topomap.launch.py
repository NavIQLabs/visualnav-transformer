import os
import sys
import launch
from ament_index_python.packages import get_package_share_directory

import launch_ros.actions
from launch.actions import DeclareLaunchArgument

import yaml

def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        # Load the YAML file
        data = yaml.safe_load(file)
        return data

def generate_launch_description():
    package_dir = get_package_share_directory('visualnav_transformer')

    params_file = os.path.join(package_dir, 'params', 'explore.yaml')
    topomap_file = os.path.join(package_dir, 'params', 'topomap.yaml')
    
    topics_file = os.path.join(package_dir, 'params', 'topics.yaml')
    robot_file = os.path.join(package_dir, 'params', 'robot.yaml')

    topics = load_yaml_file(topics_file)
    robot_params = load_yaml_file(robot_file)
    
    
    ld = launch.LaunchDescription()

    topomap_node = launch_ros.actions.Node(
        package='visualnav_transformer',
        executable='create_topomap',
        name='create_topomap_node',
        parameters=[topomap_file, topics, robot_params],
        output='screen'
    )

    mode = sys.argv[-1].split('=')[-1]
    print(mode)

    ld.add_action(topomap_node)
    if mode == 'manual':
        teleop_node = launch_ros.actions.Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard_node',
            output='screen'
        )
        ld.add_action(teleop_node)
    else:
        explore_node = launch_ros.actions.Node(
            package='visualnav_transformer',
            executable='explore',
            name='explore_node',
            parameters=[params_file, topics, robot_params],
            output='screen'
        )

        pd_controller_node = launch_ros.actions.Node(
            package='visualnav_transformer',
            executable='pd_controller',
            name='pd_controller_node',
            parameters=[topics, robot_params],
            output='screen'
        )

        ld.add_action(explore_node)
        ld.add_action(pd_controller_node)
    return ld
    


    # if mode. == 'manual':
    #     return launch.LaunchDescription([
    #         launch_ros.actions.Node(
    #             package='visualnav_transformer',
    #             executable='create_topomap',
    #             name='create_topomap_node',
    #             parameters=[topomap_file, topics, robot_params],
    #             output='screen'
    #         ),
    #         launch_ros.actions.Node(
    #             package='teleop_twist_keyboard',
    #             executable='teleop_twist_keyboard',
    #             name='teleop_twist_keyboard_node',
    #             output='screen'
    #         )
    #     ])
    # else:
    #     return launch.LaunchDescription([
    #         launch_ros.actions.Node(
    #             package='visualnav_transformer',
    #             executable='create_topomap',
    #             name='create_topomap_node',
    #             parameters=[topomap_file, topics, robot_params],
    #             output='screen'
    #         ),
    #         launch_ros.actions.Node(
    #             package='visualnav_transformer',
    #             executable='explore',
    #             name='explore_node',
    #             parameters=[params_file, topics, robot_params],
    #             output='screen'
    #         ),
    #         launch_ros.actions.Node(
    #             package='visualnav_transformer',
    #             executable='pd_controller',
    #             name='pd_controller_node',
    #             parameters=[topics, robot_params],
    #             output='screen'
    #         )
    #     ])

if __name__ == '__main__':
    generate_launch_description()