#!/bin/zsh

# Source the workspace setup.zsh
source install/setup.zsh

# Create a new tmux session named "visualnav"
tmux new-session -d -s visualnav

# Split the window vertically
tmux split-window -v

# Run pd_controller in the second pane
tmux send-keys -t visualnav:0.1 'source /opt/ros/humble/setup.zsh' Enter
tmux send-keys -t visualnav:0.1 'source install/setup.zsh' Enter
tmux send-keys -t visualnav:0.1 'ros2 run visualnav_transformer pd_controller' Enter

# Run visualnav_transformer navigate in the first pane
tmux send-keys -t visualnav:0.0 'source /opt/ros/humble/setup.zsh' Enter
tmux send-keys -t visualnav:0.0 'source install/setup.zsh' Enter
tmux send-keys -t visualnav:0.0 'ros2 run visualnav_transformer navigate' Enter

# Attach to the tmux session
tmux attach-session -t visualnav
