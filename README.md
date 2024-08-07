# General Navigation Models: GNM, ViNT and NoMaD

**Contributors**: Dhruv Shah, Ajay Sridhar, Nitish Dashora, Catherine Glossop, Kyle Stachowicz, Arjun Bhorkar, Kevin Black, Noriaki Hirose, Sergey Levine

_Berkeley AI Research_

[Project Page](https://general-navigation-models.github.io) | [Citing](https://github.com/robodhruv/visualnav-transformer#citing) | [Pre-Trained Models](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

General Navigation Models are general-purpose goal-conditioned visual navigation policies trained on diverse, cross-embodiment training data, and can control many different robots in zero-shot. They can also be efficiently fine-tuned, or adapted, to new robots and downstream tasks. Our family of models is described in the following research papers (and growing):
1. [GNM: A General Navigation Model to Drive Any Robot](https://sites.google.com/view/drive-any-robot) (_October 2022_, presented at ICRA 2023)
2. [ViNT: A Foundation Model for Visual Navigation](https://general-navigation-models.github.io/vint/index.html) (_June 2023_, presented at CoRL 2023)
3. [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html) (_October 2023_)

## Overview
This repository contains code for training our family of models with your own data, pre-trained model checkpoints, as well as example code to deploy it on a TurtleBot2/LoCoBot robot. The repository follows the organization from [GNM](https://github.com/PrieureDeSion/drive-any-robot).

- `./train/train.py`: training script to train or fine-tune the ViNT model on your custom data.
- `./train/vint_train/models/`: contains model files for GNM, ViNT, and some baselines.
- `./train/process_*.py`: scripts to process rosbags or other formats of robot trajectories into training data.
- `./deployment/launch/create_topomap.launch.py`: script to convert a ROS bag of a demo trajectory into a topological graph that the robot can use to navigate.
- `./deployment/launch/navigation.launch.py`: script that deploys a trained GNM/ViNT/NoMaD model on the robot to navigate to a desired goal in the generated topological graph. Please see relevant sections below for configuration settings.
- `./deployment/launch/explore.launch.py`: script that deploys a trained NoMaD model on the robot to randomly explore its environment. Please see relevant sections below for configuration settings.

## Train

This subfolder contains code for processing datasets and training models from your own data.

### Pre-requisites

The codebase assumes access to a workstation running Ubuntu (tested on 18.04 and 20.04), Python 3.7+, and a GPU with CUDA 10+. It also assumes access to conda, but you can modify it to work with other virtual environment packages, or a native setup.
### Setup
Run the commands below inside the `vint_release/` (topmost) directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate vint_train
    ```
3. Install the vint_train packages:
    ```bash
    pip install -e train/
    ```
4. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```


### Data-Wrangling
In the [papers](https://general-navigation-models.github.io), we train on a combination of publicly available and unreleased datasets. Below is a list of publicly available datasets used for training; please contact the respective authors for access to the unreleased data.
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2 (Modified)](https://drive.google.com/drive/folders/1xrNvMl5q92oWed99noOt_UhqQnceJYV0?usp=share_link)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

We recommend you to download these (and any other datasets you may want to train on) and run the processing steps below.

#### Data Processing 

We provide some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags_ros2.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
2. Run `data_split.py` on your dataset folder with the relevant args.

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:
- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `vint_release/train/vint_train/data/data_splits/`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### Training your General Navigation Models
Run this inside the `vint_release/train` directory:
```bash
python train.py -c <path_of_train_config_file>
```
The premade config yaml files are in the `train/config` directory. 

#### Custom Config Files
You can use one of the premade yaml files as a starting point and change the values as you need. `config/vint.yaml` is good choice since it has commented arguments. `config/defaults.yaml` contains the default config values (don't directly train with this config file since it does not specify any datasets for training).

#### Custom Datasets
Make sure your dataset and data-split directory follows the structures provided in the [Data Processing](#data-processing) section. Locate `train/vint_train/data/data_config.yaml` and append the following:

```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

Locate your training config file and add the following text under the `datasets` argument (feel free to change the values of `end_slack`, `goals_per_obs`, and `negative_mining`):
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # how many timesteps to cut off from the end of each trajectory  (in case many trajectories end in collisions)
    goals_per_obs: 1 # how many goals are sampled per observation
    negative_mining: True # negative mining from the ViNG paper (Shah et al.)
```

#### Training your model from a checkpoint
Instead of training from scratch, you can also load an existing checkpoint from the published results.
Add `load_run: <project_name>/<log_run_name>`to your .yaml config file in `vint_release/train/config/`. The `*.pth` of the file you are loading to be saved in this file structure and renamed to “latest”: `vint_release/train/logs/<project_name>/<log_run_name>/latest.pth`. This makes it easy to train from the checkpoint of a previous run since logs are saved this way by default. Note: if you are loading a checkpoint from a previous run, check for the name the run in the `vint_release/train/logs/<project_name>/`, since the code appends a string of the date to each run_name specified in the config yaml file of the run to avoid duplicate run names. 


If you want to use our checkpoints, you can download the `*.pth` files from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


## Deployment
This subfolder contains code to load a pre-trained ViNT and deploy it on the open-source [LoCoBot indoor robot platform](http://www.locobot.org/) with a [NVIDIA Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=asc_df_B0BZJTQ5YP/?tag=hyprod-20&linkCode=df0&hvadid=652427572954&hvpos=&hvnetw=g&hvrand=12520404772764575478&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1013585&hvtargid=pla-2112361227514&psc=1&gclid=CjwKCAjw4P6oBhBsEiwAKYVkq7dqJEwEPz0K-H33oN7MzjO0hnGcAJDkx2RdT43XZHdSWLWHKDrODhoCmnoQAvD_BwE). It can be easily adapted to be run on alternate robots, and researchers have been able to independently deploy it on the following robots – Clearpath Jackal, DJI Tello, Unitree A1, TurtleBot2, Vizbot – and in simulated environments like CARLA.

### LoCoBot Setup

This software was tested on a LoCoBot running Ubuntu 20.04 and docker running ubuntu 22.04.


#### Software Installation (in this order)
1. ROS: [ros-humble](https://docs.ros.org/en/humble/Installation.html)
2. ROS packages: 
    ```bash
    sudo apt-get install ros-humble-usb-cam ros-humble-joy
    ```
3. Create a Workspace
   ```bash
   mkdir -p gnm_ws/src
   cd gnm_ws/src
   git clone -b humble-dev git@github.com:NavIQLabs/visualnav-transformer.git
   ```
4. Install python dependencies
   ```bash
   pip install -r visualnav-transformer/requirements.txt
5. Install the `vint_train` packages (run this inside the `vint_release/` directory):
    ```bash
    pip install -e train/
    ```
6. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
7. Build the packages
   ```bash
   cd ~/gnm_ws
   colcon build --symlink-install
   ```

### Loading the model weights

Save the model weights *.pth file in `vint_release/deployment/model_weights` folder. Our model's weights are in [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).

### Collecting a Topological Map

_Make sure to run these scripts inside the `vint_release/deployment/src/` directory._


This section discusses a simple way to create a topological map of the target environment for deployment. For simplicity, we will use the robot in “path-following” mode, i.e. given a single trajectory in an environment, the task is to follow the same trajectory to the goal. The environment may have new/dynamic obstacles, lighting variations etc.

#### Record the ros2 bag: 
```bash
ros2 bag record # followed by the topics to be recorded for training such as odom and images.
```

#### Make the topological map: 
1. `ros2 launch visualnav_transformer create_topomap.launch.py`: This launch file opens the navigate and pd_controller node.
2. `ros2 run gopro_interface gopro`: This will run the gopro interface.
or
3. `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx camera_name=zedx` This will run zedx interface.
be sure to change the topics in topics.yaml present in the params folder.

### Running the model 
#### Navigation
1. `ros2 launch visualnav_transformer navigation.launch.py`: This launch file opens the navigate and pd_controller node.
2. `ros2 run gopro_interface gopro`: This will run the gopro interface.
or
3. `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx camera_name=zedx` This will run zedx interface.
be sure to change the topics in topics.yaml present in the params folder.

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `vint_release/deployment/config/models.yaml` file. In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `vint_release/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `vint_release/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

#### Exploration
_Make sure to run this script inside the `vint_release/deployment/src/` directory._

1. `ros2 launch visualnav_transformer explore.launch.py`: This launch file opens the navigate and pd_controller node.
2. `ros2 run gopro_interface gopro`: This will run the gopro interface.
or
3. `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx camera_name=zedx` This will run zedx interface.
be sure to change the topics in topics.yaml present in the params folder.

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `vint_release/deployment/config/models.yaml` file (note that only NoMaD works for exploration). In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `vint_release/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `vint_release/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

### Adapting this code to different robots

We hope that this codebase is general enough to allow you to deploy it to your favorite ROS-based robots. You can change the robot configuration parameters in `vint_release/deployment/config/robot.yaml`, like the max angular and linear velocities of the robot and the topics to publish to teleop and control the robot. Please feel free to create a Github Issue or reach out to the authors at shah@cs.berkeley.edu.


## Citing
```
@inproceedings{shah2022gnm,
  author    = {Dhruv Shah and Ajay Sridhar and Arjun Bhorkar and Noriaki Hirose and Sergey Levine},
  title     = {{GNM: A General Navigation Model to Drive Any Robot}},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.03370}
}

@inproceedings{shah2023vint,
  title     = {Vi{NT}: A Foundation Model for Visual Navigation},
  author    = {Dhruv Shah and Ajay Sridhar and Nitish Dashora and Kyle Stachowicz and Kevin Black and Noriaki Hirose and Sergey Levine},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2306.14846}
}

@article{sridhar2023nomad,
  author  = {Ajay Sridhar and Dhruv Shah and Catherine Glossop and Sergey Levine},
  title   = {{NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration}},
  journal = {arXiv pre-print},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.xxxx}
}
```
