import os
import pickle
from PIL import Image
import argparse
import tqdm
import yaml
import rosbag2_py

# utils
from vint_train.process_data.process_data_utils_ros2 import *


def main(args: argparse.Namespace):
    # load the config file
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recursively through all the folders and get the path of files with .bag extension in the args.input_dir
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in dirs:
            bag_files.append(os.path.join(root, file))

    if args.num_trajs >= 0:
        bag_files = bag_files[:args.num_trajs]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
            converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.split("/")[-2:])
  

        # Process the bag file
        bag_img_data, bag_traj_data = get_images_and_odom(
            reader,
            config[args.dataset_name]["imtopics"],
            config[args.dataset_name]["odomtopics"],
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
            bag_dir=bag_path
        )

        if bag_img_data is None or bag_traj_data is None:
            print(f"{bag_path} did not have the topics we were looking for. Skipping...")
            continue

        # remove backwards movement
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)
        # cut_trajs = zip(bag_img_data, bag_traj_data)
        # print(cut_trajs)
        print(f'{traj_name=}, {len(bag_img_data)=}, {len(cut_trajs[0][0])=}')
        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):

            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # save the image data to disk
            for j, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{j}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "-d", type=str, help="name of the dataset (must be in process_config.yaml)", required=True)
    parser.add_argument("--input-dir", "-i", type=str, help="path of the datasets with rosbags", required=True)
    parser.add_argument("--output-dir", "-o", default="../datasets/tartan_drive/", type=str, help="path for processed dataset (default: ../datasets/tartan_drive/)")
    parser.add_argument("--num-trajs", "-n", default=-1, type=int, help="number of bags to process (default: -1, all)")
    parser.add_argument("--sample-rate", "-s", default=4.0, type=float, help="sampling rate (default: 4.0 hz)")
    args = parser.parse_args()
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
