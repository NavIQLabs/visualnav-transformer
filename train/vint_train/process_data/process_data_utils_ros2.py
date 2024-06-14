import numpy as np
import io
import os
import rosbag2_py
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF
import yaml
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as rosImage
from rclpy.serialization import deserialize_message

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3

def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    return images

# Define custom image processing functions based on your needs

def process_tartan_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img


def process_locobot_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image


def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img

def process_isaac_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image.
    Resizes the image to a fixed size.
    """

    img = deserialize_message(msg, rosImage)

    # Convert the PIL image to a numpy array
    np_img = np.array(img.data).reshape(img.height, img.width,  3)

    # Convert the numpy array back to a PIL image
    img = Image.fromarray(np_img)

    # Resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)  # Use Image.ANTIALIAS for high-quality resizing
    # img.show()
    
    return img



############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image


#######################################################################

def process_odom(odom_list: List, odom_process_func: Any, ang_offset: float = 0.0) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}

# Define custom odometry processing functions based on your needs
def get_metadata(bag_dir):
    metadata_file = os.path.join(bag_dir, 'metadata.yaml')
    if not os.path.exists(metadata_file):
        print(f'Metadata file not found: {metadata_file}')
        return None
    with open(metadata_file, 'r') as file:
        metadata = file.read()
    return yaml.safe_load(metadata)


def get_images_and_odom(
    reader: rosbag2_py.SequentialReader,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
    bag_dir: str = ""
):
    """
    Get image and odom data from a bag file using rosbag2_py
    """
    # Initialize data containers
    img_data = []
    traj_data = []
    synced_imdata = []
    synced_odomdata = []

    currtime = get_metadata(bag_dir)['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch']

    # Define helper functions to find the required topics
    def find_topic_info(topics, target_topics):
        for topic in topics:
            if topic.name in target_topics:
                return topic
        return None

    # Extract topics info
    all_topics = reader.get_all_topics_and_types()
    imtopic_info = find_topic_info(all_topics, imtopics)
    odomtopic_info = find_topic_info(all_topics, odomtopics)

    # Validate topics
    if imtopic_info is None or odomtopic_info is None:
        return None, None  # Both topics must be present
    
    curr_imdata = None
    curr_odomdata = None
    # Read messages
    while reader.has_next():
        (topic, msg, t) = reader.read_next()
        if topic == imtopic_info.name:
            curr_imdata = msg
        elif topic == odomtopic_info.name:
            curr_odomdata = msg
        # Check if data should be processed based on the sampling rate
        if ((t- currtime)/1_000_000_000) >= 1.0 / rate:
            # print("synccccc")
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                currtime = t

    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )
    return img_data, traj_data


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    odom_msg = deserialize_message(serialized_message=odom_msg, message_type=Odometry)

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw


############ Add custom odometry processing functions here ############


#######################################################################



def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_list: List[Image.Image],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cut out non-positive velocity segments of the trajectory
    Args:
        traj_type: type of trajectory to cut
        img_list: list of images
        traj_data: dictionary of position and yaw data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        cut_trajs: list of cut trajectories
        start_times: list of start times of the cut trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
        else:
            print("backwardsssssssssssss")
    return cut_trajs


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data
