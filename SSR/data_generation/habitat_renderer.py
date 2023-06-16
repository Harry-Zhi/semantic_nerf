#!/usr/bin/env python3
import os, sys, argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import logging
import habitat_sim as hs
import numpy as np
import quaternion
import yaml
import json
from typing import Any, Dict, List, Tuple, Union
from imgviz import label_colormap
from PIL import Image
import matplotlib.pyplot as plt
import transformation
import imgviz
from datetime import datetime
import time
from settings import make_cfg

# Custom type definitions
Config = Dict[str, Any]
Observation = hs.sensor.Observation
Sim = hs.Simulator

def init_habitat(config) :
    """Initialize the Habitat simulator with sensors and scene file"""
    _cfg = make_cfg(config)
    sim = Sim(_cfg)
    sim_cfg = hs.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    # Note: all sensors must have the same resolution
    camera_resolution = [config["height"], config["width"]]
    sensors = {
        "color_sensor": {
            "sensor_type": hs.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, config["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": hs.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, config["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": hs.SensorType.SEMANTIC,
            "resolution": camera_resolution,
            "position": [0.0, config["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if config[sensor_uuid]:
            sensor_spec = hs.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = hs.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": hs.agent.ActionSpec(
            "move_forward", hs.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": hs.agent.ActionSpec(
            "turn_left", hs.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": hs.agent.ActionSpec(
            "turn_right", hs.agent.ActuationSpec(amount=30.0)
        ),
    }

    hs_cfg = hs.Configuration(sim_cfg, [agent_cfg])
    # sim = Sim(hs_cfg)

    if config["enable_semantics"]: # extract instance to class mapping function
        assert os.path.exists(config["instance2class_mapping"])
        with open(config["instance2class_mapping"], "r") as f:
            annotations = json.load(f)
        instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
        num_classes = len(annotations["classes"])
        label_colour_map = label_colormap()
        config["instance2semantic"] = instance_id_to_semantic_label_id
        config["classes"] = annotations["classes"]
        config["objects"] = annotations["objects"]

        config["num_classes"] = num_classes
        config["label_colour_map"] = label_colormap()
        config["instance_colour_map"] = label_colormap(500)


    # add camera intrinsic
    # hfov = float(agent_cfg.sensor_specifications[0].parameters['hfov']) * np.pi / 180.
    # https://aihabitat.org/docs/habitat-api/view-transform-warp.html
    # config['K'] = K
    # config['K'] = np.array([[fx, 0.0, 0.0], [0.0, fx, 0.0], [0.0, 0.0, 1.0]],
    #                        dtype=np.float64)

    # hfov = float(agent_cfg.sensor_specifications[0].parameters['hfov'])
    # fx = 1.0 / np.tan(hfov / 2.0)
    # config['K'] = np.array([[fx, 0.0, 0.0], [0.0, fx, 0.0], [0.0, 0.0, 1.0]],
    #                        dtype=np.float64)

    # Get the intrinsic camera parameters


    logging.info('Habitat simulator initialized')

    return sim, hs_cfg, config

def save_renders(save_path, observation, enable_semantic, suffix=""):
    save_path_rgb = os.path.join(save_path, "rgb")
    save_path_depth = os.path.join(save_path, "depth")
    save_path_sem_class = os.path.join(save_path, "semantic_class")
    save_path_sem_instance = os.path.join(save_path, "semantic_instance")

    if not os.path.exists(save_path_rgb):
        os.makedirs(save_path_rgb)
    if not os.path.exists(save_path_depth):
        os.makedirs(save_path_depth)
    if not os.path.exists(save_path_sem_class):
        os.makedirs(save_path_sem_class)
    if not os.path.exists(save_path_sem_instance):
        os.makedirs(save_path_sem_instance)

    cv2.imwrite(os.path.join(save_path_rgb, "rgb{}.png".format(suffix)), observation["color_sensor"][:,:,::-1])  # change from RGB to BGR for opencv write
    cv2.imwrite(os.path.join(save_path_depth, "depth{}.png".format(suffix)), observation["depth_sensor_mm"])

    if enable_semantic:
        cv2.imwrite(os.path.join(save_path_sem_class, "semantic_class{}.png".format(suffix)), observation["semantic_class"])
        cv2.imwrite(os.path.join(save_path_sem_class, "vis_sem_class{}.png".format(suffix)), observation["vis_sem_class"][:,:,::-1])

        cv2.imwrite(os.path.join(save_path_sem_instance, "semantic_instance{}.png".format(suffix)), observation["semantic_instance"])
        cv2.imwrite(os.path.join(save_path_sem_instance, "vis_sem_instance{}.png".format(suffix)), observation["vis_sem_instance"][:,:,::-1])


def render(sim, config):
    """Return the sensor observations and ground truth pose"""
    observation = sim.get_sensor_observations()

    # process rgb imagem change from RGBA to RGB
    observation['color_sensor'] = observation['color_sensor'][..., 0:3]
    rgb_img = observation['color_sensor']

    # process depth
    depth_mm = (observation['depth_sensor'].copy()*1000).astype(np.uint16)  # change meters to mm
    observation['depth_sensor_mm'] = depth_mm

    # process semantics
    if config['enable_semantics']:

        # Assuming the scene has no more than 65534 objects
        observation['semantic_instance'] = np.clip(observation['semantic_sensor'].astype(np.uint16), 0, 65535)
        # observation['semantic_instance'][observation['semantic_instance']==12]=0 # mask out certain instance
        # Convert instance IDs to class IDs


        # observation['semantic_classes'] = np.zeros(observation['semantic'].shape, dtype=np.uint8)
        # TODO make this conversion more efficient
        semantic_class = config["instance2semantic"][observation['semantic_instance']]
        semantic_class[semantic_class < 0] = 0

        vis_sem_class = config["label_colour_map"][semantic_class]
        vis_sem_instance = config["instance_colour_map"][observation['semantic_instance']]  # may cause error when having more than 255 instances in the scene

        observation['semantic_class'] = semantic_class.astype(np.uint8)
        observation["vis_sem_class"] = vis_sem_class.astype(np.uint8)
        observation["vis_sem_instance"] = vis_sem_instance.astype(np.uint8)

        # del observation["semantic_sensor"]

    # Get the camera ground truth pose (T_HC) in the habitat frame from the
    # position and orientation
    t_HC = sim.get_agent(0).get_state().position
    q_HC = sim.get_agent(0).get_state().rotation
    T_HC = transformation.combine_pose(t_HC, q_HC)

    observation['T_HC'] = T_HC
    observation['T_WC'] = transformation.Thc_to_Twc(T_HC)

    return observation

def set_agent_position(sim, pose):
    # Move the agent
    R = pose[:3, :3]
    orientation_quat = quaternion.from_rotation_matrix(R)
    t = pose[:3, 3]
    position = t

    orientation = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
    agent = sim.get_agent(0)
    agent_state = hs.agent.AgentState(position, orientation)
    # agent.set_state(agent_state, reset_sensors=False)
    agent.set_state(agent_state)

def main():
    parser = argparse.ArgumentParser(description='Render Colour, Depth, Semantic, Instance labeling from Habitat-Simultation.')
    parser.add_argument('--config_file', type=str,
                        default="./data_generation/replica_render_config_vMAP.yaml",
                        help='the path to custom config file.')
    args = parser.parse_args()

    """Initialize the config dict and Habitat simulator"""
    # Read YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config["save_path"] = os.path.join(config["save_path"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    T_wc = np.loadtxt(config["pose_file"]).reshape(-1, 4, 4)
    Ts_cam2world = T_wc

    print("-----Initialise and Set Habitat-Sim-----")
    sim, hs_cfg, config = init_habitat(config)
    # Set agent state
    sim.initialize_agent(config["default_agent"])

    """Set agent state"""
    print("-----Render Images from Habitat-Sim-----")
    with open(os.path.join(config["save_path"], 'render_config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    start_time = time.time()
    total_render_num = Ts_cam2world.shape[0]
    for i in range(total_render_num):
        if i % 100 == 0 :
            print("Rendering Process: {}/{}".format(i, total_render_num))
        set_agent_position(sim, transformation.Twc_to_Thc(Ts_cam2world[i]))

        # replica mode
        observation = render(sim, config)
        save_renders(config["save_path"], observation, config["enable_semantics"], suffix="_{}".format(i))

    end_time = time.time()
    print("-----Finish Habitat Rendering, Showing Trajectories.-----")
    print("Average rendering time per image is {} seconds.".format((end_time-start_time)/Ts_cam2world.shape[0]))

if __name__ == "__main__":
    main()


