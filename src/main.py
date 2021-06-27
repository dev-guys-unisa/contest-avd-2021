#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
import controller2d
import configparser 
import local_planner
import behavioural_planner
import cv2
import json 
from math import sin, cos, pi, tan, sqrt
from traffic_light_detector import TrafficLightDetector
from traffic_light_detection_module.postprocessing import draw_boxes
from utils import transform_world_to_ego_frame, compute_angle_diff, obstacle_to_world, check_obstacle_future_intersection

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/../..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.sensor import Camera
from carla.image_converter import labels_to_array, depth_to_array, to_bgra_array
from carla.planner.city_track import CityTrack
import carla.image_converter as image_converter


###############################################################################
# CONFIGURABLE PARAMETERS DURING EXAM
###############################################################################
PLAYER_START_INDEX     = 16    # spawn index for player
DESTINATION_INDEX      = 106    # Setting a Destination HERE
NUM_PEDESTRIANS        = 50   # total number of pedestrians to spawn
NUM_VEHICLES           = 30   # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0    # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0    # seed for vehicle spawn randomizer
###############################################################################

###############################################################################
# DECLARATION OF USEFUL CONSTANTS
###############################################################################
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 5000.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
LOOKAHEAD_PEDESTRIAN    = 10 # distance (m) within check if there is some pedestrians to be avoided during the path
LOOKAHEAD_VEHICLE       = 10 # distance (m) within check if there is some vehicles to be avoided during the path 

NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 16.0              # m
BP_LOOKAHEAD_TIME      = 1.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 2.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 5.0              # m Default: 3.5
LEAD_VEHICLE_LOOKAHEAD = 15.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the 
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural number.

DESIRED_SPEED          = 6                # maximum speed (m/s) to reach in straightforward paths
TURN_SPEED             = 2.5              # maximum speed (m/s) to reach during turns

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/controller_output/'

# Path to the configuration file of the sensors
CAMERA_FILENAME = os.path.dirname(os.path.realpath(__file__)) + '/settings/cams.json'

def make_carla_settings(args):
    """
    Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)

    # Retrieve the parameters for the sensors configuration and declare them
    with open(CAMERA_FILENAME, 'r') as cams_config_file:
        cams_config = json.load(cams_config_file)

    camera1_x_pos = cams_config['cam_front']['position'][0]
    camera1_y_pos = cams_config['cam_front']['position'][1]
    camera1_z_pos = cams_config['cam_front']['position'][2] 
    camera1_width = cams_config['cam_front']['img_size'][0]
    camera1_height = cams_config['cam_front']['img_size'][1]
    camera1_fov = cams_config['cam_front']['field_of_view']
    camera1_roll = cams_config['cam_front']['orientation'][0]
    camera1_pitch = cams_config['cam_front']['orientation'][1]
    camera1_yaw = cams_config['cam_front']['orientation'][2]

    camera2_x_pos = cams_config['cam_tl']['position'][0]
    camera2_y_pos = cams_config['cam_tl']['position'][1]
    camera2_z_pos = cams_config['cam_tl']['position'][2] 
    camera2_width = cams_config['cam_tl']['img_size'][0]
    camera2_height = cams_config['cam_tl']['img_size'][1]
    camera2_fov = cams_config['cam_tl']['field_of_view']
    camera2_roll = cams_config['cam_tl']['orientation'][0]
    camera2_pitch = cams_config['cam_tl']['orientation'][1]
    camera2_yaw = cams_config['cam_tl']['orientation'][2]

    camera3_x_pos = cams_config['cam_depth_TL']['position'][0]
    camera3_y_pos = cams_config['cam_depth_TL']['position'][1]
    camera3_z_pos = cams_config['cam_depth_TL']['position'][2] 
    camera3_width = cams_config['cam_depth_TL']['img_size'][0]
    camera3_height = cams_config['cam_depth_TL']['img_size'][1]
    camera3_fov = cams_config['cam_depth_TL']['field_of_view']
    camera3_roll = cams_config['cam_depth_TL']['orientation'][0]
    camera3_pitch = cams_config['cam_depth_TL']['orientation'][1]
    camera3_yaw = cams_config['cam_depth_TL']['orientation'][2]

    camera4_x_pos = cams_config['cam_depth_front']['position'][0]
    camera4_y_pos = cams_config['cam_depth_front']['position'][1]
    camera4_z_pos = cams_config['cam_depth_front']['position'][2] 
    camera4_width = cams_config['cam_depth_front']['img_size'][0]
    camera4_height = cams_config['cam_depth_front']['img_size'][1]
    camera4_fov = cams_config['cam_depth_front']['field_of_view']
    camera4_roll = cams_config['cam_depth_front']['orientation'][0]
    camera4_pitch = cams_config['cam_depth_front']['orientation'][1]
    camera4_yaw = cams_config['cam_depth_front']['orientation'][2]

    #############################################
    # SENSORS DECLARATION
    #############################################

    # RGB Camera for front
    camera0 = Camera("CameraRGB_Front")
    camera0.set_image_size(camera1_width, camera1_height)
    camera0.set(FOV = camera1_fov)
    camera0.set_position(camera1_x_pos, camera1_y_pos, camera1_z_pos)
    camera0.set_rotation(camera1_pitch, camera1_yaw, camera1_roll)

    # RGB Camera for Traffic Light Detection
    camera1 = Camera("CameraRGB_TL")
    camera1.set_image_size(camera2_width, camera2_height)
    camera1.set(FOV = camera2_fov)
    camera1.set_position(camera2_x_pos, camera2_y_pos, camera2_z_pos)
    camera1.set_rotation(camera2_pitch, camera2_yaw, camera2_roll)
    
    # DEPTH Camera for Traffic Light Detection
    camera2 = Camera("CameraDEPTH_TL", PostProcessing='Depth')
    camera2.set_image_size(camera3_width, camera3_height)
    camera2.set(FOV = camera3_fov)
    camera2.set_position(camera3_x_pos, camera3_y_pos, camera3_z_pos)
    camera2.set_rotation(camera3_pitch, camera3_yaw, camera3_roll)

    # DEPTH Camera for front
    camera3 = Camera("CameraDEPTH_FRONT", PostProcessing='Depth')
    camera3.set_image_size(camera4_width, camera4_height)
    camera3.set(FOV = camera4_fov)
    camera3.set_position(camera4_x_pos, camera4_y_pos, camera4_z_pos)
    camera3.set_rotation(camera4_pitch, camera4_yaw, camera4_roll)

    settings.add_sensor(camera0)
    settings.add_sensor(camera1)
    settings.add_sensor(camera2)
    settings.add_sensor(camera3)
    ######################################  

    return settings

class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    z   =  measurement.player_measurements.transform.location.z

    pitch = math.radians(measurement.player_measurements.transform.rotation.pitch)
    roll = math.radians(measurement.player_measurements.transform.rotation.roll)
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, z, pitch, roll, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement, 
                             prev_collision_vehicles, 
                             prev_collision_pedestrians,
                             prev_collision_other):
    """Obtains collision flag from player. Check if any of the three collision
    metrics (vehicles, pedestrians, others) from the player are true, if so the
    player has collided to something.

    Note: From the CARLA documentation:

    "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
    annotating undesired collision due to mistakes in the AI of non-player
    agents."
    """
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file: 
        collision_file.write(str(sum(collided_list)))

def make_correction(waypoint,previuos_waypoint, desired_speed):
    dx = waypoint[0] - previuos_waypoint[0]
    dy = waypoint[1] - previuos_waypoint[1]

    if dx < 0:
        moveY = -1.5
    elif dx > 0:
        moveY = 1.5
    else:
        moveY = 0

    if dy < 0:
        moveX = 1.5
    elif dy > 0:
        moveX = -1.5
    else:
        moveX = 0
    
    waypoint_on_lane = waypoint
    waypoint_on_lane[0] += moveX
    waypoint_on_lane[1] += moveY
    waypoint_on_lane[2] = desired_speed

    return waypoint_on_lane

def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """
    
    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)

        # recover depth camera informations
        depth_info = {"CameraDEPTH_TL":{}, "CameraDEPTH_FRONT":{}}
        for s in settings._sensors:
            if s.SensorName=="CameraDEPTH_TL":
                depth_info["CameraDEPTH_TL"] = \
                            {"w":s.ImageSizeX, "h": s.ImageSizeY, "fov": s.FOV, "x": s.PositionX, "y": s.PositionY,
                             "z": s.PositionZ, "roll": s.RotationRoll, "pitch": s.RotationPitch, "yaw": s.RotationYaw}
            elif s.SensorName=="CameraDEPTH_FRONT":
                depth_info["CameraDEPTH_FRONT"] = \
                            {"w":s.ImageSizeX, "h": s.ImageSizeY, "fov": s.FOV, "x": s.PositionX, "y": s.PositionY,
                             "z": s.PositionZ, "roll": s.RotationRoll, "pitch": s.RotationPitch, "yaw": s.RotationYaw}

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Refer to the player start folder in the WorldOutliner to see the 
        # player start information
        player_start = PLAYER_START_INDEX

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        #############################################
        # DETECTOR DECLARATION
        #############################################
        detector = TrafficLightDetector()

        #############################################
        # LOAD CONFIGURATIONS
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        ENABLE_LIVE_PLOT = demo_opt.get('live_plotting', 'true').capitalize()
        ENABLE_LIVE_PLOT = ENABLE_LIVE_PLOT == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        if ENABLE_LIVE_PLOT: live_plot_timer = Timer(live_plot_period)
        
        # Settings Mission Planner
        mission_planner = CityTrack("Town01")

        #############################################
        # DETERMINE SIMULATION AVERAGE TIMESTEP (AND TOTAL FRAMES)
        #############################################
        # Ensure at least one frame is used to compute average timestep
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.
        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        # This mainly applies for simulations that are in synchronous mode.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Computes the average timestep based on several initial iterations
        sim_duration = 0
        for i in range(num_iterations):
            # Gather current data
            measurement_data, sensor_data = client.read_data()
            # Send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 - sim_start_stamp

        # Outputs average simulation timestep and computes how many frames
        # will elapse before the simulation should end based on various
        # parameters that we set in the beginning.
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        #############################################
        # FRAME-BY-FRAME ITERATION AND INITIALIZATION
        #############################################
        # Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_timestamp = measurement_data.game_timestamp / 1000.0
        start_x, start_y, start_z, start_pitch, start_roll, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        collided_flag_history = [False]  # assume player starts off non-collided

        #############################################
        # SETTINGS WAYPOINTS
        #############################################
        starting    = scene.player_start_spots[PLAYER_START_INDEX]
        destination = scene.player_start_spots[DESTINATION_INDEX]

        # Starting position is the current position (x, y, z, pitch, roll, yaw)
        source_pos = [starting.location.x, starting.location.y, starting.location.z]
        source_ori = [starting.orientation.x, starting.orientation.y]
        source = mission_planner.project_node(source_pos)

        # Destination position
        destination_pos = [destination.location.x, destination.location.y, destination.location.z]
        destination_ori = [destination.orientation.x, destination.orientation.y]
        destination = mission_planner.project_node(destination_pos)

        # Compute route from starting to goal position
        waypoints = []
        waypoints_route = mission_planner.compute_route(source, source_ori, destination, destination_ori)

        intersection_nodes = mission_planner.get_intersection_nodes()
        intersection_pair = []
        turn_cooldown = 0
        prev_x = False
        prev_y = False

        #TODO: togliere o meno
        #intersections_location = []
        # Put waypoints in the lane
        previuos_waypoint = mission_planner._map.convert_to_world(waypoints_route[0])
        for i in range(1,len(waypoints_route)):
            point = waypoints_route[i]

            waypoint = mission_planner._map.convert_to_world(point)

            current_waypoint = make_correction(waypoint, previuos_waypoint, DESIRED_SPEED)
            
            dx = current_waypoint[0] - previuos_waypoint[0]
            dy = current_waypoint[1] - previuos_waypoint[1]

            is_turn = ((prev_x and abs(dy) > 0.1) or (prev_y and abs(dx) > 0.1)) and not(abs(dx) > 0.1 and abs(dy) > 0.1)

            prev_x = abs(dx) > 0.1
            prev_y = abs(dy) > 0.1

            if point in intersection_nodes:                
                prev_start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-2])
                center_intersection = mission_planner._map.convert_to_world(waypoints_route[i])

                start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-1])
                end_intersection = mission_planner._map.convert_to_world(waypoints_route[i+1])

                start_intersection = make_correction(start_intersection, prev_start_intersection, TURN_SPEED)
                end_intersection = make_correction(end_intersection, center_intersection, TURN_SPEED)
                
                #TODO: TOGLIERE O MENO
                # Save all the intersection in order to be able to stop at the traffic lights
                #intersections_location.append(start_intersection)

                dx = start_intersection[0] - end_intersection[0]
                dy = start_intersection[1] - end_intersection[1]

                if abs(dx) > 0 and abs(dy) > 0:
                    intersection_pair.append((center_intersection,len(waypoints)))
                    waypoints[-1][2] = TURN_SPEED
                    
                    middle_point = [(start_intersection[0] + end_intersection[0]) /2,  (start_intersection[1] + end_intersection[1]) /2]

                    turn_angle = math.atan2((end_intersection[1] - start_intersection[1]),(start_intersection[0] - end_intersection[0]))
                    print(turn_angle,  pi / 4, middle_point[0] - center_intersection[0] < 0)

                    turn_adjust = 0 < turn_angle < pi / 2 and middle_point[0] - center_intersection[0] < 0
                    turn_adjust_2 =  pi / 2 < turn_angle < pi and middle_point[0] - center_intersection[0] < 0

                    quater_part = - pi / 2 < turn_angle < 0 
                    neg_turn_adjust = quater_part and middle_point[0] - center_intersection[0] < 0
                    neg_turn_adjust_2 = - pi < turn_angle < -pi/2 and middle_point[0] - center_intersection[0] < 0
                    
                    centering = 0.55 if turn_adjust or neg_turn_adjust else 0.75 

                    middle_intersection = [(centering*middle_point[0] + (1-centering)*center_intersection[0]),  (centering*middle_point[1] + (1-centering)*center_intersection[1])]

                    # Point at intersection:
                    A = [[start_intersection[0], start_intersection[1], 1],
                         [end_intersection[0], end_intersection[1], 1],
                         [middle_intersection[0], middle_intersection[1], 1]]
                        
                    b = [-start_intersection[0]**2 - start_intersection[1]**2, 
                         -end_intersection[0]**2 - end_intersection[1]**2,
                         -middle_intersection[0]**2 - middle_intersection[1]**2]

                    coeffs = np.matmul(np.linalg.inv(A), b)

                    x = start_intersection[0]
                    
                    internal_turn = 0 if turn_adjust or turn_adjust_2 or quater_part  else 1
                    
                    center_x = -coeffs[0]/2 + internal_turn * 0.10
                    center_y = -coeffs[1]/2 + internal_turn * 0.10

                    r = sqrt(center_x**2 + center_y**2 - coeffs[2])

                    theta_start = math.atan2((start_intersection[1] - center_y),(start_intersection[0] - center_x))
                    theta_end = math.atan2((end_intersection[1] - center_y),(end_intersection[0] - center_x))

                    start_to_end = 1 if theta_start < theta_end else -1

                    theta_step = (abs(theta_end - theta_start) * start_to_end) /20

                    theta = theta_start + 6*theta_step

                    while (start_to_end==1 and theta < theta_end - 3*theta_step) or (start_to_end==-1 and theta > theta_end - 6*theta_step):
                        waypoint_on_lane = [0,0,0]

                        waypoint_on_lane[0] = center_x + r * cos(theta)
                        waypoint_on_lane[1] = center_y + r * sin(theta)
                        waypoint_on_lane[2] = TURN_SPEED

                        waypoints.append(waypoint_on_lane)
                        theta += theta_step 
                    
                    turn_cooldown = 4
            else:
                waypoint = mission_planner._map.convert_to_world(point)

                if turn_cooldown > 0:
                    target_speed = TURN_SPEED
                    turn_cooldown -= 1
                else:
                    target_speed = DESIRED_SPEED
                
                waypoint_on_lane = make_correction(waypoint,previuos_waypoint,target_speed)

                waypoints.append(waypoint_on_lane)

                previuos_waypoint = waypoint

        waypoints = np.array(waypoints)
        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # VEHICLE TRAJECTORY LIVE PLOTTING SETUP
        #############################################
        # Uses the live plotter to generate live feedback during the simulation
        # The two feedback includes the trajectory feedback and
        # the controller feedback (which includes the speed tracking).
        if ENABLE_LIVE_PLOT:
            lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
            lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

            # Add 2D position / trajectory plot
            trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                    title='Vehicle Trajectory',
                    figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                    edgecolor="black",
                    rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

            trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                            # coordinate system the X
                                            # axis in the graph is flipped
            trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

            # Add waypoint markers
            trajectory_fig.add_graph("waypoints", window_size=len(waypoints),
                                    x0=waypoints[:,0], y0=waypoints[:,1],
                                    linestyle="-", marker="", color='g')
            # Add trajectory markers
            trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                    x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                    y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                    color=[1, 0.5, 0])
            # Add starting position marker
            trajectory_fig.add_graph("start_pos", window_size=1, 
                                    x0=[start_x], y0=[start_y],
                                    marker=11, color=[1, 0.5, 0], 
                                    markertext="Start", marker_text_offset=1)

            trajectory_fig.add_graph("obstacles_points",
                                    window_size=8 * (NUM_PEDESTRIANS + NUM_VEHICLES) ,
                                    x0=[0]* (8 * (NUM_PEDESTRIANS + NUM_VEHICLES)), 
                                    y0=[0]* (8 * (NUM_PEDESTRIANS + NUM_VEHICLES)),
                                        linestyle="", marker="+", color='b')

            # Add end position marker
            trajectory_fig.add_graph("end_pos", window_size=1, 
                                    x0=[waypoints[-1, 0]], 
                                    y0=[waypoints[-1, 1]],
                                    marker="D", color='r', 
                                    markertext="End", marker_text_offset=1)
            # Add car marker
            trajectory_fig.add_graph("car", window_size=1, 
                                    marker="s", color='b', markertext="Car",
                                    marker_text_offset=1)
            # Add lead car information
            trajectory_fig.add_graph("leadcar", window_size=1, 
                                    marker="s", color='g', markertext="Lead Car",
                                    marker_text_offset=1)

            # Add lookahead path
            trajectory_fig.add_graph("selected_path", 
                                    window_size=INTERP_MAX_POINTS_PLOT,
                                    x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                    y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                    color=[1, 0.5, 0.0],
                                    linewidth=3)

            # Add local path proposals
            for i in range(NUM_PATHS):
                trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                        x0=None, y0=None, color=[0.0, 0.0, 1.0])

            ###
            # Add 1D speed profile updater
            ###
            forward_speed_fig =\
                    lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
            forward_speed_fig.add_graph("forward_speed", 
                                        label="forward_speed", 
                                        window_size=TOTAL_EPISODE_FRAMES)
            forward_speed_fig.add_graph("reference_signal", 
                                        label="reference_Signal", 
                                        window_size=TOTAL_EPISODE_FRAMES)

            # Add throttle signals graph
            throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
            throttle_fig.add_graph("throttle", 
                                label="throttle", 
                                window_size=TOTAL_EPISODE_FRAMES)
            # Add brake signals graph
            brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
            brake_fig.add_graph("brake", 
                                label="brake", 
                                window_size=TOTAL_EPISODE_FRAMES)
            # Add steering signals graph
            steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
            steer_fig.add_graph("steer", 
                                label="steer", 
                                window_size=TOTAL_EPISODE_FRAMES)

            # live plotter is disabled, hide windows
            if not ENABLE_LIVE_PLOT:
                lp_traj._root.withdraw()
                lp_1d._root.withdraw()        

        #############################################
        # PLANNING MODULES DECLARATION
        #############################################
        wp_goal_index   = 0
        local_waypoints = None
        path_validity   = np.zeros((NUM_PATHS, 1), dtype=bool)
        lp = local_planner.LocalPlanner(NUM_PATHS,
                                        PATH_OFFSET,
                                        CIRCLE_OFFSETS,
                                        CIRCLE_RADII,
                                        PATH_SELECT_WEIGHT,
                                        TIME_GAP,
                                        A_MAX,
                                        SLOW_SPEED,
                                        STOP_LINE_BUFFER)
        
        LEAD_Y_DIST = 3 # distance on y-axis (m) used in order to evaluate if a possible lead car is in our lane
        LEAD_MAX_YAW_DIFF = 30 # angle difference (deg) in order to evaluate if a possible lead car has a similar orientation to ours
        #TODO: terzo e quarto argomento del costruttore servono?
        bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                    LEAD_VEHICLE_LOOKAHEAD, 
                                                    [])

        #############################################
        # SCENARIO EXECUTION LOOP
        #############################################

        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True

        # Initialize the current timestamp.
        current_timestamp = start_timestamp

        # Initialize collision history
        prev_collision_vehicles    = 0
        prev_collision_pedestrians = 0
        prev_collision_other       = 0
 
        prev_tl_state = None # previous state of the traffic light

        ## ITERATE OVER FRAMES ##
        for frame in range(TOTAL_EPISODE_FRAMES):
            #print("\n-----FRAME ", frame, " -----")
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()

            obstacles = [] # lists containing the bounding boxes of the obstacles (pedestrian/vehicles) found on the lane
            tl_locations = [] # lists containing the locations of the traffic lights in the scene

            # Update pose and timestamp
            prev_timestamp = current_timestamp
            current_x, current_y, current_z, current_pitch, current_roll, current_yaw = get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0
            
            #TODO: SERVE?
            #bp.set_emergency_distance(calc_distance(current_speed, 0, -A_MAX))

            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else: current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
            
            # Store history
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 

            # Store collision history
            collided_flag,\
            prev_collision_vehicles,\
            prev_collision_pedestrians,\
            prev_collision_other = get_player_collided_flag(measurement_data,
                                                 prev_collision_vehicles,
                                                 prev_collision_pedestrians,
                                                 prev_collision_other)
            collided_flag_history.append(collided_flag)

            ## RETRIEVE THE SENSORS DATA IN ORDER TO USE THEM FOR THE TRAFFIC LIGHT DETECTION ##
            front_camera_data = sensor_data.get('CameraRGB_Front', None)
            tl_camera_data = sensor_data.get('CameraRGB_TL', None)
            depth_camera_data = sensor_data.get('CameraDEPTH_TL', None)
            depth_camera_data_front = sensor_data.get('CameraDEPTH_FRONT', None)

            if front_camera_data is not None:
                front_camera_data = to_bgra_array(front_camera_data)

            if tl_camera_data is not None:
                tl_camera_data = to_bgra_array(tl_camera_data)

            if depth_camera_data is not None:
                depth_camera_data = image_converter.depth_to_array(depth_camera_data)

            if depth_camera_data_front is not None:
                depth_camera_data_front = image_converter.depth_to_array(depth_camera_data_front)
            
            # Execute the behaviour and local planning in the current instance
            # Note that updating the local path during every controller update
            # produces issues with the tracking performance (imagine everytime
            # the controller tried to follow the path, a new path appears). For
            # this reason, the local planner (LP) will update every X frame,
            # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
            # to be operating at a frequency that is a division to the 
            # simulation frequency.

            ## ITERATES ALL THE AGENTS SPAWNED IN CARLA IN ORDER TO PROPERLY MANAGE THEM ##

            lead_car_pos    = None # location of the lead vehicle found
            lead_car_speed  = None # speed of the lead vehicle found
            
            LEAD_X_DIST = float('inf') # distance on x-axis (m) used in order to evaluate if a possible lead vehicle is in front of us

            bp._obstacle_on_lane = False # flag used in the behavioural planner for identifying an agent that intercepts our trajectory

            for agent in measurement_data.non_player_agents:
                # if the agent is a vehicle, check if it's a lead vehicle or an obstacle, otherwise don't care of it
                if agent.HasField("vehicle"):
                    vehicle = agent.vehicle
                    transform = vehicle.transform
                    location = transform.location
                    rotation = transform.rotation
                    
                    #TODO: controllare se con la norma nel global frame cambia qualcosa o meno
                    # Retrieve the position of the vehicle in the vehicle frame
                    car_loc_relative = transform_world_to_ego_frame([location.x, location.y, location.z],
                                                                    [current_x, current_y, current_z],
                                                                    [current_roll, current_pitch, current_yaw])
                    # Decide if car is lead car:
                    # - 0 < car_loc_relative[0] < LEAD_X_DIST => if we are approaching the lead vehicle
                    # - abs(car_loc_relative[1]) < LEAD_Y_DIST => if the lead vehicle is our lane
                    # - abs(compute_angle_diff(agent.vehicle.transform.rotation.yaw, np.rad2deg(current_yaw))) < LEAD_MIN_YAW_DIFF => 
                    #               if the difference between ego and lead vehicle orientation is within 30 degrees
                    if 0 < car_loc_relative[0] < LEAD_X_DIST and abs(car_loc_relative[1]) < LEAD_Y_DIST and \
                        abs(compute_angle_diff(agent.vehicle.transform.rotation.yaw, np.rad2deg(current_yaw))) < LEAD_MAX_YAW_DIFF:
                        LEAD_X_DIST = car_loc_relative[0]
                        lead_car_pos = [agent.vehicle.transform.location.x, agent.vehicle.transform.location.y]
                        lead_car_speed = agent.vehicle.forward_speed
                    # else check if car is close (distance from us < 7 m) ...
                    else:
                        if np.linalg.norm(car_loc_relative) < LOOKAHEAD_VEHICLE:
                            # ... for stopping if it intersects our trajectory ...
                            if check_obstacle_future_intersection(vehicle,
                                    LOOKAHEAD_VEHICLE,
                                    car_loc_relative,
                                    [current_x, current_y, current_z],
                                    [current_roll, current_pitch, current_yaw]):
                                #print("VEHICLE -> OBSTACLE ON LANE")
                                bp._obstacle_on_lane = True
                            else:
                                # ... or it has to be managed as an obstacle
                                obstacles.append(obstacle_to_world(location, vehicle.bounding_box.extent, rotation))

                # if the agent is a pedestrian, check if it intersects our trajoectory or it's an obsatcle to be avoided
                elif agent.HasField("pedestrian"):
                    pedestrian = agent.pedestrian
                    transform = pedestrian.transform
                    location = transform.location
                    rotation = transform.rotation
                    loc_relative = transform_world_to_ego_frame([location.x, location.y, location.z],
                                                                [current_x, current_y, current_z],
                                                                [current_roll, current_pitch, current_yaw])
                    # if distance from us is less than 7 m ...
                    if np.linalg.norm(loc_relative) < LOOKAHEAD_PEDESTRIAN:
                        # ... check if it intersects our trajectory for stopping
                        if check_obstacle_future_intersection(pedestrian,
                                    LOOKAHEAD_PEDESTRIAN,
                                    loc_relative,
                                    [current_x, current_y, current_z],
                                    [current_roll, current_pitch, current_yaw], 
                                    speed=10):
                            #print("PEDESTRIAN -> OBSTACLE ON LANE")
                            bp._obstacle_on_lane = True
                            # TODO: rimuovere?
                            #bp.set_pedestrian_loc(loc_relative, norm)
                        else:
                            # ...or add it as an obstacle
                            obstacles.append(obstacle_to_world(location, pedestrian.bounding_box.extent, rotation))
                
                # if the agent is a traffic light, get its location for deciding where to stop if nearest TL is red
                elif agent.HasField('traffic_light'):
                    location = agent.traffic_light.transform.location
                    tl_locations.append([location.x,location.y,location.z])
        
            ## USE PLANNING MODULES EACH LP_FREQUENCY_DIVISOR FRAMES ##
            if frame % LP_FREQUENCY_DIVISOR == 0:
                # predict traffic light state on each camera and draw boxes with results in order to display
                boxes2 = detector.predict_image(tl_camera_data)
                tl_camera_data = detector.draw_boxes(tl_camera_data, boxes2)
                boxes1 = detector.predict_image(front_camera_data)
                front_camera_data = detector.draw_boxes(front_camera_data, boxes1)
                cv2.imshow("Cameras", np.hstack((front_camera_data, tl_camera_data)))
                #cv2.imshow("Depth Camera", depth_camera_data)
                cv2.waitKey(1)

                bp.set_depth_img({"CameraDEPTH_TL":depth_camera_data, "CameraDEPTH_FRONT":depth_camera_data_front})
                bp.set_current_box(detector._current_box)
                bp.set_camera_params(depth_info)

                curr_light_state = detector.update_state(boxes1, boxes2)
                if prev_tl_state != curr_light_state:
                    prev_tl_state = curr_light_state
                    print(f"Traffic Light: {curr_light_state.name}")

                # if curr_center_boxes is not None:
                #     tl_distance = depth_camera_data[curr_center_boxes[1]][curr_center_boxes[0]]*1000 #normalized distance from TL [0,1]
                #     print("TL POS: ", curr_center_boxes, " - TL Distance: ", tl_distance)
                #     # set traffic lights locations in the behavioural planner in order to be used in state computation
                #     bp.set_tl_locations(tl_distance)
                
                # set state of the nearest traffic light in the behavioural planner in order to be used in state computation
                bp.set_lightstate(curr_light_state)

                # Compute open loop speed estimate
                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                # Calculate the goal state set in the local frame for the local planner.
                # Current speed should be open loop for the velocity profile generation.
                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                # Set lookahead distance based on current speed.
                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                # Perform a state transition in the behavioural planner.
                bp.transition_state(waypoints, ego_state, current_speed)
                
                # check for a lead vehicle
                if lead_car_pos is not None:
                    bp.check_for_lead_vehicle(ego_state, lead_car_pos)
                    # if bp._follow_lead_vehicle:
                    #     print(f"Lead vehicle to follow => pos: {lead_car_pos}, vel: {lead_car_speed}")

                # Compute the goal state set from the behavioural planner's computed goal state.
                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                # Calculate planned paths in the local frame.
                paths, path_validity = lp.plan_paths(goal_state_set)

                # Transform those paths back to the global frame.
                paths = local_planner.transform_paths(paths, ego_state)

                # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, obstacles)

                # Compute the best local path.
                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                # If no path was feasible, continue to follow the previous best path.
                if best_index == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                if best_path is not None:
                    # Compute the velocity profile for the path, and compute the waypoints.
                    desired_speed = bp._goal_state[2]

                    # compute lead car state if present
                    if lead_car_pos is not None: lead_car_state = [lead_car_pos[0], lead_car_pos[1], lead_car_speed]
                    else: lead_car_state = None

                    decelerate_to_stop = bp._state == behavioural_planner.FSMState.DECELERATE_AND_STOP
                    local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle, bp._obstacle_on_lane)

                    if local_waypoints != None:
                        # Update the controller waypoint path with the best local path.
                        # This controller is similar to that developed in Course 1 of this
                        # specialization.  Linear interpolation computation on the waypoints
                        # is also used to ensure a fine resolution between points.
                        wp_distance = []   # distance array
                        local_waypoints_np = np.array(local_waypoints)
                        for i in range(1, local_waypoints_np.shape[0]):
                            wp_distance.append(
                                    np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                            (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                        wp_distance.append(0)  # last distance is 0 because it is the distance
                                            # from the last waypoint to the last waypoint

                        # Linearly interpolate between waypoints and store in a list
                        wp_interp      = []    # interpolated values 
                                            # (rows = waypoints, columns = [x, y, v])
                        for i in range(local_waypoints_np.shape[0] - 1):
                            # Add original waypoint to interpolated waypoints list (and append
                            # it to the hash table)
                            wp_interp.append(list(local_waypoints_np[i]))
                    
                            # Interpolate to the next waypoint. First compute the number of
                            # points to interpolate based on the desired resolution and
                            # incrementally add interpolated points until the next waypoint
                            # is about to be reached.
                            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                        float(INTERP_DISTANCE_RES)) - 1)
                            wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                            wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                            for j in range(num_pts_to_interp):
                                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                                wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                        # add last waypoint at the end
                        wp_interp.append(list(local_waypoints_np[-1]))
                        
                        # Update the other controller values and controls
                        controller.update_waypoints(wp_interp)

            ## Controller Update ##
            if best_path is not None and local_waypoints is not None and local_waypoints != []:
                controller.update_values(current_x, current_y, current_yaw, 
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            # Skip the first frame or if there exists no local paths
            if skip_first_frame and frame == 0: pass
            elif local_waypoints == None: pass
            else:
                if ENABLE_LIVE_PLOT:
                    # Update live plotter with new feedback
                    trajectory_fig.roll("trajectory", current_x, current_y)
                    trajectory_fig.roll("car", current_x, current_y)
                    
                    # Load parked car points
                    if len(obstacles) > 0:
                        x = obstacles[:,:,0]
                        y = obstacles[:,:,1]
                        x = np.reshape(x, x.shape[0] * x.shape[1])
                        y = np.reshape(y, y.shape[0] * y.shape[1])

                        trajectory_fig.roll("obstacles_points", x, y)

                    
                    forward_speed_fig.roll("forward_speed", 
                                        current_timestamp, 
                                        current_speed)
                    forward_speed_fig.roll("reference_signal", 
                                        current_timestamp, 
                                        controller.desired_speed)
                    throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                    brake_fig.roll("brake", current_timestamp, cmd_brake)
                    steer_fig.roll("steer", current_timestamp, cmd_steer)

                    # Local path plotter update
                    if frame % LP_FREQUENCY_DIVISOR == 0:
                        path_counter = 0
                        for i in range(NUM_PATHS):
                            # If a path was invalid in the set, there is no path to plot.
                            if path_validity[i]:
                                # Colour paths according to collision checking.
                                if not collision_check_array[path_counter]:
                                    colour = 'r'
                                elif i == best_index:
                                    colour = 'k'
                                else:
                                    colour = 'b'
                                trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                                path_counter += 1
                            else:
                                trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
                # When plotting lookahead path, only plot a number of points
                # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                # to decrease load when live plotting
                if ENABLE_LIVE_PLOT:
                    wp_interp_np = np.array(wp_interp)
                    path_indices = np.floor(np.linspace(0, 
                                                        wp_interp_np.shape[0]-1,
                                                        INTERP_MAX_POINTS_PLOT))
                    trajectory_fig.update("selected_path", 
                            wp_interp_np[path_indices.astype(int), 0],
                            wp_interp_np[path_indices.astype(int), 1],
                            new_colour=[1, 0.5, 0.0])

                # Refresh the live plot based on the refresh rate set by the options
                if ENABLE_LIVE_PLOT and live_plot_timer.has_exceeded_lap_period():
                    lp_traj.refresh()
                    lp_1d.refresh()
                    live_plot_timer.lap()

            ## OUTPUT CONTROLLER COMMANDS TO CARLA SERVER ##
            # if there's an obstacle (pedestrian/vehicle) on the lane, perform an emergency brake (brake=1.0)
            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake if not bp._obstacle_on_lane else 1.0)

            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT: reached_the_end = True
            if reached_the_end: break
            #print ("-------------------")

        # End of demo - Stop vehicle and Store outputs to the controller output
        # directory.
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        # Stop the car
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # Store the various outputs
        if ENABLE_LIVE_PLOT:
            store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
            store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
            store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
            store_trajectory_plot(brake_fig.fig, 'brake_output.png')
            store_trajectory_plot(steer_fig.fig, 'steer_output.png')
            write_trajectory_file(x_history, y_history, speed_history, time_history,
                                collided_flag_history)
            write_collisioncount_file(collided_flag_history)
    
def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
# import wmi
if __name__ == '__main__':
    try:
        main()
        # for process in wmi.WMI().Win32_Process():
        #     if process.name == "CarlaUE4.exe":
        #         process.Terminate()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

