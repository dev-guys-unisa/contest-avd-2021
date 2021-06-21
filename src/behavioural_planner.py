#!/usr/bin/env python3
import numpy as np
import math
from traffic_light_detector import TrafficLightState
from helpers import transform_world_to_ego_frame, calc_distance

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STOP_FOR_OBSTACLES = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10

# Distance from intersection where we spot the stop line
DIST_INTER = 20 # meters
# Minimum distance from the intersection to which we want to stop for a RED light
DIST_STOP_INTER = 4.0 # meters

# Radius on the Y axis for the Intersection ahead check.
RELATIVE_DIST_INTER_Y = 3.5

# Distace from the nearest pedestrian in which we have to stop
DIST_TO_PEDESTRIAN = 1 # m


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, intersection, a_max):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._lightstate                    = TrafficLightState.NO_TL
        self._intersection                  = intersection
        self._emergency_distance            = 0
        self._a_max                         = a_max
        self._pedestrian_locs               = [] # list of pedestrian (x,y,z)
        self._pedestrian_dists              = [] # list of norms aka distance from ego
        self._intersection_index             = None
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_lightstate(self, lighstate):
        self._lightstate = lighstate
    
    def set_emergency_distance(self, emergency_distance):
        self._emergency_distance = emergency_distance

    def set_pedestrian_loc(self, loc, n):
        self._pedestrian_locs.append(loc)
        self._pedestrian_dists.append(n)
    
    def clear_ped_lists(self):
        self._pedestrian_dists.clear()
        self._pedestrian_locs.clear()

    def intersection_goal(self, waypoints, ego_state,closest_index, goal_index):
        for i in range(closest_index, goal_index):
            for inter in self._intersection:
                # We project the intersection onto the ego frame
                inter_loc_relative = transform_world_to_ego_frame(
                    [inter[0], inter[1], inter[2]],
                    [ego_state[0], ego_state[1], 0.0],
                    [0.0, 0.0, ego_state[2]]
                )
                # We calculate the distance between the current waypoint and the current intersection
                dist_spot = np.linalg.norm(np.array([waypoints[i][0] - inter[0], waypoints[i][1] - inter[1]]))
                # If this distance is smaller than DIST_SPOT_INTER, we spot the intersection
                if dist_spot < DIST_INTER:
                    # But we also check if it is ahead of ego or behind. If ahead, we choose a stop waypoint.
                    if inter_loc_relative[0] > 0 and -RELATIVE_DIST_INTER_Y <= inter_loc_relative[1] <= RELATIVE_DIST_INTER_Y:
                        print(f"Intersection ahead. Position: {inter_loc_relative}")
                        for j in range(i, len(waypoints)):
                            dist_stop = np.linalg.norm(
                                np.array([waypoints[j][0] - inter[0], waypoints[j][1] - inter[1]]))
                            print("Dist stop: ",dist_stop)
                            if dist_stop < DIST_STOP_INTER:
                                print(f"Stop Waypoint: {j - 1} {waypoints[j - 1]}")
                                return j-1
                    # Otherwise we stop checking.
                    else:
                        print(f"Intersection behind, ignored. Position: {inter_loc_relative}")
                        return None
        return None

    def get_pedestrian_goal(self, waypoints, closest_index, goal_index):
        nearest_ped = self._pedestrian_dists.index(min(self._pedestrian_dists))
        #print("Nearest index: ",nearest_ped)
        nearest_loc = self._pedestrian_locs[nearest_ped]
        
        for i in range(closest_index, len(waypoints)):
            dist = np.linalg.norm(np.array([waypoints[i][0] - nearest_loc[0], waypoints[i][1] - nearest_loc[1]])) # dist waypoint-ped
            #print("DIST ped: ",dist)
            if dist <= DIST_TO_PEDESTRIAN:
                #print("Waypoint to stop: ",waypoints[i])
                return i
        return None


    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
        if self._state == FOLLOW_LANE:
            print("State: FOLLOW_LANE")
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            #print("Semaforo Behav: ", self._lightstate)
            if self._obstacle_on_lane:
                #self._emergency_distance = calc_distance(closed_loop_speed, 0, -self._a_max)
                #self._goal_state[2] = 0
                pedestrian_index = self.get_pedestrian_goal(waypoints,closest_index,goal_index)
                print("Pedestrian index: ",pedestrian_index)
                if pedestrian_index is not None:
                    self._goal_index = pedestrian_index
                    self._goal_state = waypoints[self._goal_index]
                    self._goal_state[2] = 0
                print("FOLLOW_LANE -> STOP_FOR_OBSTACLES: stopping at ",self._goal_state)
                self._state = STOP_FOR_OBSTACLES

            elif self._lightstate == TrafficLightState.STOP:
                print("FOLLOW_LANE -> DECEL AND STOP")
                self._state = DECELERATE_TO_STOP
                # intersection_index = self.intersection_goal(waypoints, ego_state, closest_index, goal_index)

                # if intersection_index is not None:
                #     self._goal_index = intersection_index
                #     self._goal_state = waypoints[intersection_index]
                #     self._goal_state[2] = 0
            
            else: print("FOLLOW LANE: go to ",self._goal_state)
            
        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            print("State: DECELERATE_AND_STOP")

            if self._obstacle_on_lane:
                #self._emergency_distance = calc_distance(closed_loop_speed, 0, -self._a_max)
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1

                pedestrian_index = self.get_pedestrian_goal(waypoints,closest_index,goal_index)
                if pedestrian_index is not None:
                    self._goal_index = pedestrian_index
                    self._goal_state = waypoints[self._goal_index]
                    self._goal_state[2] = 0

                self._state = STOP_FOR_OBSTACLES
                print("DECELERATE_AND_STOP -> STOP_FOR_OBSTACLES: stopping at ",self._goal_state)

            elif self._lightstate == TrafficLightState.GO or self._lightstate == TrafficLightState.NO_TL:
                #l'ulteriore controllo NO_TL è per non rimanere bloccato al semaforo quando non vede nulla

                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1 
                                
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

                self._state = FOLLOW_LANE
                self._intersection_index = None
                # intersection_index = None
                print("DECELERATE_AND_STOP -> FOLLOW_LANE: go to ",self._goal_state)

            else:
                self._state = DECELERATE_TO_STOP
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1

                if self._intersection_index is None:
                    intersection_index = self.intersection_goal(waypoints, ego_state,closest_index, goal_index)
                    if intersection_index is not None:
                        self._intersection_index = intersection_index
                        self._goal_index = intersection_index
                        self._goal_state = waypoints[intersection_index]
                        self._goal_state[2] = 0
                print("Ho visto il semaforo rosso e sto rallentando verso ",self._goal_state)

        elif self._state == STOP_FOR_OBSTACLES:
            print("State: STOP_FOR_OBSTACLES -> goal: ",self._goal_state)
            if not self._obstacle_on_lane:
                self._state = FOLLOW_LANE
                print("STOP_FOR_OBSTACLES -> FOLLOW_LANE")

        else:
            raise ValueError('Invalid state value.')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)
                
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance > self._follow_lead_vehicle_lookahead + 15:
                print("IF PETRONE")
                self._follow_lead_vehicle = False
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
