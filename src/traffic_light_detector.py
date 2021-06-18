import sys
import os
import json
import cv2
import numpy as np
from typing import List, Dict
from enum import Enum

# Script level imports
from traffic_light_detection_module.yolo import YOLO
#from traffic_light_detection_module.preprocessing import preprocess_image
from traffic_light_detection_module.postprocessing import draw_boxes, BoundBox
#from data_visualization import Sensor


class TrafficLightState(Enum):
    GO = 0
    STOP = 1
    NO_TL = -1


class TrafficLightDetector(YOLO):
    # Minimum threshold to refuse false positives
    MIN_TH = 0.1
    MAX_TH1 = 0.25
    MAX_TH2 = 0.40

    # Minimum number of frames before change state
    MIN_STOP_FRAMES = 5     # Min frames to choice STOP signal 
    MIN_GO_FRAMES = 10      # Min frames to choice GO signal
    MIN_NOTL_FRAMES = 10    # Min frames to choice NO_TL signal
    MIN_INDECISION_FRAMES = 30  # Min frames to understand the state predicted is not reliable 

    __slots__ = 'config', 'labels', '_state_counter', '_indecision_counter', '_state', '_new_state'

    def __init__(self):
        #with open(os.path.join('traffic_light_detection_module', 'config.json')) as config_file:
        #    self.config = json.load(config_file)

        with open('traffic_light_detection_module/config.json', 'r') as config_file:
            self.config = json.load(config_file)
            self.config['model']['saved_model_name']=os.path.join(os.path.realpath(os.path.dirname(__file__)),"traffic_light_detection_module","checkpoints","traffic-light-detection.h5")

        self.labels = self.config['model']['classes']
        super().__init__(self.config)
        self.model.load_weights(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                             'traffic_light_detection_module',
                                             'checkpoints', self.config['model']['saved_model_name']))
        self._state_counter = 0
        self._indecision_counter = 0
        self._state = None      # Stato stabile (effettivo)
        self._new_state = None # Stato corrente (in aggiornamento)

    def get_state(self):
        return self._state

    def preprocess_image(self, image, image_h, image_w):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_h, image_w))
        image = image/255
        image = np.expand_dims(image, 0)

        return image


    def predict_image(self, image):
        """
        Predicts the bounding boxes of traffic lights from a cv2 image
        Returns a list of BoundBox objects.
        """
        image = self.preprocess_image(image, self.image_h, self.image_w)
        boxes = super().predict_image(image)
        return boxes

    def draw_boxes(self, image, boxes):
        """
        Draws the detected traffic lights' boxes on a cv2 image.
        Returns the new image.
        """
        return draw_boxes(image, boxes, self.labels)
        
    
    def light_state(self, boxes: List[BoundBox]):
        """
        filters all the detected boxes greater than MIN TH and 
        it returns the predicted label with the relative score

        Args:
            boxes (List[BoundBox]): list of Boundi box predicted
        """
        boxes_ = list(filter(lambda b: b.get_score() > TrafficLightDetector.MIN_TH, boxes))

        if len(boxes_) == 0:
            return TrafficLightState.NO_TL, 0

        # Heuristic
        box = sorted(boxes_, key=lambda b: b.get_area(), reverse=True)[0]

        # if box.get_label() == TrafficLightState.STOP.value:
        #    return TrafficLightState.STOP, box.get_score()

        if box.get_label() == TrafficLightState.GO.value:
            return TrafficLightState.GO, box.get_score()

        return TrafficLightState.STOP, box.get_score()

    def update_state(self, boxes1, boxes2):

        light_state1, light_accuracy1 = self.light_state(boxes1)
        light_state2, light_accuracy2 = self.light_state(boxes2)

        if round(light_accuracy2, 2) > self.MAX_TH2:
            current_state = light_state2
        
        else:
            if round(light_accuracy1, 2) > self.MAX_TH1:
                if light_state1 == light_state2:
                    current_state = light_state2
                else: current_state = light_state1
            else:
                current_state = TrafficLightState.NO_TL

        #print("stato/ accuracy semaforo1: ", light_state1, light_accuracy1)
        #print("stato/ accuracy semaforo2: ", light_state2, light_accuracy2)
        #print("STATO pre: ", self._state)
        #print("STATO post: ", self._new_state)

        if self._state is None:
            self._state = TrafficLightState.NO_TL
            self._new_state = TrafficLightState.NO_TL

        if self._new_state != current_state:
            self._indecision_counter += 1
            self._state_counter = 0
            self._new_state = current_state
        else:
            self._state_counter += 1
            self._new_state = current_state

        if self._indecision_counter >= self.MIN_INDECISION_FRAMES:
            print(f"Cannot decide TL state. Defaulting to NO_TL...")
            self._state = TrafficLightState.NO_TL
            self._indecision_counter = 0
        elif ((self._new_state == TrafficLightState.NO_TL and self._state_counter >= self.MIN_NOTL_FRAMES) or
              (self._new_state == TrafficLightState.GO and self._state_counter >= self.MIN_GO_FRAMES) or
              (self._new_state == TrafficLightState.STOP and self._state_counter >= self.MIN_STOP_FRAMES)):
            #print("Decisione presa !")
            self._state = self._new_state
            self._state_counter = 0
            self._indecision_counter = 0

        return self.get_state()
