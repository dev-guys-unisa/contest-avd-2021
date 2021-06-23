# Autonomous Vehicle Driving Project

![pow](https://img.shields.io/badge/Powered%20By-dev--guys--unisa-blue)
![carla](https://img.shields.io/badge/Tested%20With-Carla-green)

This is the project repository for the Autonomous Vehicle Driving course @ Unisa.
___
# Group Members

![Alt text](https://github.com/dev-guys-unisa/ContestCognitiveRobotics2020/blob/main/utils/Logo.png?raw=true "GroupMembers")

* Salvatore Ventre
* Vincenzo Russomanno
* Giovanni Puzo
* Vittorio Fina
___
# Problem Description

Integrate a traffic light detector (already supplied, but not integrated) into a baseline and design a **hierarchical planner** that allows the vehicle to navigate autonomously within the Town01 map provided by the Carla simulator.

**Behavioral planners** and **Local planners** will need to be designed and implemented in a way that manage the behavior of the vehicle in the event of a *red*, *yellow* or *green* traffic light, as well as to avoid collisions with other vehicles and pedestrians present within the scene.

### Software Requirements

This software is made relying on the Python language and tested with the [Carla Simulator](https://carla.org/) in different scenarios. It is necessary to respect some dependencies, which for simplicity we have reported in a text file to be installed with the command:

```bash
/usr/bin/python3.6 -m pip install -r requirements.txt
```

These are the requirements:
|Index|Tool|Version|
|-----|----|-------|
|1|Python|== 3.6.2|
|2|Pillow|>= 8.1.2|
|3|Numpy|>= 1.14.5|
|4|Protobuf|>= 3.6.0|
|5|PyGame|>= 1.9.4|
|6|MatPlotLib|== 3.0.0|
|7|Future|>= 0.16.0|
|8|SciPy|>= 0.17.0|
___
# Clone the Repository
First of all it is necessary to clone the repository in the PythonClient directory in Carla Simulator, by running these commands:
```bash
cd ../../CarlaSimulator/PythonClient/
git clone https://github.com/dev-guys-unisa/contest-avd-2021
```
___
# How to launch a Demo
Here are some fundamental indications to launch a demo of the project:
#### Run the Server
* Linux:
```bash
../../CarlaUE4.sh /Game/Maps/Town01 -carla-server -windowed -benchmark -fps=30 -quality-level=Epic
```
* Windows:
```bash
..\..\CarlaUE4.exe /Game/Maps/Town01 -carla-server -windowed -benchmark -fps=30 -quality-level=Epic
```
#### Run the Client
You have to move in ```src/``` folder of the previous cloned repository and to launch ``` main.py``` file. Here there are the commands for Linux and Windows:
```bash
cd ../../CarlaSimulator/PythonClient/contest-avd-2021/src/
python3.6 main.py
```

**IMPORTANT**: If you want to launch only the client side, you have to specify the *ip address* and *number of port* of the Server in which Carla is running.
```bash
cd ../../CarlaSimulator/PythonClient/contest-avd-2021/src/
python3.6 main.py --host *ipaddress* --port *port*
```
___
##### Group 18
