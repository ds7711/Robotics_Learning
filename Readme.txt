1. Dependencies:
  a. Python 2.7.11
  b. numpy 1.12.2
  c. VPython 7.1.3
  d. Jupyter/VPython Notebook
  Note: the same code may also work in Python 3 with Numpy and VPython installed.

2. Roles of the scripts
  a. robotics_library.py: 
     Defines the Robotic Manipulator and specifies the rotation axises as [z, x, x, z, y, y].
     The initial configuration/pose can be changed by specifying the initial angles and link lengths. 
     Rotation angle is decided as the angle needed to align frame i-1 to frame i (positive direction is determined by the right-hand rule). 
  b. hw1_demo.ipybn:
     Create a robotic manipulator and visualize how the robot reacts to the rotation angles.
     This code has to be run through Jupyter Notebook because VPython can only work in the Jupyter Notebook. 


3. TASK
  a. Rotation axies are pre-specified as [z, x, x, z, y, y]. Link lengths and initial rotation angles can be spcified while creating the robot. Each joint can rotate from -pi to +pi degrees. This allows the robotic manipulator to have good reachability the freedom to reach the same point with different orientations.
  b. Forward kinematic functions are implemented as methods of the Robotic Manipulator object. It takes the rotation (joint) angles and return the the coordinate of any point in any reference frame. The coordinate of the end-effector is a special case where the reference frame is the last one. 
  c. joint_abs_locations give the coordinate of every joint and the end effector in the world frame. Visualization is implemented using VPython. 
  d. Two animations are created. In the first one, each joint moves one by one to show the rotation axis. In the second one, each joint moves randomly. 

