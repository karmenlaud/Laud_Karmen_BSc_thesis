*This is the code base for my thesis.* 

**How to run the code:**

Most of the files needed are already in the repository. All of the packages in the requirements file should be installed and this should be done in a new virtual environment to avoid any dependency conflicts. The code was tested using Python 3.10.12 running in WSL on Ubuntu 22.04. To run the code:
* follow the instructions found here https://github.com/gifflet/opencv-text-detection for using the cv2 EAST model. The file frozen_east_text_detection.pb should be downloaded and located in the same directory with the rest of the codebase;
* download a ROS bag file that contains the topics /camera_fl/image_raw, /pacmod/vehicle_speed_rpt and /gps/gps;
* run the file main.py and select the bag file when prompted.
