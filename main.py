from rosbags.highlevel import AnyReader
import cv2
import numpy as np
from pathlib import Path
import time
import osmnx as ox
from text_reader import TextReader
from particle_filter import ParticleFilter
from OSMgraphml import OSMMap
import tkinter as tk
from tkinter import filedialog

# Base directory
REPO_ROOT = Path(__file__).resolve().parent

# Topics we are listening
topic_name_cam = '/camera_fl/image_raw'
topic_name_speed = '/pacmod/vehicle_speed_rpt'
topic_name_gps = '/gps/gps'

# Density of frames processed
frame_interval = 10

# If debug windows should open during the OCR
debug = False

# If detection should be done through tiling
tiled_ocr = True

# If the intermitten states should be rendered
render_between = True

particle_counts = [5000]

# Initialise the tools
G = ox.load_graphml("tartu.graphml") # could be switched for a different town.
reader = TextReader()

OSM = OSMMap(G)

def get_file_path():
    """ Opens a filer explorer and lets the user pick a file.
        Returns:
            Path to the bag file
        """
    # Hide root window
    root = tk.Tk()
    root.withdraw()

    # Open file picker
    file_path = filedialog.askopenfilename(
        title="Select ROS bag file",
        filetypes=[("ROS bag files", "*.bag"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        exit()
    bag_path = Path(file_path)

    return bag_path

def haversine(p1, p2):
    # p1, p2 = (lon, lat) in degrees
    lon1, lat1 = p1
    lon2, lat2 = p2

    R = 6371000.0  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def main():
    bag_path = get_file_path()

    for particle_count in particle_counts:
        pf = ParticleFilter(G, particle_count=particle_count)

        output_dir = REPO_ROOT / "results" / f"{particle_count}" / f"{bag_path.stem}_results"
        output_dir.mkdir(parents=True, exist_ok=True) 

        with open(str(output_dir /f"errors.csv"), 'w') as file:
            file.write("")

        # Making the initial render
        OSM.render(pf.particles, str(output_dir / "before_data.png"))
        
        # Open ROS bag using AnyReader
        with AnyReader([bag_path]) as bag:

            # Get only the connections that have topics i am looking for
            connections = [c for c in bag.connections if c.topic in[topic_name_cam, topic_name_speed, topic_name_gps]]
            print("Start reading frames")
            
            start_time = time.time()
            
            # Variables for calculatind d_distance
            last_timestamp = None
            last_speed = None

            processed_frames = 0
            ocr_frames_count = -1

            i = 0

            true_position = None

            text_results = {}

            for connection, timestamp, rawdata in bag.messages(connections=connections):
                # Deserialize ROS1 sensor_msgs/Image
                msg = bag.deserialize(rawdata, connection.msgtype)
            
                if connection.topic == topic_name_cam:

                    ocr_frames_count += 1
                    if ocr_frames_count % frame_interval != 0:
                        continue
                    
                    # Build numpy image (8-bit Bayer)
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))

                    # Convert Bayer -> RGB
                    img_color = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

                    # OCR
                    extracted_text = reader.ocr_frame(img_color, tiled=tiled_ocr, debug=debug)
                    
                    if len(extracted_text) == 0:
                        processed_frames += 1
                        continue

                    frame_time = timestamp / 1e9
                    text_results[frame_time] = extracted_text

                    print(extracted_text)

                    for word in extracted_text:
                        updated = pf.update(word)
                        
                        if not updated:
                            continue
                        
                        i += 1

                        if render_between:
                            OSM.render(pf.particles, str(output_dir /f"resample_nr_{i}_before.png"))

                        if true_position is not None:
                            estimate = pf.get_position(OSM)
                            error = haversine(true_position, estimate)
                            with open(str(output_dir /f"errors.csv"), 'a') as file:
                                file.write(f"{error}, {frame_time}, ocr\n")

                        pf.re_sample()

                        if render_between:
                            OSM.render(pf.particles, str(output_dir /f"resample_nr_{i}_after.png"))
                        
                        pf.check_weights()
                        print(f"Resample nr {i}")

                    processed_frames += 1

                if connection.topic == topic_name_gps:
                    lat = msg.latitude
                    lon = msg.longitude
                    true_position = (lon, lat)
                    
                if connection.topic == topic_name_speed:
                    speed = msg.vehicle_speed  # meters per second

                    if last_timestamp is None or last_speed is None:
                        last_timestamp = timestamp
                        last_speed = speed
                        continue

                    d_time = (timestamp - last_timestamp)
                    av_speed = (last_speed+speed)/2
                    
                    d_dist = d_time*av_speed

                    pf.predict(d_dist)

                    if true_position is not None:
                        estimate = pf.get_position(OSM)
                        error = haversine(true_position, estimate)
                        with open(str(output_dir /f"errors.csv"), 'a') as file:
                            file.write(f"{error}, {frame_time}, speed\n")
                    
                    last_timestamp = timestamp
                    last_speed = speed

        OSM.render(pf.particles, str(output_dir /"after_data.png"))
        end_time = time.time()
        duration = end_time - start_time
        print("Processing results: ", text_results)
        print(f"Processed {processed_frames} frames in {duration:.2f} seconds ")

        with open(str(output_dir /f"errors.csv"), 'a') as file:
            file.write(f"\nProcessed {processed_frames} frames in {duration:.2f} seconds ")
        

if __name__ == "__main__":
    main()