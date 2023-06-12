import pandas as pd
import sys,os

sensor_data_dir = '../fall_detection_data/sensor_data'

individuals = os.listdir(sensor_data_dir)
print(individuals)