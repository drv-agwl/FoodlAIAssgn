import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import datetime
from utils import find_closest_element_sorted

data_dir_rgb = "./pasra/pasra/0/rgb"
data_dir_ir = "./pasra/pasra/0/ir"

smell_data = pd.read_csv("./pasra/pasra/0/nose/nose.csv")

# rgb_timestamps = [datetime.datetime.fromtimestamp(osp.getctime(osp.join(data_dir_rgb, x))) for x in sorted(os.listdir(data_dir_rgb))]
# ir_timestamps = [datetime.datetime.fromtimestamp(osp.getctime(osp.join(data_dir_ir, x))) for x in sorted(os.listdir(data_dir_ir))]

rgb_timestamps = sorted([float(x.split('.')[0]) for x in os.listdir(data_dir_rgb)])
ir_timestamps = sorted([float(x.split('.')[0]) for x in os.listdir(data_dir_ir)])
nose_timestamps = list(smell_data.iloc[:, -1])

rgb_filtered_timestamps = [find_closest_element_sorted(x, rgb_timestamps) for x in nose_timestamps]
ir_filtered_timestamps = [find_closest_element_sorted(x, ir_timestamps) for x in nose_timestamps]
nose_data = np.array(smell_data.iloc[:, :-1])

for image in os.listdir(data_dir_rgb):
    img_path = osp.join(data_dir_rgb, image)
    img = Image.open(img_path)
    plt.imshow(np.array(img))
    plt.show()
