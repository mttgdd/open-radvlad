import os
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import scipy

from src.radar import load_radar, radar_polar_to_cartesian

class Dataset:
    def __init__(
            self, cfg, date_str,
            start=0, end=-1):
        self.cfg = cfg
        self.date_str = date_str

        # Some files to be read
        self.date_str_dir = os.path.join(
            self.cfg['data_dir'], 
            self.date_str + '-radar-oxford-10k')
        self.radar_timestamps_file = os.path.join(
            self.date_str_dir, 
            "radar.timestamps")
        self.microstrain_file = os.path.join(
            self.date_str_dir, 
            "gps/gps.csv")
        self.radar_dir = os.path.join(
            self.date_str_dir,
            "radar")
        
        # Read radar timestamps
        self.radar_timestamps = Dataset.get_radar_timestamps(
            self.radar_timestamps_file)
        
        # Downsample strategy
        self.radar_timestamps = self.radar_timestamps[::self.cfg['downsample']]

        # Crop timestamps
        self.radar_timestamps = self.radar_timestamps[start:end]

        # Radar pos
        self.microstrain_df = pd.read_csv(self.microstrain_file)
        self.pos = Dataset.get_radar_positions(
            self.microstrain_df, self.radar_timestamps
        )

    @staticmethod
    def get_radar_timestamps(radar_timestamps_file):
        radar_timestamps = []
        with open(radar_timestamps_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                radar_timestamps.append(
                    int(line.strip('\n').split(' ')[0]))
        return radar_timestamps
    
    def load_image(self, png_file):
        # Read fft data, ignore meta data
        timestamps, azimuths, valid, polar_img, radar_resolution = load_radar(png_file) # 1,400,3768

        # Supress early range returns
        polar_img[:, :self.cfg['min_bin']] = 0

        # Chop off late range returns
        polar_img = polar_img[:, :self.cfg['max_bin']]

        if self.cfg['polar']:
            img = np.squeeze(polar_img, 2)
            img = Image.fromarray(img)
            img = img.resize((self.cfg['bin_dim'], self.cfg['num_azis']))
            img = np.array(img)
        elif self.cfg['cartesian']:
            img = radar_polar_to_cartesian(
                azimuths, polar_img, radar_resolution, 
                self.cfg['cart_res'], 
                self.cfg['cart_pw'], True)
            img = np.squeeze(img, 2)

        # Fft optionally
        if self.cfg['fft']:
            img = np.fft.fft(img)
            img = np.abs(img)
            img = img.astype(np.float32)

        # Normalise azimuths
        if not self.cfg['cartesian']:
            norms = np.linalg.norm(img, axis=1)
            img /= norms[:, np.newaxis]

        return img

    def get_png_file(self, idx):
        radar_timestamp = self.radar_timestamps[idx]
        png_file = os.path.join(
            self.radar_dir, 
            f"{radar_timestamp}.png")
        return radar_timestamp, png_file

    @staticmethod
    def get_radar_positions(microstrain_df, radar_timestamps):

        # Use kd-tree for fast lookup
        gt_tss = microstrain_df.timestamp.to_numpy()
        keys = np.expand_dims(gt_tss, axis=-1)
        tree = cKDTree(keys)
        query = np.array(radar_timestamps)
        query = np.expand_dims(query, axis=-1)
        _, out = tree.query(query)
        gt_idxs = out.tolist()

        # Build output
        pos = {}
        for radar_timestamp, gt_idx in zip(radar_timestamps, gt_idxs):
            pos[radar_timestamp] = np.array(
                (microstrain_df.iloc[gt_idx].northing, 
                microstrain_df.iloc[gt_idx].easting))
        
        return pos
    
    def __len__(self):
        return len(self.radar_timestamps)

    def __getitem__(self, idx):
        # Image filename
        _, png_file = self.get_png_file(idx)

        # Get sample
        img = self.load_image(png_file)

        return img