import pickle
import argparse
import os
import logging
import pickle
import random
from datetime import datetime

import numpy as np

import data_processor
from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from shutil import copy2
from mediaio import ffmpeg


with open('preprocess.pkl', 'rb') as f:
	print(type(f))
	data = pickle.load(f)