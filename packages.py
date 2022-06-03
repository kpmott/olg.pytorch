import os

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl

import numpy as np

from scipy.stats import norm
from scipy.optimize import fsolve

from math import floor, ceil

from pynvml import *

from tqdm import tqdm