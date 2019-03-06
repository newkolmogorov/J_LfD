from dbn_32 import *
from plots import *

from read import *
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

"""
Read Data
"""

obs_a = pd.read_csv('/home/alivelab/PycharmProjects/HADM/data/g32/Action_Keyframes.csv').values
obs_g = pd.read_csv('/home/alivelab/PycharmProjects/HADM/data/g32/Goal_Keyframes.csv').values

demos_a = np.split(obs_a, list(range(5, 100, 5)))
demos_g = np.split(obs_g, list(range(5, 100, 5)))

demos = [(demos_a[i], demos_g[i]) for i in range(len(demos_a))]

kfs = [i[0].shape[0] for i in demos]
num_demo = len(kfs)


"""
Initialization of DBN
"""


nstates_a, nstates_g = 5, 6

model = DBN(nstates_a, nstates_g, demos, obs_a, obs_g, kfs, num_demo)

model.fit(10)

x_y_z_plot(model.means_action, model.covars_action, obs_a)

