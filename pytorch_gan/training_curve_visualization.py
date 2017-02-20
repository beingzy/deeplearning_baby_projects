""" visualize the training processing

    Author: Yi Zhang <beingzy@gmail.com>
    Date: 2017/02/19
"""
import os
from os.path import join 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 


infile_path = os.path.join(os.getcwd(), "training_log_{}.txt".format('2017feb18'))
train_log = pd.read_csv(infile_path, header=0, index_col=0)

# mean
ax = train_log[['mean_d_fake_data', 'mean_d_real_data']].plot()
ax.get_figure().savefig(os.path.join("images", "fake_vs_real_mean.png"))

# standard deviation
ax = train_log[['std_d_fake_data', 'std_d_real_data']].plot()
ax.get_figure().savefig(os.path.join("images", "fake_vs_real_std.png"))