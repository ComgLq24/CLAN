from subprocess import Popen, PIPE

import visdom
import sys
import numpy as np


class Viewer(object):
    def __init__(self):
        self.vis = visdom.Visdom(server='10.16.170.78', port=8097)
        self.loss_data = None
        self.loss_win = None
        if not self.vis.check_connection():
            cmd = sys.executable + '-m visdom server'
            Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_current_loss(self, iteration, loss):
        if self.loss_data is None:
            self.loss_data = {'X': [iteration], 'Y': [loss]}
            self.loss_win = self.vis.line(
                X=np.array(self.loss_data['X']),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'markers': True,
                    'markersize': 6,
                    'title': 'training loss',
                    'xlabel': 'iteration',
                    'ylabel': 'loss'
                }
            )
            return
        self.vis.line(
            X=np.array([iteration]),
            Y=np.array([loss]),
            win=self.loss_win,
            update='append'
        )
