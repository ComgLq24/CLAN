from subprocess import Popen, PIPE

import visdom
import sys
import numpy as np


class Viewer(object):
    def __init__(self, server_ip, port):
        self.vis = visdom.Visdom(server=server_ip, port=port)
        self.loss_data = {'loss_seg': None, 'loss_adv': None, 'loss_weight': None, 'loss_D_s': None,
                          'loss_D_t': None}
        self.loss_win = {'loss_seg': None, 'loss_adv': None, 'loss_weight': None, 'loss_D_s': None,
                         'loss_D_t': None}
        if not self.vis.check_connection():
            cmd = sys.executable + '-m visdom server'
            Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_current_loss(self, iteration, loss, type):
        if self.loss_data[type] is None:
            self.loss_data[type] = {'X': [iteration], 'Y': [loss]}
            self.loss_win[type] = self.vis.line(
                X=np.array(self.loss_data[type]['X']),
                Y=np.array(self.loss_data[type]['Y']),
                opts={
                    'markers': True,
                    'markersize': 6,
                    'title': type,
                    'xlabel': 'iteration',
                    'ylabel': 'loss'
                }
            )
            return
        self.vis.line(
            X=np.array([iteration]),
            Y=np.array([loss]),
            win=self.loss_win[type],
            update='append'
        )
