from subprocess import Popen, PIPE

import visdom
import sys
import numpy as np
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


class Viewer(object):
    def __init__(self, server_ip, port):
        self.vis = visdom.Visdom(server=server_ip, port=port)
        self.loss_data = {'loss_seg': None, 'loss_adv': None, 'loss_weight': None, 'loss_D_s': None,
                          'loss_D_t': None}
        self.loss_win = {'loss_seg': None, 'loss_adv': None, 'loss_weight': None, 'loss_D_s': None,
                         'loss_D_t': None}
        self.image_win = None
        self.mask_win = None
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

    def display_image(self, mask, name, iteration):
        mask = colorize_mask(mask).convert("RGB")
        mask = np.asarray(mask, dtype=np.uint8).transpose((2, 0, 1))
        self.vis.image(mask, win="masks", opts=dict(caption=name[0] + '_mask'+'_'+str(iteration), store_history=True))
