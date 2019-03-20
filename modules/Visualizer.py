import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.utils import *
import numpy as np
from scipy.spatial import ConvexHull


class Visualizer:

    def __init__(self):
        pass

    def print_lightcurve(self, obj, obs_rad, figsize1, figsize2, save_filename='test.png'):
        """

        :param MJD:
        :param obj:
        :param posY:
        :param posX:
        :param save_filename:
        :param SN_found:
        :return:
        """

        num_graphs = 4

        posY = obs_rad
        posX = obs_rad
        MJD = obj['MJD']

        this_fig = plt.figure(figsize=(figsize1,figsize2))

        ax1 = plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.errorbar(MJD + 0.015, obj['state'][:, 0, posY, posX], yerr=obj['state_cov'][:, 0, posY, posX], fmt='b.-',
                     label='Estimated flux')
        plt.errorbar(MJD - 0.015, obj['pred_state'][:, 0, posY, posX], yerr=obj['pred_state_cov'][:, 0, posY, posX],
                     fmt='g.', label='Predicted flux')
        plt.errorbar(MJD, obj['obs_flux'][:, posY, posX], yerr=obj['obs_var_flux'][:, posY, posX], fmt='r.',
                     label='Observed flux')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([min(obj['state'][:, 0, posY, posX]) - 500, max(obj['state'][:, 0, posY, posX]) + 500])
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) )#+ ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0), sharex=ax1)
        plt.errorbar(MJD, obj['state'][:, 1, posY, posX], yerr=obj['state_cov'][:, 2, posY, posX], fmt='b.-',
                     label='Estimated flux velocity')
        plt.errorbar(MJD - 0.03, obj['pred_state'][:, 1, posY, posX], yerr=obj['pred_state_cov'][:, 2, posY, posX],
                     fmt='g.', label='Predicted flux velocity')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (2, 0), sharex=ax1)
        plt.plot(MJD, obj['pixel_flags'][:, posY, posX], '.-', label='Pixel flags')
        plt.plot(MJD, obj['group_flags'][:, posY, posX], '.-', label='Pixel Group flags')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (3, 0), sharex=ax1)
        plt.plot(MJD - 0.011, obj['pred_state_cov'][:, 0, posY, posX], 'y.', label='Pred Flux Variance')
        plt.plot(MJD - 0.01, obj['state_cov'][:, 0, posY, posX], 'y+', label='Flux Variance')
        plt.plot(MJD - 0.001, obj['pred_state_cov'][:, 1, posY, posX], 'b.', label='Pred Flux-Velo Variance')
        plt.plot(MJD + 0.00, obj['state_cov'][:, 1, posY, posX], 'b+', label='Flux-Velo Variance')
        plt.plot(MJD + 0.009, obj['pred_state_cov'][:, 2, posY, posX], 'g.', label='Pred Velo Variance')
        plt.plot(MJD + 0.01, obj['state_cov'][:, 2, posY, posX], 'g+', label='Velo Variance')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([0, 200])

        plt.xlabel('MJD [days]')

        if len(save_filename) > 0:
            plt.savefig(save_filename, bbox_inches='tight')
        plt.close(this_fig)

    def stack_stamps(self, stamps, MJD, max_value=10000):
        """

        :param stamps:
        :param MJD:
        :param max_value:
        :return:
        """
        stack = stamps[0, :]
        prev_time = MJD[0]
        stamps_diam = stamps.shape[1]
        for i in range(1, stamps.shape[0]):
            stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            if MJD[i] - prev_time > 0.5:
                stack = np.hstack((stack, -max_value * np.ones((stamps_diam, 1))))
                stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            stack = np.hstack((stack, stamps[i]))
            prev_time = MJD[i]
        return stack

    def print_stamps(self, obj, figsize1, figsize2, save_filename='test.png'):
        """

        :param MJD:
        :param obj:
        :param save_filename:
        :param SN_found:
        :return:
        """

        num_graphs = 9
        MJD = obj['MJD']

        this_fig = plt.figure(figsize=(figsize1, figsize2))

        plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.imshow(self.stack_stamps(obj['science'], MJD), vmin=0, vmax=600, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        plt.title(
            'Science image, position: ' + str(obj['posY']) + ',' + str(obj['posX']))# + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0))
        plt.imshow(self.stack_stamps(obj['psf'], MJD), vmin=0, vmax=0.05, cmap='Greys_r', interpolation='none')
        plt.title('PSF')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (2, 0))
        plt.imshow(self.stack_stamps(obj['obs_flux'], MJD), vmin=-200, vmax=3000, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (3, 0))
        plt.imshow(self.stack_stamps(obj['obs_var_flux'], MJD), vmin=0, vmax=300, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux variance')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (4, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 0, :], MJD), vmin=-200, vmax=3000, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (5, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 1, :], MJD), vmin=-500, vmax=500, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux velocity')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (6, 0))
        plt.imshow(self.stack_stamps(-obj['pixel_flags'], MJD), vmin=-1, vmax=0, cmap='Greys_r', interpolation='none')
        plt.title('Pixel Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (7, 0))
        plt.imshow(self.stack_stamps(obj['group_flags'], MJD), vmin=-1, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Group Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (8, 0))
        plt.imshow(self.stack_stamps(obj['base_mask'], MJD), vmin=0, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Base mask')
        plt.axis('off')

        plt.tight_layout()

        if len(save_filename) > 0:
            plt.savefig(save_filename, bbox_inches='tight')
        plt.close(this_fig)

    def print_space_states(self, obj, obs_rad, figsize1, figsize2,
                           flux_thresh=200, rate_flux_thresh=50, save_filename='test.png'):
        """

        :param MJD:
        :param obj:
        :param posY:
        :param posX:
        :param save_filename:
        :param SN_found:
        :return:
        """
        posY = obs_rad
        posX = obs_rad

        this_fig = plt.figure(figsize=(figsize1, figsize2))
        epochs = list(range(obj['epochs'][0]-3, obj['epochs'][0]))+list(obj['epochs'])
        print(epochs)

        plt.plot(obj['state'][epochs, 0, posY, posX], obj['state'][epochs, 1, posY, posX], 'b.-', label='Estimation')
        plt.plot(obj['obs_flux'][epochs, posY, posX], np.diff(np.concatenate((np.zeros(1), obj['obs_flux'][epochs, posY, posX]))),
                 'r.-', label='Observation', alpha=0.25)
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        x_min, x_max, y_min, y_max = self.limits(obj['state'], [posY, posX], x_margin=100, y_margin=50, epochs=epochs)
        points = np.column_stack((obj['state'][epochs, 0, posY, posX], obj['state'][epochs, 1, posY, posX]))
        entropy_value = self.entropy_value(points)
        #plt.plot([500, 3000], [150, 0], 'k-')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.vlines(x=flux_thresh, ymin=y_min, ymax=rate_flux_thresh, color='red', zorder=2)
        plt.vlines(x=flux_thresh, ymin=rate_flux_thresh, ymax=y_max, color='red', linestyles='dashed',zorder=2)
        plt.hlines(y=rate_flux_thresh, xmin=x_min, xmax=flux_thresh, color='red', zorder=2)
        plt.hlines(y=rate_flux_thresh, xmin=flux_thresh, xmax=x_max, color='red', linestyles='dashed', zorder=2)
        plt.title('Position: ' + ( '%.2f' % obj['posY']) + ',' + ('%.2f' % obj['posX']) +
                  (' [ Entropy level: %.2f ]' %  entropy_value))#+ ', status: ' + str(obj['status']))
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')
        #plt.text(x_max * 0.75, y_max * 0.75, 'Estimated curve entropy: %.2f' % entropy_value, fontsize=10,
        #         bbox={'facecolor': 'white', 'pad': 10})

        if len(save_filename) > 0:
            plt.savefig(save_filename, bbox_inches='tight')
        plt.close(this_fig)

    def limits(self, state,  pos, x_margin, y_margin, epochs):
        x_min = min(state[epochs, 0, pos[0], pos[1]])
        x_max = max(state[epochs, 0, pos[0], pos[1]])

        y_min = min(state[epochs, 1, pos[0], pos[1]])
        y_max = max(state[epochs, 1, pos[0], pos[1]])

        return x_min-x_margin, x_max+x_margin, y_min-y_margin, y_max+y_margin

    def curve_lenght(self, points):
        d = 0
        for i in range(len(points)-1):
            d = np.linalg.norm(points[i+1] - points[i]) + d
        return d

    def entropy_value(self, points):
        hull = ConvexHull(points)
        d_hull = self.curve_lenght(points[hull.vertices])
        d_curve = self.curve_lenght(points)
        entropy_val = np.log2((d_curve * 2.0) / d_hull)
        return entropy_val