import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, num_obs, results_path, plots_path,
                 obs_rad=10, figsize1=12, figsize2=8):

        self.results_path = results_path
        self.plots_path = plots_path

        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.num_obs = num_obs
        self.obs_rad = obs_rad
        self.obs_diam = self.obs_rad * 2 + 1


    def set_plot(self, npz_list, coords):
        a, b = coords[0] - self.obs_rad, coords[0] + self.obs_rad + 1
        c, d = coords[1] - self.obs_rad, coords[1] + self.obs_rad + 1
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        self.self.mjd = []
        for npz in npz_list:
            data = np.load(npz)
            self.science = data['science'][a:b, c:d]
            self.obs_flux = data['obs_flux'][a:b, c:d]
            self.obs_flux_var = data['obs_flux_var'][:, a:b, c:d]
            self.state = data['state'][:, a:b, c:d]
            self.state_cov = data['state_cov'][:, a:b, c:d]
            self.diff = data['diff'][:, a:b, c:d]
            self.psf = data['psf']
            self.self.mjd.append(data['self.mjd'])
            data.close()


    def plot_lightcurve(self, coords=[-1, -1], position=[0, 0], filename='test.png'):

        num_graphs = 4
        if coords[0]==-1 and coords[1]==-1:
            coords = np.array([self.obs_rad, self.obs_rad])
        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        ax1 = plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.errorbar(self.self.mjd + 0.015, self.state[:, 0, coords[0], coords[1]], yerr=self.state_cov[:, 0, coords[0], coords[1]], fmt='b.-',
                     label='Estimated flux')
        #plt.errorbar(self.self.mjd - 0.015, obj['pred_state'][:, 0, coords[0], coords[1]], yerr=obj['pred_state_cov'][:, 0, coords[0], coords[1]],
        #             fmt='g.', label='Predicted flux')
        plt.errorbar(self.self.mjd, self.obs_flux[:, coords[0], coords[1]], yerr=self.obs_flux_var[:, coords[0], coords[1]], fmt='r.',
                     label='Observed flux')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.self.mjd[0] - 1,self.self.mjd[-1] + 1)
        plt.ylim([min(self.state[:, 0, coords[0], coords[1]]) - 500, max(self.state[:, 0, coords[0], coords[1]]) + 500])
        plt.title('Position: ' + position[0] + ',' + position[1])

        plt.subplot2grid((num_graphs, 1), (1, 0), sharex=ax1)
        plt.errorbar(self.self.mjd, self.state[:, 1, coords[0], coords[1]], yerr=self.state_cov[:, 2, coords[0], coords[1]], fmt='b.-',
                     label='Estimated flux velocity')
        #plt.errorbar(self.self.mjd - 0.03, obj['pred_state'][:, 1, coords[0], coords[1]], yerr=obj['pred_state_cov'][:, 2, coords[0], coords[1]],
        #             fmt='g.', label='Predicted flux velocity')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.self.mjd[0] - 1, self.self.mjd[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (2, 0), sharex=ax1)
        plt.plot(self.self.mjd, obj['pixel_flags'][:, coords[0], coords[1]], '.-', label='Pixel flags')
        plt.plot(self.self.mjd, obj['group_flags'][:, coords[0], coords[1]], '.-', label='Pixel Group flags')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.self.mjd[0] - 1, self.self.mjd[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (3, 0), sharex=ax1)

        #plt.plot(self.self.mjd - 0.011, obj['pred_state_cov'][:, 0, coords[0], coords[1]], 'y.', label='Pred Flux Variance')
        plt.plot(self.self.mjd - 0.01, self.state_cov[:, 0, coords[0], coords[1]], 'y+', label='Flux Variance')
        #plt.plot(self.self.mjd - 0.001, obj['pred_state_cov'][:, 1, coords[0], coords[1]], 'b.', label='Pred Flux-Velo Variance')
        plt.plot(self.self.mjd + 0.00, self.state_cov[:, 1, coords[0], coords[1]], 'b+', label='Flux-Velo Variance')
        #plt.plot(self.self.mjd + 0.009, obj['pred_state_cov'][:, 2, coords[0], coords[1]], 'g.', label='Pred Velo Variance')
        plt.plot(self.self.mjd + 0.01, self.state_cov[:, 2, coords[0], coords[1]], 'g+', label='Velo Variance')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.self.mjd[0] - 1, self.self.mjd[-1] + 1)
        plt.ylim([0, 200])

        plt.xlabel('self.mjd [days]')

        plt.savefig(filename, bbox_inches='tight')
        plt.close(this_fig)


    def stack_stamps(self, stamps, max_value=10000):
        """

        :param stamps:
        :param self.mjd:
        :param max_value:
        :return:
        """
        stack = stamps[0, :]
        prev_time = self.self.mjd[0]
        stamps_diam = stamps.shape[1]
        for i in range(1, stamps.shape[0]):
            stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            if self.self.mjd[i] - prev_time > 0.5:
                stack = np.hstack((stack, -max_value * np.ones((stamps_diam, 1))))
                stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            stack = np.hstack((stack, stamps[i]))
            prev_time = self.self.mjd[i]
        return stack


    def print_stamps(self, obj, filename='test_stamps.png'):
        """

        :param obj:
        :param save_filename:
        :param SN_found:
        :return:
        """

        num_graphs = 9

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.imshow(self.stack_stamps(obj['science'], self.mjd), vmin=0, vmax=600, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        plt.title(
            'Science image, position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0))
        plt.imshow(self.stack_stamps(obj['psf'], self.mjd), vmin=0, vmax=0.05, cmap='Greys_r', interpolation='none')
        plt.title('PSF')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (2, 0))
        plt.imshow(self.stack_stamps(obj['obs_flux'], self.mjd), vmin=-200, vmax=3000, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (3, 0))
        plt.imshow(self.stack_stamps(obj['obs_var_flux'], self.mjd), vmin=0, vmax=300, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux variance')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (4, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 0, :], self.mjd), vmin=-200, vmax=3000, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (5, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 1, :], self.mjd), vmin=-500, vmax=500, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux velocity')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (6, 0))
        plt.imshow(self.stack_stamps(-obj['pixel_flags'], self.mjd), vmin=-1, vmax=0, cmap='Greys_r', interpolation='none')
        plt.title('Pixel Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (7, 0))
        plt.imshow(self.stack_stamps(obj['group_flags'], self.mjd), vmin=-1, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Group Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (8, 0))
        plt.imshow(self.stack_stamps(obj['base_mask'], self.mjd), vmin=0, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Base mask')
        plt.axis('off')

        plt.tight_layout()


        plt.savefig(filename, bbox_inches='tight')
        plt.close(this_fig)

    def print_space_states(self,  obj, posY=-1, posX=-1, save_filename='', SN_found=False):
        """

        :param self.mjd:
        :param obj:
        :param posY:
        :param posX:
        :param save_filename:
        :param SN_found:
        :return:
        """

        if posY == -1:
            posY = self.obs_rad
        if posX == -1:
            posX = self.obs_rad

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.plot(obj['state'][:, 0, posY, posX], obj['state'][:, 1, posY, posX], 'b.-', label='Estimation')
        plt.plot(obj['obs_flux'][:, posY, posX], np.diff(np.concatenate((np.zeros(1), obj['obs_flux'][:, posY, posX]))),
                 'r.-', label='Observation', alpha=0.25)
        plt.grid()
        plt.plot([500, 500, 3000], [1000, 150, 0], 'k-', label='Thresholds')
        plt.legend(loc=0, fontsize='small')
        plt.plot([500, 3000], [150, 0], 'k-')
        plt.xlim(-500, 3000)
        plt.ylim(-500, 1000)
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_space_states', bbox_inches='tight')
        plt.close(this_fig)


    def plot_candidate(self, ccd, field, semester):
        pass