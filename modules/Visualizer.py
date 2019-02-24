import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.utils import *
import numpy as np
from scipy.spatial import ConvexHull


class Visualizer:

    def __init__(self,results_path, plots_path,
                 obs_rad=10, figsize1=12, figsize2=8):

        self.results_path = results_path
        self.plots_path = plots_path
        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.obs_rad = obs_rad
        self.obs_diam = self.obs_rad * 2 + 1

    def set_rectangle(self, coords):

        a, b = coords[0] - self.obs_rad -1, coords[0] + self.obs_rad + 1
        c, d = coords[1] - self.obs_rad -1, coords[1] + self.obs_rad + 1

        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)

        return a, b, c, d

    def set_plot_states(self, npz_list, sources_dict):
        sources_coords = []
        mjd_idxs = []
        print(sources_dict)
        for k,v in sources_dict.items():
            sources_coords.append(sources_dict[k]['coords'])
            mjd_idxs.append(sources_dict[k]['mjd_id'])
        n_obs = len(npz_list)

        for i in range(len(mjd_idxs)):
            obs_flux = []
            obs_flux_var = []
            state = []
            state_cov = []
            mjds = []
            a, b, c, d = self.set_rectangle(sources_coords[i])

            if mjd_idxs[i]<=n_obs-3:
                mjd_interval = range(mjd_idxs[i]-3, mjd_idxs[i]+3)
            elif mjd_idxs[i]==n_obs-2:
                mjd_interval = range(mjd_idxs[i]-4, mjd_idxs[i]+2)
            else:
                mjd_interval = range(mjd_idxs[i]-5, mjd_idxs[i] + 1)

            for j in mjd_interval:
                data = np.load(npz_list[j])
                mjds.append(data['mjd'])
                flux, var_flux, diff, psf = calc_fluxes(str(data['diff_name']), str(data['psf_name']), str(data['invvar_name']),
                                                        str(data['aflux_name']))
                obs_flux.append(flux[a:b, c:d])
                obs_flux_var.append(var_flux[a:b, c:d])
                state.append(data['state'][:, a:b, c:d])
                state_cov.append(data['state_cov'][:, a:b, c:d])
                data.close()

            obs_flux = np.array(obs_flux)
            obs_flux_var = np.array(obs_flux_var)
            state = np.array(state)
            state_cov = np.array(state_cov)

            self.print_space_states(sources_coords[i], state, state_cov, obs_flux, obs_flux_var, pos=[-1, -1],
                               save_filename='space_curve_source_%d.png' % i)

    def print_space_states(self,  coords, state, state_cov, flux, flux_var, pos=[-1, -1], save_filename='space_curve.png'):

        if pos[0] == -1 and pos[1] == -1:
            pos[0] = self.obs_rad
            pos[1] = self.obs_rad

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))
        plt.errorbar(state[:, 0, pos[0], pos[1]], state[:, 1, pos[0], pos[1]], c='b', marker='o',
                     yerr=state_cov[:, 1, pos[0], pos[1]],
                     xerr=state_cov[:, 0, pos[0], pos[1]], label='Estimation')
        #plt.errorbar(flux[:, pos[0],pos[1]], np.diff(np.concatenate((np.zeros(1), flux[:, pos[0], pos[1]]))),
        #         c='k', marker='o', xerr=flux_var[:, pos[0], pos[1]], label='Observation', alpha=0.25)
        points = np.column_stack((state[:, 0, pos[0], pos[1]], state[:, 1, pos[0], pos[1]]))
        entropy_val = self.entropy_value(points)
        #plt.text(1250, 300, 'Estimated curve entropy: %.2f' % entropy_val, fontsize=10,
        #         bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

        x_min, x_max, y_min, y_max = self.limits(state, pos, x_margin=100, y_margin=50)

        plt.vlines(x=200, ymin=y_min, ymax=50, color='red', zorder=2)
        plt.vlines(x=200, ymin=50, ymax=y_max, color='red', linestyles='dashed',zorder=2)
        plt.hlines(y=50, xmin=x_min, xmax=200, color='red', zorder=2)
        plt.hlines(y=50, xmin=200, xmax=x_max, color='red', linestyles='dashed', zorder=2)

        plt.grid()
        plt.legend(loc=0, fontsize='small')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Position: ' + str(coords[0]) + ',' + str(coords[1]) )
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')

        plt.text(x_max*0.75, y_max*0.75, 'Estimated curve entropy: %.2f' % entropy_val, fontsize=10,
                 bbox={'facecolor':'white', 'pad':10})

        plt.savefig(save_filename, bbox_inches='tight')
        plt.close(this_fig)

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

    def limits(self, state,  pos, x_margin, y_margin):
        x_min = min(state[:, 0, pos[0], pos[1]])
        x_max = max(state[:, 0, pos[0], pos[1]])

        y_min = min(state[:, 1, pos[0], pos[1]])
        y_max = max(state[:, 1, pos[0], pos[1]])

        return x_min-x_margin, x_max+x_margin, y_min-y_margin, y_max+y_margin



"""


    def set_plot_obs_flux(self, npz_list, coords): #Plot for lightcurve
        a, b, c, d = self.set_rectangle(coords)
        i = 0

        for npz in npz_list:
            print(i)
            data = np.load(npz)
            #self.obs_flux.append(data['obs_flux'][a:b, c:d])
            self.mjd.append(data['mjd'])
            self.obs_flux_var.append(data['obs_flux_var'][a:b, c:d])
            self.pixel_flags.append(data['pixel_flags'][a:b, c:d])
            data.close()
            i=i+1

        self.mjd=np.array(self.mjd)
        self.obs_flux = np.array(self.obs_flux)
        self.obs_flux_var = np.array(self.obs_flux_var)
        self.pixel_flags = np.array(self.pixel_flags)

    def set_plot_filter_stim(self, npz_list, coords ):
        a, b, c, d = self.set_rectangle(coords)
        i = 0

        for npz in npz_list:
            print(i)
            data = np.load(npz)
            self.state.append(data['state'][:, a:b, c:d])
            self.state_cov.append(data['state_cov'][:, a:b, c:d])
            self.pred_state.append(data['pred_state'][:, a:b, c:d])
            self.pred_state_cov.append(data['pred_state_cov'][:, a:b, c:d])

            data.close()
            i=i+1

        self.state = np.array(self.state)
        self.state_cov = np.array(self.state_cov)
        self.pred_state = np.array(self.pred_state)
        self.pred_state_cov = np.array(self.pred_state_cov)

    def set_plot_group_flags(self,  npz_list, coords):
        a, b, c, d = self.set_rectangle(coords)

        for npz in npz_list:
            data = np.load(npz)
            self.pixel_group_flags.append(data['pixel_group_flags'][a:b, c:d])
            self.science.append(data['science'][a:b, c:d])
            self.psf.append(data['psf'])
            data.close()

        self.pixel_group_flags = np.array(self.pixel_group_flags)
        self.science = np.array(self.science)
        self.psf = np.array(self.psf)

    def plot_lightcurve(self, coords, pos=[-1, -1], filename='test_nn.png'):

        num_graphs = 4
        if pos[0]==-1 and pos[1]==-1:
            pos = np.array([self.obs_rad, self.obs_rad])

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        ax1 = plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.errorbar(self.mjd + 0.015, self.state[:, 0, pos[0], pos[1]], yerr=self.state_cov[:, 0, pos[0], pos[1]], fmt='b.-',
                     label='Estimated flux')
        plt.errorbar(self.mjd - 0.015, self.pred_state[:, 0, pos[0], pos[1]], yerr=self.pred_state_cov[:, 0, pos[0], pos[1]],
                     fmt='g.', label='Predicted flux')
        plt.errorbar(self.mjd, self.obs_flux[:, pos[0], pos[1]], yerr=self.obs_flux_var[:, pos[0], pos[1]], fmt='r.',
                     label='Observed flux')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.mjd[0] - 1,self.mjd[-1] + 1)
        plt.ylim([min(self.state[:, 0, pos[0], pos[1]]) - 500, max(self.state[:, 0, pos[0], pos[1]]) + 500])
        plt.title('Position: ' + str(coords[0]) + ',' + str(coords[1]))

        plt.subplot2grid((num_graphs, 1), (1, 0), sharex=ax1)
        plt.errorbar(self.mjd, self.state[:, 1, pos[0], pos[1]], yerr=self.state_cov[:, 2, pos[0], pos[1]], fmt='b.-',
                     label='Estimated flux velocity')
        plt.errorbar(self.mjd - 0.03, self.pred_state[:, 1, pos[0], pos[1]], yerr=self.pred_state_cov[:, 2, pos[0], pos[1]],
                     fmt='g.', label='Predicted flux velocity')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.mjd[0] - 1, self.mjd[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (2, 0), sharex=ax1)
        plt.plot(self.mjd, self.pixel_flags[:, pos[0], pos[1]], '.-', label='Pixel flags')
        plt.plot(self.mjd, self.pixel_group_flags[:, pos[0], pos[1]], '.-', label='Pixel Group flags')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.mjd[0] - 1, self.mjd[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (3, 0), sharex=ax1)
        plt.plot(self.mjd - 0.011, self.pred_state_cov[:, 0, pos[0], pos[1]], 'y.', label='Pred Flux Variance')
        plt.plot(self.mjd - 0.01, self.state_cov[:, 0, pos[0], pos[1]], 'y+', label='Flux Variance')
        plt.plot(self.mjd - 0.001, self.pred_state_cov[:, 1, pos[0], pos[1]], 'b.', label='Pred Flux-Velo Variance')
        plt.plot(self.mjd + 0.00, self.state_cov[:, 1, pos[0], pos[1]], 'b+', label='Flux-Velo Variance')
        plt.plot(self.mjd + 0.01, self.state_cov[:, 2, pos[0], pos[1]], 'g+', label='Velo Variance')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(self.mjd[0] - 1, self.mjd[-1] + 1)
        plt.ylim([0, 200])

        plt.xlabel('MJD [days]')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(this_fig)


    def stack_stamps(self, stamps, max_value=10000):

        stack = stamps[0, :]
        prev_time = self.mjd[0]
        stamps_diam = stamps.shape[1]
        for i in range(1, stamps.shape[0]):
            stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            if self.mjd[i] - prev_time > 0.5:
                stack = np.hstack((stack, -max_value * np.ones((stamps_diam, 1))))
                stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            stack = np.hstack((stack, stamps[i]))
            prev_time = self.mjd[i]
        return stack



    def print_stamps(self, coords, filename='test_stamps.png'):


        num_graphs = 8

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.imshow(self.stack_stamps(self.science[10:], self.mjd[10:]), vmin=0, vmax=600, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        plt.title(
            'Science image, position: ' + str(coords[0]) + ',' + str(coords[1]))# + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0))
        plt.imshow(self.stack_stamps(self.psf[10:], self.mjd[10:]), vmin=0, vmax=0.05, cmap='Greys_r', interpolation='none')
        plt.title('PSF')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (2, 0))
        plt.imshow(self.stack_stamps(self.obs_flux[10:], self.mjd[10:]), vmin=-200, vmax=3000, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (3, 0))
        plt.imshow(self.stack_stamps(self.obs_flux_var[10:], self.mjd[10:]), vmin=0, vmax=300, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux variance')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (4, 0))
        plt.imshow(self.stack_stamps(self.state[10:, 0, :], self.mjd[10:]), vmin=-200, vmax=3000, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (5, 0))
        plt.imshow(self.stack_stamps(self.state[10:, 1, :], self.mjd[10:]), vmin=-500, vmax=500, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux velocity')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (6, 0))
        plt.imshow(self.stack_stamps(-self.pixel_flags[10:], self.mjd[10:]), vmin=-1, vmax=0, cmap='Greys_r', interpolation='none')
        plt.title('Pixel Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (7, 0))
        plt.imshow(self.stack_stamps(self.pixel_group_flags[10:], self.mjd[10:]), vmin=-1, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Group Flags')
        plt.axis('off')

        #plt.subplot2grid((num_graphs, 1), (8, 0))
        #plt.imshow(self.stack_stamps(self.mask, self.mjd), vmin=0, vmax=1, cmap='Greys_r', interpolation='none')
        #plt.title('Base mask')
        #plt.axis('off')

        plt.tight_layout()


        plt.savefig(filename, bbox_inches='tight')
        plt.close(this_fig)
        """
