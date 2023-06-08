import numpy as np
import matplotlib.pyplot as plt
from src.environment.odor_plumes import *
from src.environment.odor_senses import *


## Objective is to calculate statistics of the plume at different locations, parametrized in polar coordinates from source
## Plume is imported from the odor_plume.py class
## Desire a pdf of conc, grad, hrc at polar coordinates

class PlumeSummary(object):

    def __init__(self, config, ax1_bins, ax2_bins, n_stat_bins=10, stat_bin_type='linear', axes='polar', n_points=1000, samples_per_point = 5, feats=['conc', 'grad', 'hrc']):
        self.plume = OdorPlumeFromMovie(config, load=False)
        self.feats = feats
        self.n_stats = len(feats)
        self.feat_bounds = [[0, 255], [-255, 255], [-255*255, 255*255]]
        self.ax1_bins = np.array(ax1_bins) ## Could be r or x
        self.ax2_bins = np.array(ax2_bins) ## Could be theta or y
        self.n_points = n_points
        self.samples_per_point = samples_per_point
        self.bin_type = axes
        self.n_stat_bins = n_stat_bins
        self.stat_bin_type = stat_bin_type
        self._make_stat_bins()
        self.counts = np.zeros((len(ax1_bins)-1, len(ax2_bins)-1) + (n_stat_bins-1,)*self.n_stats)
        self.marginals = np.zeros((len(ax1_bins)-1, len(ax2_bins)-1, self.n_stats, self.n_stat_bins-1))
        agent_dict = config['agent']
        self.mm_per_px = config["plume"]["MM_PER_PX"]
        self.std_left_box, self.std_right_box = self._make_L_R_std_box(mm_per_px = self.mm_per_px, antenna_height_mm = agent_dict['ANTENNA_LENGTH_MM'], antenna_width_mm = agent_dict['ANTENNA_WIDTH_MM'])

    def _make_stat_bins(self):
        self.stat_bins = []
        if self.stat_bin_type == 'linear':
            for i in range(self.n_stats):
                bounds = self.feat_bounds[i]
                self.stat_bins.append(np.linspace(bounds[0], bounds[1], self.n_stat_bins))
    
    @staticmethod
    def _make_L_R_std_box(mm_per_px, antenna_height_mm, antenna_width_mm):
        """
        Make a standard left and right box for the antenna that can be rotated and translated to match the orientation and position of the antenna.
        The boxes have dimension (px_height*px_width, 2) where px_height and px_width are the height and width of the antenna in pixels.
        x,y coordinates are in units of mm.
        """
        ## Height is long axis of box, typically.
        ## If even then split left and right evenly. If odd then share middle. 

        px_height = round(antenna_height_mm/mm_per_px)
        px_width = round(antenna_width_mm/mm_per_px)

        x_coords = np.linspace(0, px_width,px_width)*mm_per_px
        y_coords = np.flip(np.linspace(-px_height/2, px_height/2,px_height)*mm_per_px)

        ## Make a big box with meshgrid.
        big_box = np.zeros((px_height*px_width,2))
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        big_box = np.column_stack((x_grid.flatten(), y_grid.flatten()))

        # Separate the box into left and right halves
        left_box = big_box[big_box[:, 1] >= 0]
        right_box = big_box[big_box[:, 1] <= 0]
        return left_box, right_box

    @staticmethod
    def _rotate_points(points, theta):
        ## Assumes that points is a 2D array with x and y coordinates in the first and second columns, respectively.
        x = np.cos(theta) * points[:, 0] - np.sin(theta) * points[:, 1]
        y = np.sin(theta) * points[:, 0] + np.cos(theta) * points[:, 1]
        return np.column_stack((x, y))
    
    def _set_sensor_idxs(self, pos_set, theta=np.pi):
        ## Set the sensor idxs for the given position and orientation
        self.left_sensors = self._rotate_points(self.std_left_box, theta)[None, :] + pos_set[:, None, :]
        self.right_sensors = self._rotate_points(self.std_right_box, theta)[None, :] + pos_set[:, None, :]
        self.left_sensor_idxs = np.rint(self.left_sensors/self.mm_per_px).astype(int)
        self.right_sensor_idxs = np.rint(self.right_sensors/self.mm_per_px).astype(int)
    
    def _compute_stats_at_location(self, points):
        ## Return the statistics of the plume at the given points in mm
        ## points has shape (n_points, 2)
        ## Set the sensor idxs for the given position and orientation
        self._set_sensor_idxs(points)
        ## Initialize the plume at some random place
        rng = np.random.default_rng(seed=0)
        self.plume.pick_random_frame(rng, delay=self.samples_per_point)
        ## Initialize the statistics
        self.prev_left_sensor = np.zeros(self.n_points)
        self.prev_right_sensor = np.zeros(self.n_points)
        self.current_stats = np.zeros((self.n_points, self.n_stats))
        all_stats = np.zeros((self.samples_per_point, self.n_points, self.n_stats))
        ## Compute the statistics of each point for each frame
        for t in range(self.samples_per_point):
            ## Compute mean odors in left and right sensors
            try:
                self.left_sensor_vals = self.plume.frame[self.left_sensor_idxs[:, :,0], self.left_sensor_idxs[:, :, 1]]
            except IndexError:
                self.left_sensor_vals = np.zeros_like(self.left_sensor_idxs[...,0]) ## If the agent is out of bounds, then the odor is zero.
            try:
                self.right_sensor_vals = self.plume.frame[self.right_sensor_idxs[...,0], self.right_sensor_idxs[...,1]]
            except IndexError:
                self.right_sensor_vals = np.zeros_like(self.left_sensor_idxs[...,0])
            self.mean_left_sensor = np.mean(self.left_sensor_vals, axis=1)
            self.mean_right_sensor = np.mean(self.right_sensor_vals, axis=1)
            ## Compute the statistics
            self._compute_stats()
            all_stats[t] = self.current_stats
            ## Advance the plume
            self.plume.advance(rng)
            ## Update prev sensor vals
            self.prev_left_sensor = self.mean_left_sensor
            self.prev_right_sensor = self.mean_right_sensor
        ## Broadcast into 2D array of dimension (n_points*samples_per_point, n_stats)
        all_stats = all_stats.reshape(-1, self.n_stats)
        return all_stats
            
    def _compute_stats(self):
        ## Compute the statistics of the plume given the mean left and right sensor values
        ## current_stats has shape (n_points, n_stats)
        self.current_stats[:, 0] = (self.mean_left_sensor + self.mean_right_sensor)/2 ## concentration
        self.current_stats[:, 1] = self.mean_left_sensor - self.mean_right_sensor ## gradient
        self.current_stats[:, 2] = self.prev_left_sensor*self.mean_right_sensor - self.prev_right_sensor*self.mean_left_sensor ## hrc
    
    def _populate_counts(self):
        for i in range(len(self.ax1_bins)-1):
            for j in range(len(self.ax2_bins)-1):
                ## Generate a set of points in the bin
                print(i,j)
                points = self._generate_points_in_bin(i, j)
                ## Normalize the counts by the number of points
                stats = self._compute_stats_at_location(points) ## stats has shape (n_points*samples_per_point, n_stats)
                self.counts[i, j], self.marginals[i, j] = self._bin_stats(stats)
    
    def _generate_points_in_bin(self, i, j):
        ## Generate a set of points in the given bin; np array of dim (n_points, 2)
        ## TODO: check that this works
        np.random.seed(0)
        ax1_samples = np.random.uniform(low=self.ax1_bins[i], high=self.ax1_bins[i+1], size=(self.n_points, 1))
        ax2_samples = np.random.uniform(low=self.ax2_bins[j], high=self.ax2_bins[j+1], size=(self.n_points, 1))
        if self.bin_type == 'polar':
            x_samples = ax1_samples * np.cos(ax2_samples) + self.plume.source_location[0]
            y_samples = ax1_samples * np.sin(ax2_samples) + self.plume.source_location[1]
        elif self.bin_type == 'cartesian':
            x_samples = ax1_samples + self.plume.source_location[0]
            y_samples = ax2_samples + self.plume.source_location[1]
        return np.hstack((x_samples, y_samples))

    def _bin_stats(self, stats):
        ## Bin the given statistics into the stat bins
        joint = np.histogramdd(stats, bins=self.stat_bins)[0]/(self.n_points*self.samples_per_point)
        marginal = np.array([np.sum(joint, axis=tuple(dim for dim in range(joint.ndim) if dim != i)) for i in range(self.n_stats)]) ## marginal is 2D array of shape (n_stats, n_stat_bins)
        return joint, marginal

    def _plot_histograms(self, plot_type='pdf'):
        ## We make a n_features sets of plots, each of which are ax1_bins x ax2_bins histograms
        ## Now, we plot the histograms
        bin_centers_1 = [(self.ax1_bins[i]+self.ax1_bins[i+1])/2 for i in range(len(self.ax1_bins)-1)]
        bin_centers_2 = [(self.ax2_bins[i]+self.ax2_bins[i+1])/2 for i in range(len(self.ax2_bins)-1)]
        if plot_type == 'pdf':
            for i in range(self.n_stats):
                fig, axes = plt.subplots(len(self.ax2_bins)-1, len(self.ax1_bins)-1, sharex=True, sharey=True, figsize=(3*len(self.ax1_bins),2*len(self.ax2_bins)))
                if self.bin_type == 'polar':
                    for j in range(len(self.ax1_bins)-1):
                        for k1 in range(len(self.ax2_bins)-1):
                            k = len(self.ax2_bins)-k1-2
                            # Create a subplot at the specified position
                            dist = self.marginals[j, k][i]
                            axes[k1, j].bar(self.stat_bins[i][:-1], dist, width=self.stat_bins[i][1]-self.stat_bins[i][0], align='edge', label=f"({bin_centers_1[j]:.2f}, {bin_centers_2[k]:.2f})")
                            axes[k1, j].legend()
                            axes[k1, j].set_yscale('log')
                elif self.bin_type == 'cartesian':
                    fig, axes = plt.subplots(len(self.ax2_bins)-1, len(self.ax1_bins)-1)
                    for j in range(len(self.ax1_bins)-1):
                        for k1 in range(len(self.ax2_bins)-1):
                            k = len(self.ax2_bins)-k1-2
                            dist = self.marginals[j, k][i]
                            axes[k1, j].bar(self.stat_bins[i][:-1], dist, width=self.stat_bins[i][1]-self.stat_bins[i][0], align='edge', label=f"({bin_centers_1[j]:.2f}, {bin_centers_2[k]:.2f})")
                            axes[k1, j].legend()
                            axes[k1, j].set_yscale('log')
                plt.savefig(f"histogram_{self.feats[i]}.png")

        elif plot_type == 'heatmap':
            ## Plot the heatmap of the marginal distribution averages
            marg_avg = np.mean(self.marginals, axis=3)
            fig, axes = plt.subplots(1, self.n_stats, sharex=True, sharey=True, figsize=(3*self.n_stats, 2))
            for i in range(self.n_stats):
                im = axes[i].imshow(marg_avg[:, :, i].T, cmap='hot')
                axes[i].set_xticks(np.arange(len(self.ax1_bins)-1))
                axes[i].set_yticks(np.arange(len(self.ax2_bins)-1))
                axes[i].set_xticklabels([f"{bin_centers_1[j]:.0f}" for j in range(len(self.ax1_bins)-1)])
                axes[i].set_yticklabels([f"{bin_centers_2[j]:.2f}" for j in range(len(self.ax2_bins)-1)])
                fig.colorbar(im, ax=axes[i])
                axes[i].set_title(f"{self.feats[i]}")
            plt.savefig("heatmap.png")

            if self.bin_type == 'polar':
                ## Plot polar heatmap of marg_avg
                pass


        """
        for i in range(self.n_stats):
            fig, axes = plt.subplots(len(self.ax1_bins)-1, len(self.ax2_bins)-1)
            for j in range(len(self.ax1_bins)-1):
                for k in range(len(self.ax2_bins)-1):
                    dist = self.counts[j, k]
                    marginal_data = np.sum(dist, axis=tuple(dim for dim in range(dist.ndim) if dim != i))/self.n_points
                    print(marginal_data, i)
                    axes[j, k].bar(self.stat_bins[i][:-1], marginal_data, width=[self.stat_bins[i][b+1]-self.stat_bins[i][b] for b in range(self.n_stat_bins-1)], align='edge')
                    axes[j, k].set_title("({}, {})".format(j, k))
            plt.savefig("histogram_{}.png".format(i))
        """
    def plot(self, ptype='pdf'):
        self._populate_counts()
        self._plot_histograms(plot_type=ptype)