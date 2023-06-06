import numpy as np
import matplotlib.pyplot as plt
from src.environment.odor_plumes import *
from src.environment.odor_senses import *


## Objective is to calculate statistics of the plume at different locations, parametrized in polar coordinates from source
## Plume is imported from the odor_plume.py class
## Desire a pdf of conc, grad, hrc at polar coordinates

class PlumeSummary(object):

    def __init__(self, config, ax1_bins, ax2_bins, n_stat_bins=10, stat_bin_type='linear', axes='polar', n_points=1000):
        self.plume = OdorPlumeFromMovie(config, load=False)
        self.features = OdorFeatures(config) ## OdorFeatures class used to detect features
        self.n_stats = len(self.features.func_evals)
        self.feat_names = self.features.features
        self.ax1_bins = ax1_bins ## Could be r or x
        self.ax2_bins = ax2_bins ## Could be theta or y
        self.n_points = n_points
        self.axes = axes
        self.n_stat_bins = n_stat_bins
        self.stat_bin_type = stat_bin_type
        self._make_stat_bins()
        self.counts = np.zeros((len(ax1_bins)-1, len(ax2_bins)-1) + (n_stat_bins-1,)*self.n_stats)
        self.marginals = np.zeros((len(ax1_bins)-1, len(ax2_bins)-1, self.n_stats, self.n_stat_bins-1))
    
    def _make_stat_bins(self):
        self.stat_bins = []
        if self.stat_bin_type == 'linear':
            for i in range(self.n_stats):
                bounds = self.features.feat_bounds[i]
                self.stat_bins.append(np.linspace(bounds[0], bounds[1], self.n_stat_bins))
    
    def _compute_stats_at_location(self, points):
        ## Return the statistics of the plume at the given points
        ## Points is a list of tuples (pixel_x, pixel_y)
        ## Return numpy array of features computed by OdorFeatures class
        stats = []
        rng = np.random.default_rng(seed=9)
        for point in points:
            ## Pick a random plume frame
            self.plume.pick_random_frame()
            ## Compute the statistics at the given point
            for i in range(2): ## long enough to contain all the history dependence required
                self.plume.advance(rng)
                feats = self.features.update(np.pi, point, self.plume.frame)
            stats.append(feats)
        return np.array(stats)

    def _populate_counts(self):
        for i in range(len(self.ax1_bins)-1):
            for j in range(len(self.ax2_bins)-1):
                ## Generate a set of points in the bin
                print(i,j)
                points = self._generate_points_in_bin(i, j)
                ## Normalize the counts by the number of points
                stats = self._compute_stats_at_location(points)
                self.counts[i, j], self.marginals[i, j] = self._bin_stats(stats)
    
    def _generate_points_in_bin(self, i, j):
        ## Generate a set of points in the given bin; np array of dim (n_points, 2)
        ## TODO: check that this works
        ax1_samples = np.random.uniform(low=self.ax1_bins[i], high=self.ax1_bins[i+1], size=(self.n_points, 1))
        ax2_samples = np.random.uniform(low=self.ax2_bins[j], high=self.ax2_bins[j+1], size=(self.n_points, 1))
        if self.axes == 'polar':
            x_samples = ax1_samples * np.cos(ax2_samples) + self.plume.source_location[0]
            y_samples = ax1_samples * np.sin(ax2_samples) + self.plume.source_location[1]
        elif self.axes == 'cartesian':
            x_samples = ax1_samples + self.plume.source_location[0]
            y_samples = ax2_samples + self.plume.source_location[1]
        return np.hstack((x_samples, y_samples))

    def _bin_stats(self, stats):
        ## Bin the given statistics into the stat bins
        joint = np.histogramdd(stats, bins=self.stat_bins)[0]/self.n_points
        marginal = np.array([np.sum(joint, axis=tuple(dim for dim in range(joint.ndim) if dim != i)) for i in range(self.n_stats)])
        return joint, marginal

    def _plot_histograms(self, plot_type='pdf'):
        ## We make a n_features sets of plots, each of which are ax1_bins x ax2_bins histograms
        ## Now, we plot the histograms
        if plot_type == 'pdf':
            for i in range(self.n_stats):
                if self.axes == 'polar':
                    # Create a figure with polar projection
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, polar=True)
                    # Remove radial and angular tick labels
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    # Calculate subplot size and spacing
                    subplot_width = 0.2
                    subplot_height = 0.15
                    r_avg, theta_avg = (self.ax1_bins[:-1]+self.ax1_bins[1:])/2, (self.ax2_bins[:-1]+self.ax2_bins[1:])/2
                    r_norm = r_avg / np.max(r_avg)
                    x_points = r_norm * np.cos(theta_avg)
                    y_points = r_norm * np.sin(theta_avg)
                    for j, r in enumerate(r_norm):
                        for k, t in enumerate(theta_avg):
                            # Create a subplot at the specified position
                            sub_ax = fig.add_axes([x_points[j] - subplot_width / 2, y_points[k] - subplot_height / 2, subplot_width, subplot_height])
                            dist = self.marginals[j, k]
                            sub_ax.bar(self.stat_bins[i][:-1], dist, width=[self.stat_bins[i][b+1]-self.stat_bins[i][b] for b in range(self.n_stat_bins-1)], align='edge')
                            sub_ax.set_title(f'Subplot {r}-{t}')
                
                elif self.axes == 'cartesian':
                    fig, axes = plt.subplots(len(self.ax1_bins)-1, len(self.ax2_bins)-1)
                    for j in range(len(self.ax1_bins)-1):
                        for k in range(len(self.ax2_bins)-1):
                            dist = self.marginals[j, k]
                            axes[j, k].bar(self.stat_bins[i][:-1], dist, width=[self.stat_bins[i][b+1]-self.stat_bins[i][b] for b in range(self.n_stat_bins-1)], align='edge')
                            axes[j, k].set_title("({}, {})".format(j, k))
                plt.savefig("histogram_{}.png".format(i))

        elif plot_type == 'heatmap':
            ## Plot the heatmap of the marginal distribution averages
            marg_avg = np.mean(self.marginals, axis=3)
            fig, axes = plt.subplots((1, self.n_stats))
            for i in range(self.n_stats):
                axes[i].imshow(marg_avg[:, :, i], cmap='hot')
                axes[i].set_title("{}".format(i))
            plt.savefig("heatmap.png")

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
    def plot(self, ptype='histogram'):
        self._populate_counts()
        self._plot_histograms(plot_type=ptype)