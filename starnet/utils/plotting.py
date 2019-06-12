import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from pylab import setp


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][1], color='magenta', alpha=0.4)
    setp(bp['caps'][2], color='magenta', alpha=0.4)
    setp(bp['caps'][3], color='magenta', alpha=0.4)
    setp(bp['whiskers'][2], color='magenta', alpha=0.4)
    setp(bp['whiskers'][3], color='magenta', alpha=0.4)
    setp(bp['medians'][1], color='magenta', alpha=0.4)

    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    # fill with colors
    colors = ['lightblue', 'pink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)


def make_boxplots_splitSNR(targets, SNR, difference, high_SNR, low_SNR,
                           bin_centers, bin_width=[100, 0.25],
                           ylims=[-250, 250],
                           labels=[r'T$_{eff}$', r'log$g$', r'[M/H]'],
                           xlabel='', ylabel=''):
    # Statistics
    bias = np.nanmedian(difference, axis=0)
    scatter = np.nanstd(difference, axis=0)

    numPlots = targets.shape[1]
    fig, axes = plt.subplots(numPlots, 1, figsize=(8, numPlots * 1.5))

    for i in range(numPlots):
        bins = []
        for center in bin_centers[i]:
            inds1 = np.logical_and(abs(targets[:, i] - center) < bin_width[i], SNR > high_SNR)
            inds2 = np.logical_and(abs(targets[:, i] - center) < bin_width[i], SNR < low_SNR)
            bins.append([difference[:, i][inds1],
                         difference[:, i][inds2]])

        for j in range(len(bins)):
            # Find the smallest bin, get number of elements in it so we can ensure every bin has
            # the same number of elements. Then plot each boxplot.
            min_bin = min([min(len(binnn[0]), len(binnn[1])) for binnn in bins])
            np.random.shuffle(bins[j][0])
            np.random.shuffle(bins[j][1])
            new_bins1 = bins[j][0][:min_bin]
            new_bins2 = bins[j][1][:min_bin]
            bp = axes[i].boxplot([new_bins1, new_bins2], showfliers=False,
                                 positions=[3 * j + 1.5, 3 * j + 1.5], widths=[2.4, 2.6],
                                 patch_artist=True,
                                 notch=True)
            setBoxColors(bp)

        # set axes limits and labels
        left, right = axes[i].get_xlim()
        axes[i].set_xlim(0, right+1)
        axes[i].set_ylim(ylims[i][0], ylims[i][1])
        axes[i].set_xticklabels(bin_centers[i])
        axes[i].set_xticks([j * 3 + 1.5 for j in range(len(bins))])
        axes[i].axhline(y=0.0, color='black', linestyle='--', alpha=0.4)

        # Annotate median and std of residuals
        bbox_props = dict(boxstyle="square,pad=0.4", fc="w", ec="k", lw=1)
        axes[i].annotate(
            r'\textbf{{{}}}:  '.format(labels[i]) + '$\widetilde{{m}}=${0:6.2f}$;\ s =${1:6.2f}'.format(bias[i],
                                                                                                        scatter[i],
                                                                                                        width=6),
            xy=(0.05, 0.12), xycoords='axes fraction', fontsize=10, bbox=bbox_props)

        # annotate plots
        # bbox_props = dict(boxstyle="square,pad=0.4", fc="w", ec="k", lw=1)
        # annotation = labels[i]
        # axes[i].annotate(annotation, xy=(0.05, 0.12), xycoords='axes fraction', fontsize=9, bbox=bbox_props)

        axes[i].grid(False)

        # draw temporary red and blue lines and use them to create a legend
        hB, = axes[i].plot([0, 0], 'b-')
        hR, = axes[i].plot([0, 0], 'm-')
        if i != numPlots - 1:
            hB.set_visible(False)
            hR.set_visible(False)

    # Set x and y figure labels
    fig.text(0.02, 0.5, ylabel, ha='center', va='center',
             rotation='vertical', fontsize=15)
    fig.text(0.5, 0.01, xlabel, ha='center', va='center',
             fontsize=15)

    # Set figure legend
    fig.legend((hB, hR), (r'S/N $>$ %s' % high_SNR, r'S/N $<$ %s' % low_SNR), 'upper center', ncol=2,
               framealpha=1.0)

    fig.subplots_adjust(wspace=.01, hspace=.3)
    fig.subplots_adjust(right=0.9, left=0.095, bottom=0.06, top=0.95)

    # fig.tight_layout()
    plt.show()


def make_boxplots_two_datasets(xdata, ydata1, ydata2,
                               bin_centers, bin_width=[100, 0.25],
                               xlims=[0, 22], ylims=[-250, 250],
                               labels=[r'T$_{eff}$', r'log$g$', r'[M/H]'],
                               xlabel='',
                               ylabel='',
                               legend=('', '')):
    # Statistics
    # bias = np.nanmedian(difference, axis=0)
    # scatter = np.nanstd(difference, axis=0)

    numPlots = ydata1.shape[1]
    fig, axes = plt.subplots(numPlots / 2, 2, figsize=(14, numPlots * 1.2))
    axes = axes.ravel()

    for i in range(numPlots):
        bins = []
        for center in bin_centers[i]:
            inds1 = np.array(abs(xdata[:, i] - center) < bin_width[i])
            inds2 = np.array(abs(xdata[:, i] - center) < bin_width[i])
            bins.append([ydata1[:, i][inds1],
                         ydata2[:, i][inds2]])

        for j in range(len(bins)):
            # Find the smallest bin, get number of elements in it so we can ensure every bin has
            # the same number of elements. Then plot each boxplot.
            min_bin = min([min(len(binnn[0]), len(binnn[1])) for binnn in bins])
            np.random.shuffle(bins[j][0])
            np.random.shuffle(bins[j][1])
            new_bins1 = bins[j][0][:min_bin]
            new_bins2 = bins[j][1][:min_bin]
            bp = axes[i].boxplot([new_bins1, new_bins2], showfliers=False,
                                 positions=[3 * j + 1.5, 3 * j + 1.5], widths=[2.4, 2.6],
                                 patch_artist=True,
                                 notch=True,
                                 zorder=1)
            setBoxColors(bp)

        # set axes limits and labels
        axes[i].set_xlim(xlims[i][0], xlims[i][1])
        axes[i].set_ylim(ylims[i][0], ylims[i][1])
        axes[i].set_xticklabels(bin_centers[i])
        axes[i].set_xticks([j * 3 + 1.5 for j in range(len(bins))])
        axes[i].xaxis.set_tick_params(labelsize=13)
        if i == 0:
            for label in axes[i].xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        axes[i].yaxis.set_tick_params(labelsize=13)
        axes[i].axhline(y=0.0, color='black', linestyle='--', alpha=1.0, lw=3, zorder=2)
        # plt.axhline(y=0, color='r', linestyle=':')

        # Annotate median and std of residuals
        # bbox_props = dict(boxstyle="square,pad=0.4", fc="w", ec="k", lw=1)
        # axes[i].annotate(r'\textbf{{{}}}:  '.format(labels[i]) + '$\widetilde{{m}}=${0:6.1f}$;\ s =${1:6.1f}'.format(bias[i],scatter[i],width=6),
        #                xy=(0.05, 0.12), xycoords='axes fraction', fontsize=10, bbox=bbox_props)

        # annotate plots
        bbox_props = dict(boxstyle="square,pad=0.4", fc="w", ec="k", lw=1)
        annotation = labels[i]
        axes[i].annotate(annotation, xy=(0.05, 0.12), xycoords='axes fraction', fontsize=15, bbox=bbox_props)

        axes[i].grid(False)

        # draw temporary red and blue lines and use them to create a legend
        hB, = axes[i].plot([0, 0], 'b-')
        hR, = axes[i].plot([0, 0], 'm-')
        if i != numPlots - 1:
            hB.set_visible(False)
            hR.set_visible(False)

    # Set x and y figure labels
    fig.text(0.02, 0.5, ylabel, ha='center', va='center',
             rotation='vertical', fontsize=20)
    fig.text(0.48, 0.01, xlabel, ha='center', va='center',
             fontsize=20)

    # Set figure legend
    fig.legend((hB, hR), legend, 'upper center', ncol=2,
               framealpha=1.0, fontsize=15)

    fig.subplots_adjust(wspace=.15, hspace=.2)
    fig.subplots_adjust(right=0.9, left=0.065, bottom=0.08, top=0.92)

    # fig.tight_layout()
    plt.show()


def plot_compare_estimates_gaiaeso_resid(x_data, y_data, snr, savename=None, x_lab='', y_lab='', snr_max=200, cmap='Blues',
                                         label_names=[],
                                         lims=[[3000., 6300.], [-.3, 5.2], [-3.5, 1.],
                                               [-0.4, 0.6], [-1., 1.], [-1., 1.]],
                                         resid_lims=[[-1000., 1000.], [-2, 2], [-1, 1],
                                                     [-1., 1.], [-1., 1.], [-1., 1.]],
                                         indx=3,
                                         categories=None,
                                         groups_wanted=None,
                                         x_tick_steps=None,
                                         grid_lims=None):
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'
    plt.rcParams["text.usetex"] = True
    markers = ('o', '*', 'D', 'X', 's', 'h', 'v', '^', '<', '>', '8', 'p', 'H', 'd', 'P')
    colormaps = ('Reds', 'Greens', 'Oranges', 'Blues', 'Purples', 'Greys', 'Reds', 'Greens',
                 'Oranges', 'Blues', 'Purples', 'Greys', 'Reds', 'Greens', 'Oranges', 'Blues', 'Purples', 'Greys')

    # label names
    # label_names = ['$T_{\mathrm{eff}}$',r'$\log(g)$','$[Fe/H]$',r'[$\alpha/M$]',r'$[N/M]$',r'$[C/M]$']

    # overplot high s/n
    order = (snr).reshape(snr.shape[0], ).argsort()
    x_data = x_data[order]
    y_data = y_data[order]
    snr = snr[order]

    # Set maximum S/N
    snr[snr > snr_max] = snr_max

    # Prune data if needed
    if categories is not None:
        categories = categories[order]
        ind = np.where([categories == group for group in groups_wanted])[1]
        x_data = x_data[ind]
        y_data = y_data[ind]
        snr = snr[ind]
        categories = categories[ind]

    # Calculate residual, median residual, and std of residuals
    resid = y_data - x_data
    bias = np.nanmedian(resid, axis=0)
    scatter = np.nanstd(resid, axis=0)
    resid_a = resid[snr >= 120, :]
    resid_b = resid[snr < 100, :]

    # Plot data
    fig = plt.figure(figsize=(22, 25))
    # gs = gridspec.GridSpec(indx, 5,  width_ratios=[4., 1., 1.8, 4., 1.])
    gs = gridspec.GridSpec(indx, 3, width_ratios=[1., 0.15, 0.1])
    x_plt_indx = 0
    for i in range(y_data.shape[1]):

        # Set column index
        if i >= indx:
            x_plt_indx = 3
        ax0 = plt.subplot(gs[i % indx, x_plt_indx])
        # ax0 = axes[i%3,x_plt_indx]

        if categories is not None:
            for j, category in enumerate(groups_wanted):
                if category == b'GE_CL':
                    legend_text = 'Open Cluster field'
                elif category == b'GE_MW':
                    legend_text = 'Milky Way field'
                elif category == b'GE_MW_BL':
                    legend_text = 'Milky Way: bulge field'
                elif category == b'GE_SD_BM':
                    legend_text = 'FGKM benchmark stars'
                elif category == b'GE_SD_CR':
                    legend_text = 'CoRoT field'
                elif category == b'GE_SD_GC':
                    legend_text = 'Standard: globular clusters'
                elif category == b'GE_SD_OC':
                    legend_text = 'Standard: open clusters'
                points = ax0.scatter(x_data[categories == category][:, i],
                                     resid[categories == category][:, i],
                                     marker=markers[j],
                                     c=snr[categories == category],
                                     s=80,
                                     cmap=cmap,
                                     label=legend_text)
                # label=category.replace('_', ' '))
                handles, labels = ax0.get_legend_handles_labels()
        else:
            # Plot resid vs x coloured with snr
            points = ax0.scatter(x_data[:, i], resid[:, i], c=snr, s=20, cmap=cmap)

        if i == 0 and groups_wanted is not None:
            plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                       mode="expand", borderaxespad=0, ncol=3,
                       prop={'size': 20})
            leg = ax0.get_legend()
            for legHandle in leg.legendHandles:
                legHandle.set_color('red')

        # Set axes labels
        # ax0.set_xlabel(r'%s' % (label_names[i]), fontsize=50, labelpad=20)
        # ax0.set_ylabel(r'%s' % (label_names[i]), fontsize=50, labelpad=20)

        # Set axes limits
        ax0.tick_params(labelsize=30, width=1, length=10)
        ax0.set_xlim(lims[i])
        ax0.set_ylim(resid_lims[i])
        ax0.plot([lims[i][0], lims[i][1]], [0, 0], 'k--', lw=3)

        # Draw limits of grid
        if grid_lims is not None:
            ax0.axvline(x=grid_lims[i][0], ls='--', lw=3, c='black')
            ax0.axvline(x=grid_lims[i][1], ls='--', lw=3, c='black')

        # Annotate median and std of residuals
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=3)
        if i == 0:
            ax0.annotate(
                r'\textbf{{{}}}:  '.format(label_names[i]) + '$\widetilde{{m}}=${0:6.1f}$;\ s =${1:6.1f}'.format(
                    bias[i], scatter[i], width=6),
                xy=(0.03, 0.82), xycoords='axes fraction', fontsize=28, bbox=bbox_props)
        else:
            ax0.annotate(
                r'\textbf{{{}}}:  '.format(label_names[i]) + '$\widetilde{{m}}=${0:6.2f}$;\ s =${1:6.2f}'.format(
                    bias[i], scatter[i], width=6),
                xy=(0.03, 0.82), xycoords='axes fraction', fontsize=28, bbox=bbox_props)

        # Set axes ticks
        start, end = ax0.get_xlim()
        if x_tick_steps == None:
            stepsize = (end - start) / 4
            if i == 0:
                xticks = np.round(np.arange(start, end, stepsize)[1:], -2)
            else:
                xticks = np.round(np.arange(start, end, stepsize)[1:], 1)
        else:
            xticks = np.arange(start, end, x_tick_steps[i])[1:]
        start, end = ax0.get_ylim()
        stepsize = (end - start) / 4
        if i == 0:
            yticks = np.round(np.arange(start, end, stepsize)[1:], -2)
        else:
            yticks = np.round(np.arange(start, end, stepsize)[1:], 1)
        ax0.xaxis.set_ticks(xticks)
        ax0.yaxis.set_ticks(yticks)

        ax1 = plt.subplot(gs[i % indx, x_plt_indx + 1])

        xmin, xmax = resid_lims[i]

        y_a = resid_a[:, i][(resid_a[:, i] >= xmin) & (resid_a[:, i] <= xmax)]
        y_b = resid_b[:, i][(resid_b[:, i] >= xmin) & (resid_b[:, i] <= xmax)]

        a = sns.distplot(y_a, vertical=True, hist=False, rug=False, ax=ax1,
                         kde_kws={"color": points.cmap(200), "lw": 4})
        b = sns.distplot(y_b, vertical=True, hist=False, rug=False, ax=ax1,
                         kde_kws={"color": points.cmap(100), "lw": 4})

        a.set_ylim(resid_lims[i])
        b.set_ylim(resid_lims[i])

        ax1.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False, width=1, length=10)

        ax1.tick_params(
            axis='y',
            which='both',
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            labelsize=30, width=1, length=10)
        ax1.xaxis.set_ticks([])
        ax1.yaxis.set_ticks(yticks)

    # Create colorbar
    cbar_ax = fig.add_axes([0.88, 0.22, 0.02, 0.6])
    fig.colorbar(points, cax=cbar_ax)
    cbar = fig.colorbar(points, cax=cbar_ax, extend='neither',
                        spacing='proportional', orientation='vertical')
    cbar.set_label(r'$S/N$', size=70)
    cbar.ax.tick_params(labelsize=40, width=1, length=10)
    start, end = int(np.round(np.min(snr), -1)), int(np.max(snr))
    stepsize = int(np.round((end - start) / 4, -1))
    tick = end
    yticks = []
    while tick > start:
        yticks = [tick] + yticks
        tick -= stepsize
    yticks = np.array(yticks, dtype=int)
    ytick_labs = np.array(yticks, dtype=str)
    ytick_labs[-1] = '$>$' + ytick_labs[-1]
    cbar.set_ticks(yticks)
    cbar_ax.set_yticklabels(ytick_labs)

    # Set x and y figure labels
    fig.text(0.06, 0.5, y_lab, ha='center', va='center',
             rotation='vertical', fontsize=40)
    fig.text(0.5, 0.04, x_lab, ha='center', va='center',
             fontsize=40)

    fig.subplots_adjust(wspace=.05, hspace=.4)
    fig.subplots_adjust(right=0.82, left=0.12, bottom=0.09)
    if savename is not None:
        plt.savefig(savename)

    plt.show()
