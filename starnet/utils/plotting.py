import numpy as np
import matplotlib.pyplot as plt
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
                           xlims=[0, 22], ylims=[-250, 250],
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
        axes[i].set_xlim(xlims[i][0], xlims[i][1])
        axes[i].set_ylim(ylims[i][0], ylims[i][1])
        axes[i].set_xticklabels(bin_centers[i])
        axes[i].set_xticks([j * 3 + 1.5 for j in range(len(bins))])
        axes[i].axhline(y=0.0, color='black', linestyle='--', alpha=0.4)
        # axes[i].set_xlabel('ASPCAP', fontsize=12)
        # axes[i].set_ylabel('', fontsize=12)

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