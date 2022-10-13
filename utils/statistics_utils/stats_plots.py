import numpy  as np
import pandas as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt

def plot_cloud_ener_distr(df,
                        xrange,
                        fignum,
                        label = ' ',
                        title = ' ',
                        xlabel = 'Energy (MeV)',
                        nbins = 100,
                        figsize = (10, 7),
                        alpha = 0.5,
                        fill = True,
                        lw = 2,
                        histtype = 'step',
                        color = 'tab:blue'):
    plt.figure(fignum, figsize = figsize)
    data = df.cloud_ener
    nevents = len(data)

    label = label + ' cloud \nentries {}'.format(nevents)

    plt.hist(data, bins = nbins, range = xrange, weights=np.ones(nevents) / nevents,
             histtype = histtype, alpha = alpha, fill = fill, linewidth = lw, color = color,
             label = label)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(prop={'size': 12})

def histogram_statistics(full_df,
                         var,
                         xlabel = None,
                         binclass = (0, 1),
                         mean = True,
                         discrete_var = False,
                         print_statistics = False,
                         names = ['e$^{-}$', 'e$^{-}$ e$^{+}$']):
    '''
    discrete_var is used for discrete variables just to have the perfect number of bins
    '''
    plot_df = full_df[['dataset_id', 'binclass', var]].drop_duplicates()
    if plot_df[var].dtype == bool:
        plot_df[var] = plot_df[var].astype(int)
    plot_df[var] = plot_df[var].fillna(-1)
    plot_values = plot_df[var].unique()

    nbins = len(plot_values[~np.isnan(plot_values)])
    xrange = (plot_df[var].min(), plot_df[var].max() + 1)
    if discrete_var:
        nbins  = int(xrange[1] - xrange[0])

    if nbins > 100:
        nbins = 50

    class_stats = []
    for i in binclass:
        df = plot_df[plot_df.binclass == i]
        if df.empty:
            class_stats.append([np.nan] * 2)
            continue

        info = '\nentries {:.0f}'.format(float(len(df)))
        if mean == True:
            info = info + '\nmean {:.3f} \nstd {:.3f}'.format(np.mean(df[var]), np.std(df[var]))
        hist_info = plt.hist(df[var], nbins, xrange, label = names[i] + info,
                             histtype = 'step', density = True, alpha = 0.5, fill = True, linewidth = 2)
        if print_statistics:
            print(names[i], hist_info[0] * 100)
        class_stats.append(hist_info[0])

    plt.legend()
    if xlabel == None:
        xlabel = var
    plt.xlabel(xlabel)
    plt.grid()
    tot = np.histogram(plot_df[var], bins = nbins, range = xrange, density = True)[0] * 100
    if print_statistics:
        print('total', tot)
    return class_stats, tot, xrange

def plot_secondary_clouds_elements(orig_df,
                                title = ' ',
                                xlabel = 'Segmentation class',
                                var = 'segclass',
                                names = ['e$^{-}$', 'e$^{-}$ e$^{+}$'],
                                binclass = (0, 1),
                                figsize = (10, 7),
                                alpha = 0.5,
                                density = True,
                                fill = True,
                                lw = 2,
                                histtype = 'step'):

    plt.figure(figsize = figsize)

    plot_df = orig_df.copy()
    plot_values = plot_df[var].unique()

    xrange = (plot_df[var].min(), plot_df[var].max() + 1)
    nbins  = int(xrange[1] - xrange[0])

    if nbins > 100:
        nbins = 50


    for i in binclass:
        df = plot_df[plot_df.binclass == i]
        if df.empty:
            continue

        info = '\nentries {:.0f}'.format(float(len(df)))

        hist_info = plt.hist(df[var], nbins, xrange, label = names[i] + info,
                             histtype = histtype, density = density, alpha = alpha, fill = fill, linewidth = lw)

        print(names[i], hist_info[0] * 100)

    tot = np.histogram(plot_df[var], nbins, xrange, density = True)[0] * 100
    print('total', tot)

    plt.legend()
    if xlabel == None:
        xlabel = var
    plt.title(title)
    plt.xticks(np.arange(1.5, 5.5, 1), ['other', 'track', 'blob', 'ghost'])
    plt.xlabel(xlabel)
    plt.grid()
    plt.show()
    return tot
