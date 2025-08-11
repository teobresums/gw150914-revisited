import numpy as np
import matplotlib.pyplot as plt
import h5py; import corner
import bilby

import seaborn as sns
from bilby.gw.prior import BBHPriorDict
# set colormap to colorblind

# use latex for the labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_chi_prec(m1,m2,s1,s2,tilt1,tilt2):
    """ Compute chi precessing spin parameter (for given 3-dim spin vectors)
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        s1 = primary spin megnitude [dimensionless]
        s2 = secondary spin megnitude [dimensionless]
        tilt1 = primary spin tilt [rad]
        tilt2 = secondary spin tilt [rad]
    """

    s1_perp = np.abs(s1*np.sin(tilt1))
    s2_perp = np.abs(s2*np.sin(tilt2))
    one_q   = m2/m1

    # check that m1>=m2, otherwise switch
    if one_q > 1. :
        one_q = 1./one_q
        s1_perp, s2_perp = s2_perp, s1_perp

    return np.max([s1_perp , s2_perp*one_q*(4.*one_q+3.)/(3.*one_q+4.)])

class Posterior(object):
    def __init__(self, filename):

        self.load_bilby_hdf5(filename)

    

    def load_o3a_hdf5(self, filename):
        data = h5py.File(filename, "r")
        default_keys = ['mass_ratio', 'chirp_mass', 'a_1', 'a_2', 
                        'luminosity_distance', 'iota', 'cos_theta_jn', 
                        'chi_eff', 'chi_1', 'chi_2', 'chi_p', 
                        'log_likelihood']

        for this_key in default_keys:
            try:
                self.__setattr__(this_key, data['ProdF4']['posterior_samples'][this_key][:])
            except Exception:
                continue
                #print(f"Key {this_key} not found in {filename}")

    def load_bilby_hdf5(self, filename):
        data = h5py.File(filename, "r")
        
        for this_key in data['posterior'].keys():
            try:
                self.__setattr__(this_key, data['posterior'][this_key])
            except Exception:
                continue
                #print(f"Key {this_key} not found in {filename}")
        pass

    def make_hist(self, key, color, fig=None):
        data = self.__getattribute__(key)
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0]

        ax.hist(data, density=True, histtype='step', bins=50, color=color)
        return fig
    
    def find_maxL(self):
        """
        Find the maximum likelihood sample."""
        logL = self.log_likelihood
        maxL = np.argmax(logL)
        params = {}
        for key in self.__dict__.keys():
            params[key] = self.__getattribute__(key)[maxL]
        return params

    def make_corner_plot(self, keys, limits, color, ylimits=None, fig=None, bin=50, lbl=None, plot_maxL=False):
        """
        Make a corner plot for the posterior samples.
        keys: list of keys to plot
        range: list of ranges for each key
        color: color of the plot
        fig: figure to plot on (optional)
        bin: number of bins (optional)
        lbl: label for the histogram (optional)
        """
        
        matrix = np.transpose([self.__getattribute__(key) for key in keys])
        labels = [keys_latex[key] for key in keys]
        
        fig, axes = make_corner_plot(matrix, labels, limits, color, ylimits=ylimits, fig=fig, bin=bin, lbl=lbl)

        if plot_maxL:
            maxL_pars = self.find_maxL()
            for i, key in enumerate(keys):
                if key in maxL_pars.keys():
                    ax = axes[i, i]
                    ax.axvline(maxL_pars[key], color=color, linestyle='-', linewidth=1.5)
            
            # add stars for the maxL points to 2D histograms
            for i in range(len(keys)):
                for j in range(i):
                    ax = axes[i, j]
                    if keys[i] in maxL_pars.keys() and keys[j] in maxL_pars.keys():
                        ax.plot(maxL_pars[keys[j]], maxL_pars[keys[i]], '*', color=color)

        return fig, axes
    

def make_corner_plot(matrix, labels, limits, color, ylimits=None, fig=None, bin=30, lbl=None):

    L = max(len(matrix[0]), len(np.transpose(matrix)[0]))
    N = int(min(len(matrix[0]), len(np.transpose(matrix)[0])))

    if fig == None:
        fig = cornerfig=corner.corner(matrix,
                                labels          = labels,
                                weights         = np.ones(L)*100./L,
                                bins            = bin,
                                range           = limits,
                                color           = color,
                                levels          = [.5, .9],
                                quantiles       = [.05, .95],
                                contour_kwargs  = {'colors':color,'linewidths':0.95},
                                label_kwargs    = {'size':12.},
                                hist2d_kwargs   = {'label':lbl},
                                #hist_kwargs     = {'density':True},
                                plot_datapoints = False,
                                show_titles     = False,
                                plot_density    = True,
                                smooth1d        = True,
                                smooth          = True)
    else:
        fig = cornerfig=corner.corner(matrix,
                                fig             = fig,
                                weights         = np.ones(L)*100./L,
                                labels          = labels,
                                range           = limits,
                                bins            = bin,
                                color           = color,
                                levels          = [.5, .9],
                                quantiles       = [.05, .95],
                                contour_kwargs  = {'colors':color,'linewidths':0.95},
                                label_kwargs    = {'size':12.},
                                hist2d_kwargs   = {'label':lbl},
                                #hist_kwargs     = {'density':True},
                                plot_datapoints = False,
                                show_titles     = False,
                                plot_density    = True,
                                smooth1d        = True,
                                smooth          = True)
    axes = np.array(cornerfig.axes).reshape((N,N))
    
    if(ylimits is not None):
        for i in np.arange(N):
            if ylimits[i] is None:
                continue
            ax = axes[i, i]
            ax.set_ylim((0,ylimits[i]))

    return fig, axes

def prior_from_file(fname):
    prior_dict = BBHPriorDict({})
    prior_dict.from_file(fname)
    return prior_dict

def sample_from_prior(fname, n_samples=1000):
    prior_dict = prior_from_file(fname)
    samples = prior_dict.sample(n_samples)
    
    samples['mass_1'] = samples['chirp_mass'] * (1+samples['mass_ratio'])**(1/5) / samples['mass_ratio']**(3/5)
    samples['mass_2'] = samples['mass_1'] * samples['mass_ratio']
    
    samples['chi_p'] = np.vectorize(compute_chi_prec)(samples['mass_1'], samples['mass_2'], samples['a_1'], samples['a_2'], samples['tilt_1'], samples['tilt_2'])
    
    samples['chi_eff'] = (samples['a_1'] * np.cos(samples['tilt_1']) + samples['a_2'] * np.cos(samples['tilt_2']) * samples['mass_ratio']) / (
        1 + samples['mass_ratio']
    )
    
    return samples

keys_latex = {
    'mass_ratio'         : r'$1/q$',
    'chirp_mass'         : r'$\mathcal{M} [M_{\odot}]$',
    'chi_1'              : r'$\chi_1$',
    'chi_2'              : r'$\chi_2$',
    'luminosity_distance': r'$d_L$ [Mpc]',
    'iota'               : r'$\iota$ [rad]',
    'chi_eff'            : r'$\chi_{\rm eff}$',
    'chi_p'              : r'$\chi_{\rm p}$',
    'eccentricity'       : r'$e$',
    'mean_per_ano'       : r'$\zeta$ [rad]',
    'cos_theta_jn'       : r'$\cos(\theta_{\rm JN})$',
}