'''
Plot likelihood function and show 
force law indices.

P. Gallardo, K. Pardo.
'''
import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy import optimize

# Config plot
import seaborn as sns
# set fig params
sns.set_context("paper")
sns.set_style('ticks')
sns.set_palette('colorblind')
figparams = {
    'text.latex.preamble': r'\usepackage{amsmath} \boldmath',
    'text.usetex': True,
    'axes.labelsize': 14.,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [10., 8.],
    'font.family': 'DejaVu Sans',
    'legend.fontsize': 18}
plt.rcParams.update(figparams)
cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
#matplotlib.use('Agg')
show = False
# End plot config

first_bin, last_bin = 2, 17
show = False
h = 0.6736  # planck PR3 2018 https://arxiv.org/abs/1807.06209


zbin = 2
bincent = 0
obs_name = "L43_150_REST_Z"


def xi_r_sq(r, xi_interp):
    r_sq = r**2
    return (xi_interp(r)) * r_sq

def get_max_lik_fit(r, p, cov, amp_exponent_function_tofit):
    '''Received observed r, p inv_cov and a function to fit, fits the amplitude
    and returns it.'''
    x0 = np.array([1/15, 1.2])
    opt = optimize.minimize(chisq, x0,
                            args=(r, p, amp_exponent_function_tofit, C_pw, cov),
                            method='Nelder-Mead')
    amp, exp = opt.x[0], opt.x[1]
    res = {'amp': amp,
           'exp': exp}
    return res

# Filename config
corr_fun_fname = "Ross_2016_COMBINEDDR12_zbin%i_correlation_function_monopole_post_recon_bincent%i.dat" % (
    zbin, bincent)
corr_fun_dir = "Ross_2016_COMBINEDDR12"
obsdir_150 = './DR6_res/'
obsdir_090 = './DR6_res/'
obs_fnames = {"L43_150_REST_Z": obsdir_150 + "DR6v4_simple_v1_coadd_150GHz_mond_lum_gt_4p3e10_and_zgt0p44_and_zlt0p66_step_10mpc.hdf"}
obs_fname = obs_fnames[obs_name]
# end filename config

# open data
df_corr_fun = pd.read_csv(corr_fun_dir + '/' + corr_fun_fname, delim_whitespace=True, skiprows=3,
                          names=["R_ov_h", "xi", "err_xi"])
df_corr_fun["R_Mpc"] = df_corr_fun.R_ov_h/h
df_curve = pd.read_hdf(obs_fname, 'df_ksz_err')[first_bin:last_bin]

# compute integrals
xi_interp = interp1d(df_corr_fun.R_Mpc,
                     df_corr_fun.xi,
                     kind='quadratic')
r = np.linspace(6, df_corr_fun.R_Mpc.values[-1] - 1, 50)
I = np.zeros(len(r))

for j in range(len(r)):
    I_j = quad(xi_r_sq, 6, r[j], args=(xi_interp))[0]
    I[j] = I_j
# end compute integrals

# interpolate g
# EXPONENT
pow_g = interp1d(r, np.power(I/r**2, 1/2),
                 kind='cubic',
                 bounds_error=False,
                 fill_value='extrapolate')
# end interpolation

rsep = df_curve.r_mp.values


#Prefactor and exponent
c = 2.998e+8 # meters per second
G = 6.67e-11 #(Newtons m^2/kg^2)
Omega_b = 0.0224/h**2 # Omega barion
Omega_c = 0.120/h**2 #

critical_density = 3*(100*h*3.241e-20)**2/(8*3.14*G) # kg/m^3

#rho_c = Omega_c * critical_density
rho_b = Omega_b * critical_density
#a_bar = c * np.sqrt(G * critical_density)#insert formula here
a_bar = 1e-14 #ms^-2
prefactor = 4*np.pi * G * rho_b/a_bar # in units of 1/meter
prefactor = prefactor / 3.24078e-23 # in units of 1/Mpc https://www.wolframalpha.com/input?i=1+meter+in+Mpc
#end prefactor calculations

def amp_exponent_function_tofit(amplitude, exponent,
                                rsep, I):
    f_pow_g = interp1d(r, np.power(I/r**2, exponent/2),
                       kind='cubic',
                       bounds_error=False,
                       fill_value='extrapolate')
    g_exp_power_times_A = -1.0 * amplitude * f_pow_g(rsep) * prefactor**(exponent/2)/prefactor
    return g_exp_power_times_A


# start likelihood exploration
df_cov = pd.read_hdf(obs_fname, 'df_cov')
C_pw = df_cov.values.astype(float)[first_bin:last_bin, first_bin:last_bin]


def chisq(amp_exp,
          rsep, p_pw, amp_exponent_function_tofit,
          C_pw, C_sdss=None):
    '''Receives an amplitude and exponent,
    r, p_pw, p_sdss, C_pw, C_sdss and fits 
    an amplitude.
    First implementation C_sdss=None, only use the pairwise cov matrix
    '''
    amplitude, exponent = amp_exp[0], amp_exp[1]
    p_predicted = amp_exponent_function_tofit(amplitude, exponent, rsep, I)
    delta = (p_pw - p_predicted)
    #C = combine_covs(amplitude, C_pw, C_sdss)
    C = C_pw
    C_inv = np.linalg.inv(C)
    chisq_toreturn = np.dot(np.dot(delta.T, C_inv), delta)
    return chisq_toreturn


# Plot contours
N_samples_exp = 500
N_samples_amp = 500
amp_range = [0.0, 0.05]
exp_range = [0.01, 6.00]

amps = np.linspace(amp_range[0], amp_range[1], N_samples_amp)
exps = np.linspace(exp_range[0], exp_range[1], N_samples_exp)

# need to swap the coords
exp_mat, amp_mat = np.meshgrid(exps, amps)
chisq_mat = np.zeros_like(amp_mat)

res = get_max_lik_fit(rsep, df_curve.ksz_curve.values,
                      C_pw, amp_exponent_function_tofit)

for j in range(len(exps)):
    for k in range(len(amps)):
        amp, exp = amp_mat[k, j], exp_mat[k, j]
        amp_exp = [amp, exp]
        chisq_mat[k, j] = chisq(amp_exp, rsep, df_curve.ksz_curve.values,
                                amp_exponent_function_tofit, C_pw)
Likelihood = np.exp(-chisq_mat/2)
Likelihood = Likelihood/Likelihood.max()

########### Make n-amp plot ####################
amplitude_normalization = 0.0216
plt.subplots(figsize=[4.0, 4.0],
             constrained_layout=True)
levels = np.exp(-np.arange(3, -1, -1)**2/2)
plt.contour(exp_mat, amp_mat/amplitude_normalization, Likelihood,
            levels=levels,
            colors='black')
plt.ylabel(r'$\mathrm{ \bar \tau/\bar \tau_{\Lambda CDM}}$')
plt.xlabel(r'$\mathrm{Force~Law~Index,~}n$')
plt.scatter(res['exp'], res['amp']/amplitude_normalization,
            label='ML', color='black', marker='X')
plt.axvline(2, color=cs[0], alpha=0.5, ls='dashed')
plt.text(2.01, 0.03,
         r'$\Lambda\mathrm{CDM}$',
         color=cs[0])
plt.axvline(1, color=cs[1], alpha=0.5, ls='dashed')
plt.text(1.01, 0.03,
         r'$\mathrm{MOND}$',
         color=cs[1])
plt.ylim([0, amp_range[1]/amplitude_normalization])
plt.xlim([0.80, 4.0])
#plt.yticks(np.arange(0, 2.5, 1.0))
plt.savefig('plots/contour_plot_a_bar_1e-14mssq_prefactor.pdf')
plt.close()
#################### End n-amp plot



L_n = Likelihood.sum(axis=0)
L_n = L_n/L_n.max()

####### Now compute uncertainty in n #######


cumsum = np.cumsum(L_n)
cumsum = cumsum/cumsum.max()

lowerlimit = 0.5-0.68/2
upperlimit = 0.5+0.68/2
sel = np.logical_and(cumsum>lowerlimit, cumsum<upperlimit)
sel_median = cumsum > 0.5
lower_exp = exps[sel][0]
upper_exp = exps[sel][-1]
median = exps[sel_median][0]



sigma_n = 0.5 * (upper_exp - lower_exp)
print("Marginalized Median n: %1.3f" % median)
print("sigma_n=%1.3f" % sigma_n)
dist = (median - 1.0)/sigma_n
print("(marginalized median - 1.0)/sigma= %1.3f" % dist)

# make output for latex
def convert_variable_to_latex_command(varname, value, decplaces=2):
    s = "\\newcommand{\\%s}{%1.2f}" % (varname, value)
    if decplaces == 1:
        s = "\\newcommand{\\%s}{%1.1f}" % (varname, value)
    if decplaces == 3:
        s = "\\newcommand{\\%s}{%1.3f}" % (varname, value)
    if decplaces == 0:
        s = "\\newcommand{\\%s}{%i}" % (varname, value)
    return s + "\n"
string_out = ""
string_out += convert_variable_to_latex_command("MARGMEDIAN", median, 2)
string_out += convert_variable_to_latex_command("SIGMAN", sigma_n, 2)
string_out += convert_variable_to_latex_command("NSIGMA", dist, 1)

string_out += convert_variable_to_latex_command("MARGMEDIANONEDEC", median, 1)
string_out += convert_variable_to_latex_command("SIGMANONEDEC", sigma_n, 1)
string_out += convert_variable_to_latex_command("NSIGMAONEDEC", dist, 1)


# compute the 95% interval
lowerlimit = 0.5-0.95/2
upperlimit = 0.5+0.95/2
sel = np.logical_and(cumsum>lowerlimit, cumsum<upperlimit)
lower_exp = exps[sel][0]
upper_exp = exps[sel][-1]
sigma_n95 = 0.5 * (upper_exp - lower_exp)
string_out += convert_variable_to_latex_command("NLOWERBOUNDNINETYFIVEPCT", median-sigma_n95, 2)
string_out += convert_variable_to_latex_command("NLOWERBOUNDNINETYFIVEPCTONEDEC", median-sigma_n95, 1)


print("delta95pct_n=%1.3f" % sigma_n95)
print("n> %1.3f" %(median-sigma_n95))

print(string_out)
