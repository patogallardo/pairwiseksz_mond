import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
show = False

fnames = ["Ross_2016_COMBINEDDR12_zbin1_correlation_function_monopole_post_recon_bincent0.dat"]

j = 0
fname = fnames[j]
directory = "Ross_2016_COMBINEDDR12"

df = pd.read_csv(directory + '/' + fname, delim_whitespace=True, skiprows=3,
                 names=["R_ov_h", "xi", "err_xi"])
# visualize
plt.figure(figsize=[8, 4])
plt.scatter(df.R_ov_h, df.R_ov_h**2*df.xi, 
            label='$\\xi(r) \\times r^2$')
plt.xlabel('r [Mpc/h]')
plt.ylabel('$\\xi r^2$')
if show:
    plt.show()
else:
    plt.savefig('plots/%s.pdf' %fname)
plt.close()
# end visualize

xi_interp = interp1d(df.R_ov_h, df.xi, kind='quadratic')

def xi_r_sq(r):
    r_sq = r**2
    return (xi_interp(r)) * r_sq

r = np.linspace(5, 150, 1000)

I = np.zeros(len(r))
for j in range(len(r)):
    I_j = quad(xi_r_sq, 5, r[j])[0]
    I[j] = I_j

# visualize integral
plt.figure(figsize=[8, 4])
plt.plot(r, I)

plt.xlabel('r [Mpc/h]')
plt.ylabel("$\\int \\xi(r) r^2 dr$")

if show:
    plt.show()
else:
    plt.savefig("plots/integral_%s.pdf" % fname)
plt.close()
# end visualize integral

# visualize integral/r^2
plt.figure(figsize=[8, 4])
plt.plot(r, I/r**2, label='$g$')
plt.plot(r, np.sqrt(I/r**2), label='$\\sqrt{g}$')

plt.text(100, 3.5, "$g=\\frac{1}{r^2}\\int \\xi(r) r^2 dr}$")

plt.legend()
plt.xlabel("r[Mpc/h]")
plt.ylabel("")

if show:
    plt.show()
else:
    plt.savefig("plots/g_%s.pdf" % fname)
plt.close()
# end viz

