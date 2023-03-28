import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib

font = {'size': 18}
matplotlib.rc('font', **font)
show = True


firstbin = 1
fname = 'L43_S18_ksz_vij_iz1.dat'
plt.figure(figsize=[8, 4.5])


tau = 0.54e-4
T_cmb_over_c = 2.726e6/3e5
df = pd.read_csv(fname, delim_whitespace=True,
                     names=['id1', 'id2', 'r', 'p'])
r = df.r.values[firstbin:]
p = df.p.values[firstbin:]
plt.plot(r, -p * T_cmb_over_c * tau,
         ls='-', label=fname, marker='o')

plt.ylim([-0.1, 0])
plt.legend(fontsize=10)
if show:
    plt.show()
else:
    plt.savefig('theory_curves.png', dpi=120)
