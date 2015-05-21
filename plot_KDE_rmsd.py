#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def kde_sklearn(inputfilename, bandwidth):
    rmsd = np.loadtxt(inputfilename)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(rmsd[:, np.newaxis])
    xgrid = np.linspace(rmsd.min(), rmsd.max(), 100)
    log_density = kde.score_samples(xgrid[:, np.newaxis])
    return (np.exp(log_density), xgrid)

(density_wt, xgrid_wt) = kde_sklearn('../Mod/test11/merge3us/rmsd_after15000_Angstrom.xvg', 0.1)
(density_mut, xgrid_mut) = kde_sklearn('../Mutants/M4_GluN1/test1/merge3us/rmsd_after15000_Angstrom.xvg', 0.1)
print 'Max density %8.3f in WT at: %8.3f' % (density_wt.max(), xgrid_wt[np.argmax(density_wt)])
print 'Max density %8.3f in mutant at: %8.3f' % (density_mut.max(), xgrid_mut[np.argmax(density_mut)])
plt.plot(xgrid_wt, density_wt, c='red', linewidth=2, label="GluA1", zorder=10)
plt.plot(xgrid_mut, density_mut, c='blue', linewidth=2.5, linestyle='--', label="Swapped M4")
plt.xlabel(r'RMSD ($\AA$)', fontsize=20)
plt.ylabel('Distribution', fontsize=20)
lg = plt.legend(prop={"size":15})
lg.draw_frame(False)
#plt.savefig('plot_KDE_rmsd.png', dpi=300)
plt.show()
