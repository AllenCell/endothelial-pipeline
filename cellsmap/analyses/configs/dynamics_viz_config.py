# config file for visualization of dynamical systems model fits (dynamics_summarize.py workflow)
import numpy as np
from cellsmap.analyses.configs.manifest_postproc_config import PCs

# specification of plot limits for phase plane plots and bins for histogram plots
pplane_xlim = [-4,4]
bin_xlim = [-5,5]

pplane_ylim = [-3.5,2.5]
bin_ylim = [-4,3.5]

# fix bins and centers for all datasets using bin limits defined above
ndim = len(PCs)
Nbins_plot = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]

pplane_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel':'PC'+str(PCs[0]+1),'plt_ylabel':'PC'+str(PCs[1]+1),
            'truncate_p':[True,[0,Nbins_plot[0]-0],[0,Nbins_plot[1]-0]]}

shear_range = np.linspace(4,30,60) # range of shear stresses to consider when computing epr, fixed points, etc.

# plotting args for summary plot of fixed points by shear stress
fpt_args = {'plt_lims':[[pplane_xlim[0],pplane_xlim[1]],[pplane_ylim[0],pplane_ylim[1]]],
            'plt_xlabel':'Shear stress (dyn/cm$^2$)',
            'plt_ylabel':['PC'+str(PCs[0]+1)+'$^*$','PC'+str(PCs[1]+1)+'$^*$'],
            'plt_title':'Fixed points by shear stress'}

# plotting args for plotting generalized potential energy landscape
Nbins_gp = [60 for i in range(ndim)]
shear_range_gp = np.linspace(5,30,15) # range of shear stresses to consider when plotting generalized potential energy landscape

# plotting args for plotting generalized potential energy landscape
gp_args = {'downsample':10, 'normed':False, 'p_tol':1e-8, 'cmap':'jet',
           'plt_xlabel':'PC'+str(PCs[0]+1),'plt_ylabel':'PC'+str(PCs[1]+1),
           'plt_title':'Generalized potential energy landscape'}