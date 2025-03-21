# config file for visualization of dynamical systems model fits (dynamics_summarize.py workflow)

# specification of plot limits for phase plane plots and bins for histogram plots
pplane_xlim = [-4,4]
bin_xlim = [-5,5]

pplane_ylim = [-3.5,2.5]
bin_ylim = [-4,3]

# fix bins and centers for all datasets using bin limits defined above
Nbins_plot = [50 for i in range(2)]
bin_limits = [bin_xlim,bin_ylim]

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'truncate_p':[True,[0,Nbins_plot[0]-0],[0,Nbins_plot[1]-0]]}

