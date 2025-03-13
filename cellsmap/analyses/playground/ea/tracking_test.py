# %%
import numpy as np

import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd
from scipy.optimize import curve_fit

import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane

import cellsmap.util.io as io
import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis

# %%

path_to_data = '//allen/aics/assay-dev/users/Serge/cellsmap_out/test_tracking_output_exploration/filtered_tracking_results.tsv'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/tracking_test/'

eaio.make_savedir(savedir,subfolders=False)

# %%
df = eaio.load_array(path_to_data)
df.head()


# %%

min_length = 50
df_long = df.loc[df['track_duration'] > min_length].copy()
df_long.sort_values(by=['track_id','dataset_name','T','track_duration'],inplace=True)
df_long.head()

# %%
track_pairs = df_long[['track_id','dataset_name']].values
unique_track_pairs = []
for j in range(track_pairs.shape[0]):
    tup = tuple(track_pairs[j])
    if tup not in unique_track_pairs:
        unique_track_pairs.append(tup)
unique_track_pairs[0]
# %%
track_ids = np.unique(df_long['track_id'].values)
print('Number of unique tracks longer than', min_length, 'timepoints : ', len(unique_track_pairs))
track_durations = np.sort(np.unique(df_long['track_duration'].values)) # sorted low to high
print('Maximum track duration: ', track_durations.max())
print('Minimum track duration: ', track_durations.min())

dataset_names = list(set(df_long['dataset_name'].values))
print("Available datasets: ")
print(dataset_names)

# %%
ds_ID = 3
mv_name = dataset_names[ds_ID]
df_ = df_long[df_long['dataset_name'] == mv_name].copy()
track_ids = np.unique(df_['track_id'].values)
print('Number of tracks in dataset', mv_name, ': ', len(track_ids))
# %%
track_idx = 0
df__ = df_[df_['track_id'] == track_ids[track_idx]].copy()
print('Track ID: ', track_ids[track_idx])
print('Track duration: ', df__['track_duration'].values[0])
# %%
data_config = io.get_dataset_info(mv_name)
first_flow = float(data_config['flow'][0][-1])

change_frame = 0
flow_list = [first_flow]
if len(data_config['flow']) > 1:
    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    second_flow = float(data_config['flow'][1][-1])
    flow_list.append(second_flow)

print(flow_list, change_frame)
if df__['T'].values.max() < change_frame:
    print('Track under ', flow_list[0], 'dyn/cm^2 shear stress')
elif df__['T'].values.min() > change_frame and len(flow_list) == 1:
    print('Track under ', flow_list[0], 'dyn/cm^2 shear stress')
elif df__['T'].values.min() > change_frame and len(flow_list) > 1:
    print('Track under ', flow_list[1], 'dyn/cm^2 shear stress')
else:
    print('Track spans both shear stress conditions')
# %%
ctr_x = [eval(x)[0] for x in df__['centroid'].values]
ctr_y = [eval(x)[1] for x in df__['centroid'].values]

plt.plot(ctr_x, ctr_y)
plt.title('Centroid')
# %%
print(df__.keys())
key_name = 'orientation'
plt.plot(df__['T'].values, df__[key_name].values,'k')
if change_frame >= df__['T'].values.min() and change_frame <= df__['T'].values.max():
    plt.vlines(change_frame, df__[key_name].values.min()-0.1, 
            df__[key_name].values.max()+0.1, 
            colors='r', linestyles='dashed')
plt.title(key_name)

# %%
X = df__[key_name].values
T = df__['T'].values
dX = X[1:] - X[:-1]
dT = T[1:] - T[:-1]
dXdT = dX/dT
plt.plot(T[1:][dT<2], dXdT[dT<2],'k')
if change_frame >= df__['T'].values.min() and change_frame <= df__['T'].values.max():
    plt.vlines(change_frame, dXdT.min()-0.1, 
           dXdT.max()+0.1, 
           colors='r', linestyles='dashed')

# %%
key_name = 'orientation'
n_traj = len(track_ids)

Nbins=7
bins = np.linspace(0,np.pi/2,Nbins+1)
centers = 0.5*(bins[1:] + bins[:-1])
dT_thresh = 1

for ds_ID, mv_name in enumerate(dataset_names):
    print('Analyzing ', mv_name)
    df_ = df_long[df_long['dataset_name'] == mv_name].copy()
    track_ids = np.unique(df_['track_id'].values)
      
    data_config = io.get_dataset_info(mv_name)
    first_flow = float(data_config['flow'][0][-1])

    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    flow_list = [first_flow]
    if len(data_config['flow']) > 1:
        second_flow = float(data_config['flow'][1][-1])
        flow_list.append(second_flow)
    max_T = int(data_config['flow'][-1][1]*60/5)

    for flow_idx, flow in enumerate(flow_list):
        print('Analyzing ', key_name, ' for shear stress =', flow, ' dyn/cm^2')
        print("\n")

        if flow_idx == 0:
            t_vec = np.linspace(0,min(change_frame,max_T),min(change_frame,max_T)+1)

        else:
            t_vec = np.linspace(change_frame,max_T,max_T-change_frame+1)

        p_t = np.zeros((len(t_vec),Nbins)) # density of trajectories over states at each time point

        f_KM = np.nan*np.ones((Nbins,len(track_ids)))
        D_KM = np.nan*np.ones(f_KM.shape)
        f_err = np.nan*np.ones(f_KM.shape)
        D_err = np.nan*np.ones(f_KM.shape)

        for idx,track_id in enumerate(track_ids):
            df__ = df_[df_['track_id'] == track_id].copy()
            X = df__[key_name].values
            T = df__['T'].values

            if flow_idx == 0 and T.min() >= change_frame and len(flow_list) > 1:
                #print("Skipping track ",track_id,", under second shear stress condition",sep='')
                skip = True
                continue
            elif flow_idx == 1 and T.max() <= change_frame and len(flow_list) > 1:
                #print("Skipping track ",track_id,", under first shear stress condition",sep='')
                continue
            elif T.min() < change_frame and T.max() > change_frame:
                #print("Clipping track ",track_id,", spans both shear stress conditions",sep='')
                if flow_idx == 0:
                    X = X[T<=change_frame]
                    T = T[T<=change_frame]
                elif flow_idx == 1:
                    X = X[T>=change_frame]
                    T = T[T>=change_frame]
            
            if flow_idx == 0:
                # density of this trajectory over states at each time point
                for ii, t in enumerate(T):
                    jj = np.digitize(X[ii],bins)
                    p_t[t,jj-1] += 1
            else:
                # density of this trajectory over states at each time point
                for ii, t in enumerate(T):
                    jj = np.digitize(X[ii],bins)
                    p_t[t-change_frame,jj-1] += 1

            dX = X[1:] - X[:-1]
            dT = T[1:] - T[:-1]
            dXdT = dX/dT
            dXdT = dXdT[dT<=dT_thresh]
            dX2dT = dX**2/dT
            dX2dT = dX2dT[dT<=dT_thresh]
            X_T = X[:-1][dT<=dT_thresh]

            id_list = np.digitize(X_T,bins)
            uids = np.unique(id_list) # unique bin ids
            if any([Nbins+1 in id_list]):
                raise ValueError('Data point outside of histogram bins. Please update bounds.')

            for uid in uids:
                my_cond = id_list==uid
                mask = np.where(my_cond)[0]
                # At each histogram bin, find time series points where the state falls into this bin
                f_KM[uid-1,idx] = np.mean(dXdT[mask],axis=0) # Conditional average  ~ drift
                D_KM[uid-1,idx] = 0.5*np.mean(dX2dT[mask],axis=0) # Conditional variance  ~ diffusion

                # Estimate error by variance of samples in the bin
                if len(mask) > 1:
                    #inTrajVariation = True
                    f_err[uid-1,idx] = np.nanstd(dXdT[mask],axis=0)/np.sqrt(len(mask))
                    D_err[uid-1,idx] = np.nanstd(dX2dT[mask],axis=0)/np.sqrt(len(mask))

        # normalize p_T
        p_t = np.nan_to_num(p_t,nan=0.0)
        p_t = p_t/np.sum(p_t,axis=-1)[:,np.newaxis]

        fig,ax = plt.subplots()
        cax = ax.pcolormesh(p_t.T,cmap='viridis')
        # set xtick labels to be the time points
        ax.set_xticks(np.arange(0,len(t_vec),50))
        ax.set_xticklabels(np.arange(t_vec.min(),t_vec.max(),50).astype(int))
        # set ytick labels to be the centers of every other bin
        ax.set_yticks(np.arange(0.5,Nbins,2))
        ax.set_yticklabels(np.round(centers[::2],1))
        ax.set_xlabel('Time')
        ax.set_ylabel(key_name)
        fig.colorbar(cax,label='Hist')
        plt.show()

        # compute mean of histogram at each time point (~ mean trajectory)
        X_t = np.zeros_like(t_vec)
        for i in range(len(t_vec)):
            for j in range(Nbins):
                X_t[i] += p_t[i,j] * centers[j]
            X_t[i] = X_t[i] / np.sum(p_t[i,:])
        X_t = np.nan_to_num(X_t,nan=0.0)

        fit_cond = (flow_idx == 1) or (flow_idx == 0 and flow<10)

        if fit_cond:
            def f_(x,a,b,c):
                return b*np.exp(a*x) + c

            # Here you give the initial parameters for a,b,c which Python then iterates over
            # to find the best fit
            if flow < 10:
                p0 = (0.0,1.0,1.0)
            else:
                p0 = (0.0,-1.0,1.0)
            
            if flow_idx == 0:
                popt, pcov = curve_fit(f_,t_vec[70:],X_t[70:],p0=p0)
            else:
                popt, pcov = curve_fit(f_,t_vec,X_t,p0=p0)

            print(popt) # This contains your three best fit parameters

            p1 = popt[0] # This is your a
            p2 = popt[1] # This is your b
            p3 = popt[2] # This is your c

            print("Relaxation timescale:",5*(-1/p1)/60,"hours")

            curvey = f_(t_vec,p1,p2,p3) # This is your y axis fit-line
            residuals = X_t - curvey
            fres = sum( (residuals**2)/curvey ) # The chi-sqaure of your fit

            print("Residual sum of squares:",fres)

        SD = np.zeros_like(t_vec)
        for i in range(len(t_vec)):
            for j in range(Nbins):
                SD[i] += p_t[i,j] * (centers[j] - X_t[i])**2
            SD[i] = np.sqrt(SD[i]/np.sum(p_t[i,:]))

        fig,ax = plt.subplots()
        ax.plot(t_vec,X_t,'k',linewidth=2.5)
        if fit_cond:
            ax.plot(t_vec, curvey, 'r--',linewidth=1.5)
        ax.fill_between(t_vec,X_t-SD,X_t+SD,color='k',alpha=0.2)
        ax.set_ylabel(key_name)
        ax.set_xlabel('Time')
        ax.set_xlim(t_vec.min(),t_vec.max())
        ax.set_ylim(0,np.pi/2)
        ax.set_title('Shear stress = '+str(flow_list[flow_idx])+' dyn/cm^2')
        plt.show()

        f_KM_avg = np.nanmean(f_KM,axis=-1)
        D_KM_avg = np.nanmean(D_KM,axis=-1)

        f_err_mean = np.nanmean(f_err,axis=-1)
        f_err_mean = np.nan_to_num(f_err_mean,nan=1e10)
        f_KM_std = np.nanstd(f_KM,axis=-1)/np.sqrt(n_traj)
        f_KM_std = np.nan_to_num(f_KM_std,nan=1e10)
        f_err = f_err_mean + f_KM_std

        D_err_mean = np.nanmean(D_err,axis=-1)
        D_err_mean = np.nan_to_num(D_err_mean,nan=1e10)
        D_KM_std = np.nanstd(D_KM,axis=-1)/np.sqrt(n_traj)
        D_KM_std = np.nan_to_num(D_KM_std,nan=1e10)
        D_err = D_err_mean + D_KM_std

        fig,ax = plt.subplots(1,2,figsize=(15,7))
        ax[0].plot(centers,f_KM_avg,'r-o')
        ax[0].fill_between(centers,f_KM_avg-f_err,f_KM_avg+f_err,color='r',alpha=0.2)
        ax[0].plot(centers,np.zeros_like(centers),'k--',alpha=0.75)
        ax[0].set_xlabel(key_name)
        ax[0].set_ylabel('Drift')
        ax[0].set_ylim(-0.2,0.15)

        ax[1].plot(centers,D_KM_avg,'b-o')
        ax[1].fill_between(centers,D_KM_avg-D_err,D_KM_avg+D_err,color='b',alpha=0.2)
        ax[1].set_xlabel(key_name)
        ax[1].set_ylabel('Diffusion')
        ax[1].set_ylim(-0.001,0.06)

        fig.suptitle('Shear stress = '+str(flow_list[flow_idx])+' dyn/cm^2',fontsize=26)
        plt.show()
# %%
