# %%
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import GridSplit, NormalizeIntensity
from bioio import BioImage
import bioio_ome_tiff
import cellsmap.analyses.utils.viz as viz

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 2**128

# %%
# load cdh5 small patch features (PC1/PC2 trajectories)
load_dir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/cdh5_small_patch/'
X_t = np.load(load_dir+'data/traj_array.npy')

# %%
fig,ax = viz.init_subplots(1,2,figsize=(15,5))

# plot 1st PCA mode vs time for each group of trajectories
fig, ax = viz.plot_top_PCs(X_t, np.arange(X_t.shape[1]), fig=fig, ax=ax, xlabel='frame number')

# %%
analyze_PC1 = False
if analyze_PC1:
    # find trajectories with high and low initial PC1 values
    PC=0
    hightraj = np.where(X_t[:,0,PC] > 7)[0]
    lowtraj = np.where(X_t[:,0,PC] < -7)[0]
    # select one trajectory from each group, plot trajectories
    hightraj_idx = hightraj[1]
    lowtraj_idx = lowtraj[0]
    idxs = [hightraj_idx,lowtraj_idx]
    time_pts = [0,25,75,250,400,500,550]
    print(hightraj_idx,lowtraj_idx)
else: # analyze PC2
    PC=1
    print('Traj with max near T=75: '+str(np.unravel_index(np.argmax(X_t[:,50:100,PC]),X_t[:,50:100,PC].shape)[0]))
    print('Traj with max near T=250: '+str(np.unravel_index(np.argmax(X_t[:,200:275,PC]),X_t[:,200:275,PC].shape)[0]))
    print('Traj with max near T=325: '+str(np.unravel_index(np.argmax(X_t[:,310:350,PC]),X_t[:,310:350,PC].shape)[0]))
    print('Traj with max near T=500: '+str(np.unravel_index(np.argmax(X_t[:,450:550,PC]),X_t[:,450:550,PC].shape)[0]))
    idxs = [np.unravel_index(np.argmax(X_t[:,50:100,PC]),X_t[:,50:100,PC].shape)[0],
            np.unravel_index(np.argmax(X_t[:,200:275,PC]),X_t[:,200:275,PC].shape)[0],
            np.unravel_index(np.argmax(X_t[:,310:350,PC]),X_t[:,310:350,PC].shape)[0],
            np.unravel_index(np.argmax(X_t[:,450:550,PC]),X_t[:,450:550,PC].shape)[0]]
    time_pts = [np.unravel_index(np.argmax(X_t[:,50:100,PC]),X_t[:,50:100,PC].shape)[1]+50,
                np.unravel_index(np.argmax(X_t[:,200:275,PC]),X_t[:,200:275,PC].shape)[1]+200,
                np.unravel_index(np.argmax(X_t[:,310:350,PC]),X_t[:,310:350,PC].shape)[1]+310,
                np.unravel_index(np.argmax(X_t[:,450:550,PC]),X_t[:,450:550,PC].shape)[1]+450]


# %%
fig,ax=viz.init_plot(figsize=(10,5))
# default matplotlib colors
my_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
for (i,idx) in enumerate(idxs):
    fig, ax = viz.plot_traj_1D(X_t[idx,:,PC], np.arange(X_t.shape[1]), fig=fig, ax=ax, color=my_colors[i], label='crop '+str(idx), xlabel='frame number', ylabel='PC'+str(PC+1),linewidth=1.5)


# %%
# define function to generate side by side crop analysis with trejctories along specified PC axis
def analyze_crops_on_PC(PC,idxs,time_pts,X_t,img_dir,crop_grid,plot_imgs=True):
    num_high = np.zeros((len(idxs),len(time_pts))) # count of high intensity pixels in each crop
    top_pct = np.zeros((len(idxs),len(time_pts))) # 95% percentile of intensity in each crop
    fig,ax=viz.init_plot(figsize=(10,5))
    my_colors = ['k','b']
    for (i,idx) in enumerate(idxs):
        fig, ax = viz.plot_traj_1D(X_t[idx,:,PC], np.arange(X_t.shape[1]), fig=fig, ax=ax, color=my_colors[i], label='crop '+str(idx), xlabel='frame number', ylabel='PC'+str(PC+1),linewidth=1.5)
        if plot_imgs:
            for time_pt in time_pts:
                ax.plot(time_pt, X_t[idx,time_pt,PC], 'r*', markersize=10)
                ax.set_xlim([0,X_t.shape[1]])
    plt.show()

    for (j,time_pt) in enumerate(time_pts):
        ome_tif = BioImage(img_dir+'/20240305_T01_001_TP'+str(time_pt).zfill(5)+'.ome.tif',reader=bioio_ome_tiff.Reader)
        im = ome_tif.get_image_dask_data('CYX', T=0, C=0).compute()
        im_patches = crop_grid(im)

        if plot_imgs:
            fig, ax = plt.subplots(1,2,figsize=(15,10))

        for i, idx in enumerate(idxs):
            img_crop= im_patches[idx][0]
            norm = NormalizeIntensity(img_crop.mean(),img_crop.std())
            img_crop = norm(img_crop)
            if plot_imgs:
                ax[i].imshow(img_crop,cmap='gray')
                ax[i].set_title('Crop '+str(idx)+' time point '+str(time_pt))
            num_high[i,j] = len(np.where(img_crop > 0.9)[0])
            top_pct[i,j] = np.percentile(img_crop,95)

        if plot_imgs:
            plt.show()
    return num_high, top_pct

# %%
# select time points (frame numbers) to analyze images at
path_to_imgs = '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/tiff_temp_folder'
tf = GridSplit(grid=(3, 19), size=(480,480))
if analyze_PC1:
    num_high, top_pct = analyze_crops_on_PC(PC,idxs,time_pts,X_t,path_to_imgs,tf)
else:
    num_high, top_pct1 = analyze_crops_on_PC(PC,idxs[:2],time_pts,X_t,path_to_imgs,tf)
    num_high, top_pct2 = analyze_crops_on_PC(PC,idxs[2:],time_pts,X_t,path_to_imgs,tf)
# %%
if analyze_PC1:
    fig,ax = viz.init_subplots(1,2,figsize=(15,5))
    ax[0].scatter(X_t[idxs[0],time_pts,0],num_high[0,:])
    ax[1].scatter(X_t[idxs[1],time_pts,0],num_high[1,:])
else:
    fig,ax = viz.init_plot()
    ax.scatter(X_t[idxs[0],time_pts,1],top_pct1[0,:])
    ax.scatter(X_t[idxs[1],time_pts,1],top_pct1[1,:])
    ax.scatter(X_t[idxs[2],time_pts,1],top_pct2[0,:])
    ax.scatter(X_t[idxs[3],time_pts,1],top_pct2[1,:])
# %%
# run again for both crops at all time points, don't plot images
if analyze_PC1:
    time_pts = np.arange(X_t.shape[1])
    num_high, top_pct = analyze_crops_on_PC(PC,idxs,time_pts,X_t,path_to_imgs,tf,plot_imgs=False)

    # plot fraction of high intensity pixels against PC1 for each crop at each time point, color by time
    fig,ax = viz.init_plot()
    num_pixels = 480*480
    sct = ax.scatter(X_t[idxs[0],time_pts,0],num_high[0,:]/num_pixels,label='crop '+str(hightraj_idx),c=np.arange(len(time_pts)))
    sct = ax.scatter(X_t[idxs[1],time_pts,0],num_high[1,:]/num_pixels,label='crop '+str(lowtraj_idx),c=np.arange(len(time_pts)))
    fig.colorbar(sct, ax=ax, label='time point')
    ax.set_xlabel('PC1')
    ax.set_ylabel('fraction of high intensity pixels in crop (> 0.9)')

    fig,ax = viz.init_plot()
    sct = ax.scatter(X_t[idxs[0],time_pts,0],top_pct[0,:],label='crop '+str(hightraj_idx),c=np.arange(len(time_pts)))
    sct = ax.scatter(X_t[idxs[1],time_pts,0],top_pct[1,:],label='crop '+str(lowtraj_idx),c=np.arange(len(time_pts)))
    fig.colorbar(sct, ax=ax, label='time point')
    ax.set_xlabel('PC1')
    ax.set_ylabel('95th percentile of intensity in crop')
else:
    time_pts = np.arange(X_t.shape[1])
    num_high, top_pct1 = analyze_crops_on_PC(PC,idxs[:2],time_pts,X_t,path_to_imgs,tf,plot_imgs=False)
    num_high, top_pct2 = analyze_crops_on_PC(PC,idxs[2:],time_pts,X_t,path_to_imgs,tf,plot_imgs=False)
    top_pct = np.concatenate((top_pct1,top_pct2),axis=0)
    fig,ax = viz.init_plot()
    sct = ax.scatter(X_t[idxs[0],time_pts,1],top_pct[0,:],label='crop '+str(idxs[0]),c=np.arange(len(time_pts)))
    sct = ax.scatter(X_t[idxs[1],time_pts,1],top_pct[1,:],label='crop '+str(idxs[1]),c=np.arange(len(time_pts)))
    sct = ax.scatter(X_t[idxs[2],time_pts,1],top_pct[2,:],label='crop '+str(idxs[2]),c=np.arange(len(time_pts)))
    sct = ax.scatter(X_t[idxs[3],time_pts,1],top_pct[3,:],label='crop '+str(idxs[3]),c=np.arange(len(time_pts)))
    fig.colorbar(sct, ax=ax, label='time point')
    ax.set_xlabel('PC2')
    ax.set_ylabel('95th percentile of intensity in crop')

# %%
def plot_colorline(ax,x,y,c):
    col = mpl.cm.viridis((c-np.min(c))/(np.max(c)-np.min(c)))
    if len(c) != len(x):
        clr = c[:len(x)]
    else:
        clr = c.copy()
    if len(x) != len(clr):
        raise ValueError('data and color object must be same length')
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=col[i])
    sct = ax.scatter(x, y, c=clr, s=0, cmap=mpl.cm.viridis)
    return sct, ax    
# %%
# animate specified PC axis trajectory for one crop along with images at each time point
fig, ax = plt.subplots(1,2,figsize=(15,5),width_ratios=[1,2])
subfigs = fig.subfigures(2, 1, wspace=0.07)

ii = 3
idx = idxs[ii]
prop = top_pct[ii,:]
sct,ax[1] = plot_colorline(ax[1],[0,1],X_t[idx,:2,PC],prop)
ax[1].set_xlim([0,X_t.shape[1]])
fig.colorbar(sct,ax=ax[1])
sct.set_clim([np.min(prop),np.max(prop)])
if analyze_PC1:
    ax[1].set_ylim([-20,15])
else:
    ax[1].set_ylim([-20,40])
ax[1].set_xlabel('frame number',fontsize=14)
ax[1].set_ylabel('PC'+str(PC+1),fontsize=14)

ome_tif = BioImage(path_to_imgs+'/20240305_T01_001_TP'+str(0).zfill(5)+'.ome.tif',reader=bioio_ome_tiff.Reader)
im = ome_tif.get_image_dask_data('CYX', T=0, C=0).compute()
im_patches = tf(im)

img_crop= im_patches[idx][0]
norm = NormalizeIntensity(img_crop.mean(),img_crop.std())
img_crop = norm(img_crop)
implot = ax[0].imshow(img_crop,cmap='gray')
ax[0].set_title('Crop '+str(idx),fontsize=16)

def update(frame):
    ax[1].clear()
    sct,ax[1] = plot_colorline(ax[1],np.arange(frame),X_t[idx,:frame,PC],prop)
    ax[1].set_xlim([0,X_t.shape[1]])
    if analyze_PC1:
        ax[1].set_ylim([-20,15])
    else:
        ax[1].set_ylim([-20,40])
    ax[1].set_xlabel('frame number',fontsize=14)
    ax[1].set_ylabel('PC'+str(PC+1),fontsize=14)

    ome_tif = BioImage(path_to_imgs+'/20240305_T01_001_TP'+str(frame).zfill(5)+'.ome.tif',reader=bioio_ome_tiff.Reader)
    im = ome_tif.get_image_dask_data('CYX', T=0, C=0).compute()
    im_patches = tf(im)

    img_crop = im_patches[idx][0]
    norm = NormalizeIntensity(img_crop.mean(),img_crop.std())
    img_crop = norm(img_crop)
    ax[0].clear()
    implot = ax[0].imshow(img_crop,cmap='gray')
    ax[0].set_title('Crop '+str(idx), fontsize=16)
    
    return sct,implot


# %%
fps=15
anim = FuncAnimation(fig, update, frames=X_t.shape[1], interval=fps)
writer = PillowWriter(fps=fps, bitrate=1800)
plt.tight_layout()
plt.show()

anim.save('PC'+str(PC+1)+'_animation_crop'+str(idx)+'.gif', writer=writer)



# %%
