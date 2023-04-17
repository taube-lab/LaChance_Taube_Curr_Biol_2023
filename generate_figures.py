# -*- coding: utf-8 -*-
"""

functions for making figures

run this file to generate panels for all main figures

@author: Patrick
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
import seaborn as sns
from utilities import circular
import pickle
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel

shift_bins = 60
hd_bins = 30
d_bins = 20
s_bins = 20
framerate = 30.

sns.set_style("white")
rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':21})
rc('xtick',**{'bottom':True,'major.size':6,'minor.size':6,'major.width':1.5,'minor.width':1.5})
rc('ytick',**{'left':True,'major.size':6,'minor.size':6,'major.width':1.5,'minor.width':1.5})
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
    
    
def rayleigh_r(spike_angles,rates=None,ego=False):
    ''' finds rayleigh mean vector length for direction curves '''
    
    #start vars for x and y rayleigh components
    rx = 0
    ry = 0
    
    #convert spike angles into x and y coordinates, sum up the results -- 
    #if firing rates are provided along with HD plot edges instead of spike angles,
    #do the same thing but with those
    if rates is None:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))
            ry += np.sin(np.deg2rad(spike_angles[i]))
    else:
        for i in range(len(spike_angles)):
            rx += np.cos(np.deg2rad(spike_angles[i]))*rates[i]
            ry += np.sin(np.deg2rad(spike_angles[i]))*rates[i]

    #calculate average x and y values for vector coordinates
    if rates is None:
        if len(spike_angles) == 0:
            spike_angles.append(1)
        rx = rx/len(spike_angles)
        ry = ry/len(spike_angles)
    
    else:
        rx = rx/sum(rates)
        ry = ry/sum(rates)

    #calculate vector length
    r = np.sqrt(rx**2 + ry**2)

    if rx == 0:
        mean_angle = 0
    elif rx > 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx))
    elif rx < 0:
        mean_angle = np.rad2deg(np.arctan(ry/rx)) + 180
    try:
        if mean_angle < 0:
            mean_angle = mean_angle + 360
    except:
        mean_angle = 0
        
    if ego:
        return r,rx,ry, mean_angle
    else:
        return r, mean_angle

    
def plot_hd_map(center_x,center_y,angles,spike_train,destination):
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
    markersize = plt.rcParams['lines.markersize'] ** 2
    
    if 'rect s' in destination:
        markersize = (1.5 * plt.rcParams['lines.markersize']) ** 2
    elif '.6m s' in destination:
        markersize = (2. * plt.rcParams['lines.markersize']) ** 2
    else:
        markersize = plt.rcParams['lines.markersize'] ** 2
    
    spike_x = center_x[spike_train>0]
    spike_y = center_y[spike_train>0]
    spike_angles = angles[spike_train>0]
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(center_x,center_y,color='gray',alpha=0.6,zorder=0)
    ax.scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
    ax.axis('off')
    ax.axis('equal')
    
    fig.savefig(destination,dpi=300)
    
    plt.close()
    

def plot_heatmap(center_x,center_y,spike_train,session_type,destination):
    
    if 'rect' in session_type:
        x_gr = 30
        y_gr = 15
    elif '.6m' in session_type:
        x_gr = 15
        y_gr= 15
    elif '1.2m' in session_type:
        x_gr = 30
        y_gr = 30
    elif 'l_shape' in session_type:
        x_gr = 30
        y_gr = 30
    elif '1m' in session_type:
        x_gr = 25
        y_gr = 25
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    xbins = np.digitize(center_x,bins=np.linspace(np.min(center_x),np.max(center_x)+.01,x_gr+1,endpoint=True)) - 1
    ybins = np.digitize(center_y,bins=np.linspace(np.min(center_y),np.max(center_y)+.01,y_gr+1,endpoint=True)) - 1

    spikes = np.zeros((x_gr,y_gr))
    occ = np.zeros((x_gr,y_gr))
    
    for i in range(len(spike_train)):
        spikes[xbins[i],ybins[i]] += spike_train[i]
        occ[xbins[i],ybins[i]] += 1./30.
    occ[occ<5./30.] = np.nan
        
    heatmap = spikes/occ
    
    smoothed_heatmap = convolve(heatmap,kernel=Gaussian2DKernel(x_stddev=1,y_stddev=1))
    
    if 'l_shape' in session_type:
        for i in range(15,30):
            for j in range(16,30):
                smoothed_heatmap[i,j] = np.nan
                
    
    smoothed_heatmap = np.rot90(smoothed_heatmap,-1)
    smoothed_heatmap = np.flipud(smoothed_heatmap)
    smoothed_heatmap = np.fliplr(smoothed_heatmap)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    
    im = ax.imshow(smoothed_heatmap,vmin=0,vmax = np.ceil(np.nanmax(smoothed_heatmap)),cmap='viridis') 
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,np.ceil(np.nanmax(smoothed_heatmap))])
    cbar.set_ticklabels(['0 Hz','%i Hz' % np.ceil(np.nanmax(smoothed_heatmap))])
    
    ax.axis('off')
    ax.axis('equal')
        
    fig.savefig(destination,dpi=300)
    
    plt.close()


def polar_dot_plot(angles,savefile=None,labels=None,order=None,pretty=None,labelcolors=None):
    
    
    max_lim = 4.82279776210871
        
    angles = np.deg2rad((angles+(360./shift_bins)/2.)%360)
    
    circ_rad = 1.
    initial_spacing = .07
    dot_spacing = .1
    
    circ = lambda r : (r*np.cos(np.linspace(0,2.*np.pi,360)), r*np.sin(np.linspace(0,2.*np.pi,360)))
    pol2cart = lambda pol : (pol[0]*np.cos(pol[1]),pol[0]*np.sin(pol[1]))
    circ_x,circ_y=circ(circ_rad)
    
    bins = np.linspace(0,2.*np.pi,int(shift_bins)+1,endpoint=True)
    
    palette = ['green','blue','red','orange','magenta','gray','skyblue','yellow','aqua','pink','purple','cyan','black','white','indigo','violet','crimson']


    
    if labels is None:

        counts,hist_bins = np.histogram(angles,bins=bins)
        
        mr = np.nansum(counts*np.exp(1j*np.linspace(0,2.*np.pi,int(shift_bins),endpoint=False)))/np.nansum(counts)
        mra = np.arctan2(np.imag(mr),np.real(mr))
        mrl = np.abs(mr)
        
        rx,ry = pol2cart((mrl,mra))
        rxs = [rx]
        rys = [ry]
        
        rs = [mrl]
        
        xs = []
        ys = []
        for i in range(len(counts)):
            for j in range(counts[i]):
                r = circ_rad + initial_spacing + j*dot_spacing
                theta = hist_bins[i]
                x,y = pol2cart((r,theta))
                xs.append(x)
                ys.append(y)
                
    else:
        cell_counts = np.zeros(0)
        cell_hist_bins = np.zeros(0)
        animalnums = []
        colors = []
        rxs = []
        rys = []
        rs = []
        r_colors = []
        bin_counts = np.tile(np.arange(int(shift_bins)),len(np.unique(labels)))
        if order is None:
            unique_labels = np.unique(labels)
        else:
            unique_labels = order
        for l in range(len(unique_labels)):
            label = unique_labels[l]
            label_angles = angles[labels==label]
            c,h = np.histogram(label_angles,bins=bins)
            cell_counts = np.concatenate((cell_counts,c))
            cell_hist_bins = np.concatenate((cell_hist_bins,bins[:int(shift_bins)]))
            for i in range(len(c)):
                animalnums.append(label)
                if labelcolors is not None:
                    colors.append(labelcolors[label])
                else:
                    colors.append(palette[l])
            
            mr = np.nansum(c*np.exp(1j*np.linspace(0,2.*np.pi,int(shift_bins),endpoint=False)))/np.nansum(c)
            mra = np.arctan2(np.imag(mr),np.real(mr))
            mrl = np.abs(mr)
            
            rx,ry = pol2cart((mrl,mra))
            rxs.append(rx)
            rys.append(ry)
            rs.append(mrl)
            if labelcolors is not None:
                r_colors.append(labelcolors[label])
            else:
                r_colors.append(palette[l])
            
        xs = []
        ys = []
        animal_labels = []
        color_labels = []
        counts = np.zeros(int(shift_bins))
        spacing = np.zeros(int(shift_bins))
        for i in range(len(cell_counts)):
            for j in range(int(cell_counts[i])):
                r = circ_rad + initial_spacing + spacing[int(bin_counts[i])]
                theta = cell_hist_bins[i]
                x,y = pol2cart((r,theta))
                xs.append(x)
                ys.append(y)
                animal_labels.append(animalnums[i])
                color_labels.append(colors[i])
                spacing[int(bin_counts[i])] += dot_spacing
                counts[int(bin_counts[i])] += 1.
                
                
    rx,ry = pol2cart((mrl,mra))
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
#    ax=plt.gca()
    ax.plot(circ_x,circ_y,'k-')
    if labels is None:
        ax.plot(xs,ys,'ko',clip_on=False)
        ax.arrow(0,0,rx,ry,head_width=0.12,head_length=0.13,fc='k',ec='k',linewidth=2,length_includes_head=True,clip_on=False)
    else:
        for animal in unique_labels:
            ix = np.where(np.array(animal_labels)==animal)[0]
            ax.scatter(np.array(xs)[ix],np.array(ys)[ix],c=np.array(color_labels)[ix],label=animal,s=76,clip_on=False)
            
            
    longest_axis = np.max([ax.get_xlim()[1]-ax.get_xlim()[0],ax.get_ylim()[1]-ax.get_ylim()[0]])
    height = 10. * longest_axis/max_lim
    width = 10. * longest_axis/max_lim
    
    fig.set_figwidth(width)
    fig.set_figheight(height)

            
    if pretty is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, pretty)
    else:
        ax.legend(fontsize=30)
        
    if labels is None:
        for i in range(len(rxs)):
            ax.arrow(0,0,rxs[i],rys[i],head_width=0.14,head_length=0.13,fc=palette[i],ec=palette[i],linewidth=2,length_includes_head=True)
            ax.text(.1,.9-.1*float(i),'r=%s' % str(round(rs[i],2)),transform=ax.transAxes,color=palette[i],fontsize=20)
    else:
        for i in range(len(rxs)):
            ax.arrow(0,0,rxs[i],rys[i],head_width=0.14,head_length=0.13,fc=r_colors[i],ec=r_colors[i],linewidth=2,length_includes_head=True)
            ax.text(.1,.9-.1*float(i),'r=%s' % str(round(rs[i],2)),transform=ax.transAxes,color=r_colors[i],fontsize=20) 
        
    ax.plot([-1,-.95],[0,0],'k-')
    ax.text(-.81,-.01,'180',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([.95,1],[0,0],'k-')
    ax.text(.88,-.01,'0',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([0,0],[-1,-.95],'k-')
    ax.text(0,-.88,'270',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([0,0],[.95,1],'k-')
    ax.text(0,.86,'90',horizontalalignment='center',verticalalignment='center',fontsize=18)


    plt.axis('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
    
    if savefile is not None:
        dest_folder = os.path.dirname(savefile)
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)
            
        fig.patch.set_visible(False)
        ax.axis('off')
        
        fig.savefig(savefile)
        plt.close()
        
    else:
        plt.show()

        
        
def hd_vector_plots(center_x,center_y,angles,spike_train,session_type,destination):
    
    if 'rect' in session_type:
        x_gr = 8
        y_gr = 4
    elif '.6m' in session_type:
        x_gr = 4
        y_gr= 4
    elif '1.2m' in session_type:
        x_gr = 8
        y_gr = 8
    elif '1m' in session_type:
        x_gr = 8
        y_gr = 8
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    hd_bins = 12

    x_edges = np.linspace(np.min(center_x),np.max(center_x),x_gr+1,endpoint=True)
    y_edges = np.linspace(np.min(center_y),np.max(center_y),y_gr+1,endpoint=True)
    
    curves = np.zeros((8,8,hd_bins))
    
    for i in range(y_gr):
        for j in range(x_gr): 
            
            xmin = x_edges[j]
            xmax = x_edges[j+1]
            ymin = y_edges[i]
            ymax = y_edges[i+1]
            done = False
            
            while not done:
                inds = [(center_x >= xmin) & (center_x <= xmax) & (center_y >= ymin) & (center_y <= ymax)]

                bin_angles = angles[inds]
                bin_spikes = spike_train[inds]
                
                angle_bins = np.digitize(bin_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
                
                spikes = np.zeros(hd_bins)
                occ = np.zeros(hd_bins)
                
                for k in range(len(bin_angles)):
                    occ[angle_bins[k]] += 1./framerate
                    spikes[angle_bins[k]] += bin_spikes[k]
                    
                if np.any(occ < 0.2):
                    xmin -= 0.5
                    xmax += 0.5
                    ymin -= 0.5
                    ymax += 0.5
                    
                elif np.all(occ >= 0.2) or (xmax - xmin >= 30.) or (ymax - ymin >= 30.):
                    
                    curve = spikes/occ
                    
                    if np.nanmax(curve) < .5:
                        curve = np.ones_like(curve)

                    curves[i][j] = curve
         
                    done = True
                    
                    
    rs = np.zeros((8,8))
    rxs = np.zeros_like(rs)
    rys = np.zeros_like(rs)
    mean_angles = np.zeros_like(rs)
    
    angle_edges = np.linspace(15,375,hd_bins,endpoint=False)
    
    for i in range(y_gr):
        for j in range(x_gr):
            rs[i][j], rxs[i][j], rys[i][j], mean_angles[i][j] = rayleigh_r(angle_edges,curves[i][j],ego=True)
            
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x,y=np.meshgrid(np.arange(8),np.arange(8))
    
    rxs[rs<1e-10]=np.nan
    rys[rs<1e-10]=np.nan
    
    if 'l_shape' in session_type:
        rs[4:,4:] = np.nan
        rys[4:,4:] = np.nan
        rxs[4:,4:] = np.nan
    
    ax.quiver(x, y, rxs, rys, pivot='mid',width=0.009,clip_on=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    ax.axis('equal')

    
    fig.savefig(destination,dpi=300)
    plt.close()


def plot_pop_vector(rates,title,destination):
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rates,cmap='viridis',aspect=float(hd_bins)/len(rates))
    ax.set_title(title)
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels(['0','90','180','270','360'])
    ax.set_yticks([0,len(rates)-1])
    ax.set_yticklabels(['1',str(len(rates))])
    ax.set_xlabel('Center bearing (deg)')
    ax.set_ylabel('Cell number')
    
    plt.tight_layout()
    
    fig.savefig(destination)
    
    plt.close()
    
    
def plot_pop_vector_dist(rates,title,destination,celltype='cd'):
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
    xvals=np.linspace(0,81,d_bins + 1,endpoint=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rates,cmap='viridis',aspect=float(len(xvals))/len(rates))
    ax.set_title(title)
    ax.set_yticks([0,len(rates)-1])
    ax.set_yticklabels(['1',str(len(rates))])
    ax.set_xticks([0,5,10,15,20])
    ax.set_xticklabels([0,20,40,60,80])
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Cell number')
    
    plt.tight_layout()
    
    fig.savefig(destination + '_dists_%s.png' % celltype)
    
    plt.close()
    
            
def get_cells_baseline(data,celltype):
    
    for c in np.unique(data['animalcell']):
        row = data[(data['animalcell']==c)&(data['session_num']==1)]
        if len(row) == 0:
            data = data[~(data['animalcell']==c)]
        if len(row) > 0:
            if int(row[celltype])==0:
                data = data[~(data['animalcell']==c)]
    
    return data


def make_all_figures():
    
    ''' figure 1 '''
    
    baseline_datafile = os.getcwd() + '/figures/baseline_data.pickle'
    figdir = os.getcwd() + '/figures'
    
    for i in range(1,8):
        if not os.path.exists(figdir+'/fig%i' % i):
            os.mkdir(figdir+'/fig%i' % i)
    
    with open(baseline_datafile,'rb') as f:
        data = pickle.load(f)

    hd_data=get_cells_baseline(data,'hd')
    cb_data=get_cells_baseline(data,'cb')
    cdl_data=get_cells_baseline(data,'cdl')
    cdg_data=get_cells_baseline(data,'cdg')
    cd_data = pd.merge(cdl_data,cdg_data,'outer')
    

    ''' panel C '''
    
    npurecdcs = len(cd_data[(cd_data['hd']==0)&(cd_data['cb']==0)])
    ncdcxhdcs = len(cd_data[(cd_data['hd']==1)&(cd_data['cb']==0)])
    ncbcxcdcs = len(cd_data[(cd_data['hd']==0)&(cd_data['cb']==1)])
    nall = len(cd_data[(cd_data['hd']==1)&(cd_data['cb']==1)])
    npurecbcs = len(cb_data[(cb_data['hd']==0)&(cb_data['cdl']==0)&(cb_data['cdg']==0)])
    ncbcxhdcs = len(cb_data[(cb_data['hd']==1)&(cb_data['cdl']==0)&(cb_data['cdg']==0)])
    npurehdcs = len(hd_data[(hd_data['cb']==0)&(hd_data['cdl']==0)&(hd_data['cdg']==0)])
    nunclassified = len(data) - (npurecbcs+npurecdcs+npurehdcs+ncbcxcdcs+ncbcxhdcs+ncdcxhdcs+nall)
    
    nums = [npurecbcs,npurecdcs,npurehdcs,ncbcxcdcs,ncbcxhdcs,ncdcxhdcs,nall,nunclassified]
    labels = ['Center Bearing','Center Dist','Head Direction','Center Bearing x \nCenter Dist','Center Bearing x \nHead Direction','Center Dist x Head Direction','Center Bearing x \nCenter Dist x \nHead Direction','Other/Unclassified']
    
    colors = ['#1f77b4','#2ca02c','#d62728','orange','lightblue','pink','lightgreen','gray']
    
    blank = list(np.where(np.array(nums)==0)[0])
    for i in range(len(blank)):
        ind = blank[i] - i
        nums.pop(ind)
        labels.pop(ind)
        colors.pop(ind)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pie(x=nums,labels=labels,colors=colors,autopct='%1.00f%%')
    ax.set_title('All POR Cells (N = %i)' % len(data))
    plt.tight_layout()
    fig.savefig(figdir + '/fig1/panel_C.svg')
    plt.show()
    
    
    ''' panel D '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(cb_data['center ego mean angle'],bins=np.linspace(0,360,19,endpoint=True),color='cornflowerblue',align='mid')
    ax.set_xticks([0,90,180,270,360])
    ax.set_ylim([0,25])
    fig.savefig(figdir + '/fig1/panel_D.png',dpi=400)
    plt.close()
    
    
    ''' panel C '''
    
    npurecdcs = len(cd_data[(cd_data['hd']==0)&(cd_data['cb']==0)])
    ncdcxhdcs = len(cd_data[(cd_data['hd']==1)&(cd_data['cb']==0)])
    ncbcxcdcs = len(cd_data[(cd_data['hd']==0)&(cd_data['cb']==1)])
    nall = len(cd_data[(cd_data['hd']==1)&(cd_data['cb']==1)])
    npurecbcs = len(cb_data[(cb_data['hd']==0)&(cb_data['cdl']==0)&(cb_data['cdg']==0)])
    ncbcxhdcs = len(cb_data[(cb_data['hd']==1)&(cb_data['cdl']==0)&(cb_data['cdg']==0)])
    npurehdcs = len(hd_data[(hd_data['cb']==0)&(hd_data['cdl']==0)&(hd_data['cdg']==0)])
    nunclassified = len(data) - (npurecbcs+npurecdcs+npurehdcs+ncbcxcdcs+ncbcxhdcs+ncdcxhdcs+nall)
    
    nums = [npurecbcs,npurecdcs,npurehdcs,ncbcxcdcs,ncbcxhdcs,ncdcxhdcs,nall,nunclassified]
    labels = ['Center Bearing','Center Dist','Head Direction','Center Bearing x \nCenter Dist','Center Bearing x \nHead Direction','Center Dist x Head Direction','Center Bearing x \nCenter Dist x \nHead Direction','Other/Unclassified']
    
    colors = ['#1f77b4','#2ca02c','#d62728','orange','lightblue','pink','lightgreen','gray']
    
    blank = list(np.where(np.array(nums)==0)[0])
    for i in range(len(blank)):
        ind = blank[i] - i
        nums.pop(ind)
        labels.pop(ind)
        colors.pop(ind)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pie(x=nums,labels=labels,colors=colors,autopct='%1.00f%%')
    ax.set_title('All POR Cells (N = %i)' % len(data))
    plt.tight_layout()
    plt.show()
    
    ''' panel E '''
    
    rates_file = os.getcwd() + '/figures/baseline_rates.pickle'
    with open(rates_file,'rb') as f:
        baseline_rates = pickle.load(f)
    
    dest_folder = figdir + '/fig1'
    plot_pop_vector(baseline_rates['bearing_rates'],'CB',dest_folder + '/panel_E.png')
    
    ''' panel F '''

    plot_pop_vector_dist(baseline_rates['large_dist_rates'],'CD 1.2m',dest_folder + '/panel_F_bottom.png','cd')
    plot_pop_vector_dist(baseline_rates['small_dist_rates'],'CD 1m',dest_folder + '/panel_F_top','cd')
    
    
    ''' center vs. 2-wall figures - Figs 2-4 '''
                            
    figdir = os.getcwd() + '/figures'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    datafile = os.getcwd() + '/figures/center_vs_2wall_data.pickle'    
    with open(datafile,'rb') as f:
        exp_dict = pickle.load(f)

    for celltype in ['cb','cdl']:
        for shape in ['l_shape','rect']:
            
            if celltype == 'cb':
                measures = ['center_vs_wall_ind','center_vs_wall_rayleigh']
            elif celltype == 'cdl':
                measures = ['center_vs_wall_ind','dist_fit_diff']
            
            for measure in measures:

                pre_index = exp_dict[shape][celltype][measure]['pre']
                exp_index = exp_dict[shape][celltype][measure]['exp']
                post_index = exp_dict[shape][celltype][measure]['post']

                jitter = exp_dict[shape][celltype][measure]['jitter']
                
                line1 = exp_dict[shape][celltype][measure]['line1']
                line2 = exp_dict[shape][celltype][measure]['line2']
                line3 = exp_dict[shape][celltype][measure]['line3']

                offset = 0
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                ax.plot(np.zeros_like(pre_index)+jitter,pre_index-offset,'ko',alpha=0.8,zorder=100,clip_on=False)
                ax.plot(np.zeros_like(exp_index)+1+jitter,exp_index-offset,'ko',alpha=0.8,zorder=101,clip_on=False)
                ax.plot(np.zeros_like(post_index)+2+jitter,post_index-offset,'ko',alpha=0.8,zorder=102,clip_on=False)

                cutoff = 2. * np.std(pre_index)
                
                ax.plot([-.5,2.5],[np.mean(pre_index) + cutoff - offset, np.mean(pre_index) + cutoff - offset],'r--',alpha=0.5)
                ax.plot([-.5,2.5],[np.mean(pre_index) - cutoff - offset, np.mean(pre_index) - cutoff - offset],'r--',alpha=0.5)
    
                if celltype == 'cb':
                    for i in range(len(line1)):
                        if line1[i] > 0:
                            ax.plot([0+jitter[i],1+jitter[i]],[pre_index[i]-offset,exp_index[i]-offset],'cornflowerblue',clip_on=False)
                        elif line1[i] < 0:
                            ax.plot([0+jitter[i],1+jitter[i]],[pre_index[i]-offset,exp_index[i]-offset],'gray',clip_on=False)
                    for i in range(len(line2)):
                        if line2[i] > 0:
                            ax.plot([1+jitter[i],2+jitter[i]],[exp_index[i]-offset,post_index[i]-offset],'cornflowerblue',clip_on=False)
                        elif line2[i] < 0:
                            ax.plot([1+jitter[i],2+jitter[i]],[exp_index[i]-offset,post_index[i]-offset],'gray',clip_on=False)
                            
                elif celltype == 'cdl':
                    for i in range(len(line1)):
                        if line1[i] > 0:
                            ax.plot([0+jitter[i],1+jitter[i]],[pre_index[i]-offset,exp_index[i]-offset],'limegreen',clip_on=False)
                        elif line1[i] < 0:
                            ax.plot([0+jitter[i],1+jitter[i]],[pre_index[i]-offset,exp_index[i]-offset],'gray',clip_on=False)
                    for i in range(len(line2)):
                        if line2[i] > 0:
                            ax.plot([1+jitter[i],2+jitter[i]],[exp_index[i]-offset,post_index[i]-offset],'limegreen',clip_on=False)
                        elif line2[i] < 0:
                            ax.plot([1+jitter[i],2+jitter[i]],[exp_index[i]-offset,post_index[i]-offset],'gray',clip_on=False)


                ax.set_xlim([-.5,2.5])

                temp_offset = np.mean(pre_index)
                
                high_val = np.nanmax(np.concatenate([pre_index,exp_index,post_index])) - temp_offset
                low_val = np.nanmin(np.concatenate([pre_index,exp_index,post_index])) - temp_offset

                if high_val < .3 and low_val > -.3:
                    ax.set_ylim([-.3+temp_offset,.3+temp_offset])
                elif high_val > .3 and low_val > -.3:
                    ax.set_ylim([-.3+temp_offset,1.2*(high_val+temp_offset)])
                elif high_val < .3 and low_val < -.3:
                    ax.set_ylim([1.2*(low_val+temp_offset),.3+temp_offset])   
                
                ax.set_xticks([0,1,2])
                if shape == 'l_shape':
                    ax.set_xticklabels(['Square 1','L-shape','Square 2'])
                elif shape == 'rect':
                    ax.set_xticklabels(['Square 1','Rectangle','Square 2'])
                
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                                
                
                if celltype == 'cb' and shape == 'l_shape':
                    if measure == 'center_vs_wall_rayleigh':
                        ax.set_ylabel('MVL center - MVL wall')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig2/panel_A.png',dpi=400)
                    elif measure == 'center_vs_wall_ind':
                        ax.set_ylabel('Likelihood index')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig2/panel_B.png',dpi=400)
                elif celltype == 'cb' and shape == 'rect':
                    if measure == 'center_vs_wall_rayleigh':
                        ax.set_ylabel('MVL center - MVL wall')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig4/panel_A.png',dpi=400)
                    elif measure == 'center_vs_wall_ind':
                        ax.set_ylabel('Likelihood index')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig4/panel_B.png',dpi=400)
                        
                        
                elif celltype == 'cdl' and shape == 'l_shape':
                    if measure == 'dist_fit_diff':
                        ax.set_ylabel('R2 center - R2 wall')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig3/panel_A.png',dpi=400)
                    elif measure == 'center_vs_wall_ind':
                        ax.set_ylabel('Likelihood index')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig3/panel_B.png',dpi=400)
                elif celltype == 'cdl' and shape == 'rect':
                    if measure == 'dist_fit_diff':
                        ax.set_ylabel('R2 center - R2 wall')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig4/panel_F.png',dpi=400)
                    elif measure == 'center_vs_wall_ind':
                        ax.set_ylabel('Likelihood index')
                        plt.tight_layout()
                        fig.savefig(figdir + '/fig4/panel_G.png',dpi=400)

                plt.close()
                
                
            if celltype == 'cb':
                non_llps = 'center_vs_wall_rayleigh'
            elif celltype == 'cdl':
                non_llps = 'dist_fit_diff'
                                
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            rayleigh_pre = exp_dict[shape][celltype][non_llps]['pre']
            high_rayleigh = 2. * np.std(rayleigh_pre) + np.mean(rayleigh_pre)
            low_rayleigh = -2. * np.std(rayleigh_pre) + np.mean(rayleigh_pre)
            
            llps_pre = exp_dict[shape][celltype]['center_vs_wall_ind']['pre']
            high_llps = 2. * np.std(llps_pre) + np.mean(llps_pre)
            low_llps = -2. * np.std(llps_pre) + np.mean(llps_pre)

            
            rayleigh_index = exp_dict[shape][celltype][non_llps]['exp']
            llps_index = exp_dict[shape][celltype]['center_vs_wall_ind']['exp']

            rayleigh_offset = 0
            llps_offset = 0
        
            colors = np.array(['gray'] * len(rayleigh_index))
            colors[rayleigh_index > high_rayleigh] = 'green'
            colors[llps_index > high_llps] = 'green'
            colors[rayleigh_index < low_rayleigh] = 'red'
            colors[llps_index < low_llps] = 'red'
            colors[(llps_index < low_llps)&(rayleigh_index > high_rayleigh)] = 'gray'
            colors[(llps_index > high_llps)&(rayleigh_index < low_rayleigh)] = 'gray'
            
            strong_center = np.where(colors=='gree')
            strong_wall = np.where(colors=='red')
            intermediate = np.where(colors=='gray')
            
            if celltype == 'cb':
                ax.scatter(rayleigh_index[strong_center] - rayleigh_offset, llps_index[strong_center] - llps_offset,zorder=101,c='cornflowerblue')
                ax.scatter(rayleigh_index[strong_wall] - rayleigh_offset, llps_index[strong_wall] - llps_offset,zorder=102,c='darkorange')
                ax.scatter(rayleigh_index[intermediate] - rayleigh_offset, llps_index[intermediate] - llps_offset,zorder=103,c='gray')
            elif celltype == 'cdl':
                ax.scatter(rayleigh_index[strong_center] - rayleigh_offset, llps_index[strong_center] - llps_offset,zorder=101,c='limegreen')
                ax.scatter(rayleigh_index[strong_wall] - rayleigh_offset, llps_index[strong_wall] - llps_offset,zorder=102,c='darkorange')
                ax.scatter(rayleigh_index[intermediate] - rayleigh_offset, llps_index[intermediate] - llps_offset,zorder=103,c='gray')

            plt.plot([-1,1],[high_llps - llps_offset,high_llps - llps_offset],'r--',alpha=0.6)
            plt.plot([-1,1],[low_llps - llps_offset,low_llps - llps_offset],'r--',alpha=0.6)
            plt.plot([high_rayleigh - rayleigh_offset,high_rayleigh - rayleigh_offset],[-1,1],'r--',alpha=0.6)
            plt.plot([low_rayleigh - rayleigh_offset,low_rayleigh - rayleigh_offset],[-1,1],'r--',alpha=0.6)
                
            llps_target = 0.3 + np.mean(llps_pre)
            rayleigh_target = 0.3 + np.mean(rayleigh_pre)

            high_x = np.nanmax(rayleigh_index - rayleigh_offset)
            low_x = np.nanmin(rayleigh_index - rayleigh_offset)

            if high_x < rayleigh_target and low_x > rayleigh_target - 0.6:
                ax.set_xlim([rayleigh_target - 0.6, rayleigh_target])
            elif high_x > rayleigh_target and low_x > rayleigh_target - 0.6:
                ax.set_xlim([rayleigh_target - 0.6, high_x  * 1.2])
            elif high_x < rayleigh_target and low_x < rayleigh_target - 0.6:
                ax.set_xlim([low_x * 1.2, rayleigh_target])
                
            high_y = np.nanmax(llps_index - llps_offset)
            low_y = np.nanmin(llps_index - llps_offset)

            if high_y < llps_target and low_y > llps_target - 0.6:
                ax.set_ylim([llps_target - 0.6, llps_target])
            elif high_y > llps_target and low_y > llps_target - 0.6:
                ax.set_ylim([llps_target - 0.6, high_y  * 1.2])
            elif high_y < llps_target and low_y < llps_target - 0.6:
                ax.set_ylim([low_y * 1.2, llps_target])
            
            if celltype == 'cb' and shape == 'l_shape':
                ax.set_ylabel('MVL center - MVL wall')
                ax.set_xlabel('Likelihood index')
                plt.tight_layout()
                fig.savefig(figdir + '/fig2/panel_C.png',dpi=400)
            elif celltype == 'cb' and shape == 'rect':
                ax.set_ylabel('MVL center - MVL wall')
                ax.set_xlabel('Likelihood index')
                plt.tight_layout()
                fig.savefig(figdir + '/fig4/panel_C.png',dpi=400)
                
            if celltype == 'cdl' and shape == 'l_shape':
                ax.set_ylabel('R2 center - R2 wall')
                ax.set_xlabel('Likelihood index')
                plt.tight_layout()
                fig.savefig(figdir + '/fig3/panel_C.png',dpi=400)
            elif celltype == 'cdl' and shape == 'rect':
                ax.set_ylabel('R2 center - R2 wall')
                ax.set_xlabel('Likelihood index')
                plt.tight_layout()
                fig.savefig(figdir + '/fig4/panel_H.png',dpi=400)
                
            plt.close()



    ''' average ratemaps '''
    
    ''' fig 2 panel F '''
    
    ratemap_file = os.getcwd() + '/figures/avg_ratemaps.pickle'
    with open(ratemap_file,'rb') as f:
        avg_ratemaps = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig2/panel_F'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    mean_pre = avg_ratemaps['l_shape']['cb']['strong_center']['pre']
    mean_exp = avg_ratemaps['l_shape']['cb']['strong_center']['exp']
    mean_post = avg_ratemaps['l_shape']['cb']['strong_center']['post']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_pre,vmin=0, vmax = 0.6, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,.6])
    cbar.set_ticklabels(['0','0.6'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_pre_center.png', dpi=400)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_exp,vmin=0, vmax = 0.6, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,.6])
    cbar.set_ticklabels(['0','0.6'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_exp_center.png', dpi=400)
    plt.close()
    
    celltype = 'cb'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)

    im = ax.imshow(mean_exp - mean_pre, vmin=-.2, vmax = 0.2, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.2,0.2])
    cbar.set_ticklabels(['-0.2','+0.2'])

    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/exp_minus_pre_center.png', dpi=400)
    plt.close()
    
    
    ''' fig 2 panel G '''
    
    figdir = os.getcwd() + '/figures/fig2/panel_G'
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    mean_pre = avg_ratemaps['l_shape']['cb']['strong_wall']['pre']
    mean_exp = avg_ratemaps['l_shape']['cb']['strong_wall']['exp']
    mean_post = avg_ratemaps['l_shape']['cb']['strong_wall']['post']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_pre,vmin=0, vmax = 0.6, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,0.6])
    cbar.set_ticklabels(['0','0.6'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_pre_wall.png', dpi=400)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_exp,vmin=0, vmax = 0.6, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,0.6])
    cbar.set_ticklabels(['0','0.6'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_exp_wall.png', dpi=400)
    plt.close()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)

    im = ax.imshow(mean_exp - mean_pre, vmin=-.2, vmax = 0.2, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.2,0.2])
    cbar.set_ticklabels(['-0.2','+0.2'])

    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/exp_minus_pre_wall.png', dpi=400)
    plt.close()
    
    
    
    ''' fig 3 panel F '''
    
    ratemap_file = os.getcwd() + '/figures/avg_ratemaps.pickle'
    with open(ratemap_file,'rb') as f:
        avg_ratemaps = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig3/panel_F'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    mean_pre = avg_ratemaps['l_shape']['cd']['strong_center']['pre']
    mean_exp = avg_ratemaps['l_shape']['cd']['strong_center']['exp']
    mean_post = avg_ratemaps['l_shape']['cd']['strong_center']['post']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_pre,vmin=0, vmax = 0.7, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,.7])
    cbar.set_ticklabels(['0','0.7'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_pre_center.png', dpi=400)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_exp,vmin=0, vmax = 0.7, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,.7])
    cbar.set_ticklabels(['0','0.7'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_exp_center.png', dpi=400)
    plt.close()
    
    celltype = 'cb'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)

    im = ax.imshow(mean_exp - mean_pre,vmin=-.2, vmax = 0.4, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.2,0.4])
    cbar.set_ticklabels(['-0.2','+0.4'])
    
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/exp_minus_pre_center.png', dpi=400)
    plt.close()
    
    
    ''' fig 3 panel G '''
    
    figdir = os.getcwd() + '/figures/fig3/panel_G'
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    mean_pre = avg_ratemaps['l_shape']['cd']['strong_wall']['pre']
    mean_exp = avg_ratemaps['l_shape']['cd']['strong_wall']['exp']
    mean_post = avg_ratemaps['l_shape']['cd']['strong_wall']['post']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_pre,vmin=0, vmax = 0.7, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,0.7])
    cbar.set_ticklabels(['0','0.7'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_pre_wall.png', dpi=400)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    im = ax.imshow(mean_exp,vmin=0, vmax = 0.7, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,0.7])
    cbar.set_ticklabels(['0','0.7'])
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/mean_exp_wall.png', dpi=400)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)

    im = ax.imshow(mean_exp - mean_pre,vmin=-.2, vmax = 0.4, cmap='viridis')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.2,0.4])
    cbar.set_ticklabels(['-0.2','+0.4'])
    
    ax.axis('off')
    ax.axis('equal')
    fig.savefig(figdir + '/exp_minus_pre_wall.png', dpi=400)
    plt.close()


    ''' example cells for main figure 2 '''
    
    ''' fig 2 panel D top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig2_panel_Dtop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig2/panel_Dtop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 2 panel D bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig2_panel_Dbottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig2/panel_Dbottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 2 panel E top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig2_panel_Etop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig2/panel_Etop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 2 panel E bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig2_panel_Ebottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig2/panel_Ebottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)
        
    stacked_ego_curves(celldict,figdir)
    
    

    ''' example cells for main figure 3 '''
    
    ''' fig 3 panel D top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig3_panel_Dtop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig3/panel_Dtop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    ''' fig 3 panel D bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig3_panel_Dbottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig3/panel_Dbottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    ''' fig 3 panel E top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig3_panel_Etop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig3/panel_Etop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    ''' fig 3 panel E bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig3_panel_Ebottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig3/panel_Ebottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '1.2m l_shape'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    
    ''' example cells for main figure 4 '''
    
    
    ''' fig 4 panel D '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig4_panel_D.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig4/panel_D'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = 'rect'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 4 panel E '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig4_panel_E.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig4/panel_E'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = 'rect'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 4 panel I '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig4_panel_I.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig4/panel_I'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = 'rect'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    ''' fig 4 panel J '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig4_panel_J.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig4/panel_J'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = 'rect'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    
    ''' example cells for main figure 5 '''

    ''' fig 5 panel A '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig5_panel_A.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig5/panel_A'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '.6m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_ego_curves(celldict,figdir)
    
    
    ''' fig 5 panel B '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig5_panel_B.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig5/panel_B'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '.6m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    
    
    ''' fig 5 panel C '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig5_panel_C.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig5/panel_C'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['pre','exp','post']:
        if session == 'pre' or session == 'post':
            session_type = '1.2m'
        elif session == 'exp':
            session_type = '.6m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)

    stacked_dist_curves(celldict,figdir)
    stacked_ego_curves(celldict,figdir)
    
    
    ''' small environment slope comparisons '''
    
    ''' center distance slopes '''
    
    slope_file = os.getcwd() + '/figures/small_dist_slopes.pickle'
    with open(slope_file,'rb') as f:
        slope_data = pickle.load(f)

    figdir = os.getcwd() + '/figures/fig5/panel_D'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    std1 = slope_data['cd']['pre']
    exp = slope_data['cd']['exp']
    
    min_val = 1.2*np.min([np.min(exp),np.min(std1)])
    max_val = 1.2*np.max([np.max(exp),np.max(std1)])

    slope, intercept, r_value,p_value,std_err = linregress(std1,exp)
    fit_y = []
    fit_x = np.linspace(-.5,.5,20,endpoint=True)

    for i in range(len(np.sort(fit_x))):
        fit_y.append(slope*np.sort(fit_x)[i] + intercept)
        
    r2 = r_value**2
    
    print('r: %f' % r_value)
    print('m: %f' % slope)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(std1,exp,color='green',s=60,clip_on=False,zorder=100)
    ax.plot(fit_x,fit_y,color='gray',linestyle='--',linewidth=3,alpha=.9,clip_on=True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    plt.axis('square')

    ax.plot([-.5,.5],[-.5+intercept,.5+intercept],'k',alpha=.8)
    ax.plot([-.5,.5],[-1+intercept,1+intercept],'cornflowerblue',alpha=.8)

    ax.set_xticks([-.3,0,.3])
    ax.set_yticks([-.3,0,.3])
    ax.set_xlim([-.5,.5])
    ax.set_ylim([-.5,.5])
    
    ax.set_title('Center distance slopes')
    ax.set_xlabel('Large1 slope (Hz/cm)')
    ax.set_ylabel('Small slope (Hz/cm)')

    plt.tight_layout()
    
    fig.savefig(figdir + '/center_slopes.png', dpi=400)
    plt.close()
        
        
    ''' wall distance slopes '''
    
    std1 = slope_data['wd']['pre']
    exp = slope_data['wd']['exp']
    
    min_val = 1.2*np.min([np.min(exp),np.min(std1)])
    max_val = 1.2*np.max([np.max(exp),np.max(std1)])

    slope, intercept, r_value,p_value,std_err = linregress(std1,exp)
    fit_y = []
    fit_x = np.linspace(-.6,.6,20,endpoint=True)

    for i in range(len(np.sort(fit_x))):
        fit_y.append(slope*np.sort(fit_x)[i] + intercept)
        
    r2 = r_value**2
    
    print('r: %f' % r_value)
    print('m: %f' % slope)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(std1,exp,color='black',s=60,alpha=.7,clip_on=False,zorder=100)
    ax.plot(fit_x,fit_y,color='gray',linestyle='--',linewidth=3,alpha=.9,clip_on=True)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    plt.axis('square')

    ax.plot([-.6,.6],[-.6+intercept,.6+intercept],'k',alpha=.8)
    ax.plot([-.6,.6],[-1.2+intercept,1.2+intercept],'cornflowerblue',alpha=.8)

    ax.set_xticks([-.6,-.3,0,.3,.6])
    ax.set_yticks([-.6,-.3,0,.3,.6])
    ax.set_xlim([-.6,.6])
    ax.set_ylim([-.6,.6])
    
    ax.set_title('Wall distance slopes')
    ax.set_xlabel('Large1 slope (Hz/cm)')
    ax.set_ylabel('Small slope (Hz/cm)')

    plt.tight_layout()
    fig.savefig(figdir + '/wall_slopes.png', dpi=400)

    plt.close()
    
    
    ''' example cells for figure 6 '''
    
    ''' fig 6 panel B top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig6_panel_Btop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig6/panel_Btop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['std1','plexi','raised','raised_plexi','std2']:
        session_type = '1m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)
    
    plexi_stacked_rates(celldict, 'cb', figdir)
    
    
    ''' fig 6 panel B bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig6_panel_Bbottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig6/panel_Bbottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['std1','plexi','raised','raised_plexi','std2']:
        session_type = '1m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        hd_vector_plots(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],session_type,figdir+'/hd_vectors_%s.png' % session)
    
    plexi_stacked_rates(celldict, 'cb', figdir)
    


    ''' example cells for figure 7 '''

    ''' fig 7 panel B top '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig7_panel_Btop.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig7/panel_Btop'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['std1','plexi','raised','raised_plexi','std2']:
        session_type = '1m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)
    
    plexi_stacked_rates(celldict, 'cd', figdir)

    ''' fig 7 panel B bottom '''
    
    cell_file = os.getcwd() + '/figures/example_cells/fig7_panel_Bbottom.pickle'
    with open(cell_file,'rb') as f:
        celldict = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig7/panel_Bbottom'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for session in ['std1','plexi','raised','raised_plexi','std2']:
        session_type = '1m'
        plot_hd_map(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['angles'],celldict[session]['spike_train'],figdir+'/hd_path_spike_%s.png' % session)
        plot_heatmap(celldict[session]['center_x'],celldict[session]['center_y'],celldict[session]['spike_train'],session_type,figdir+'/heatmap_%s.png' % session)
    
    plexi_stacked_rates(celldict, 'cd', figdir)



    ''' Fig 6 population data '''
    
    ''' mean angle comparisons '''
    
    plexi_file = os.getcwd() + '/figures/plexi_data.pickle'
    with open(plexi_file,'rb') as f:
        plexi_data = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig6'
    
    plexi_diffs = circular.closest_difference(plexi_data['cb']['plexi']['mean_angle'],plexi_data['cb']['std1']['mean_angle'])
    polar_dot_plot(plexi_diffs,savefile=figdir + '/plexi_vs_std1.svg')
    
    raised_diffs = circular.closest_difference(plexi_data['cb']['raised']['mean_angle'],plexi_data['cb']['std1']['mean_angle'])
    polar_dot_plot(raised_diffs,savefile=figdir + '/raised_vs_std1.svg')
    
    raised_plexi_diffs = circular.closest_difference(plexi_data['cb']['raised_plexi']['mean_angle'],plexi_data['cb']['std1']['mean_angle'])
    polar_dot_plot(raised_plexi_diffs,savefile=figdir + '/raised_plexi_vs_std1.svg')
    
    std2_diffs = circular.closest_difference(plexi_data['cb']['std2']['mean_angle'],plexi_data['cb']['std1']['mean_angle'])
    polar_dot_plot(std2_diffs,savefile=figdir + '/std2_vs_std1.svg')
    
    
    ''' rayleigh r comparisons '''

    plexi = plexi_data['cb']['plexi']['rayleigh_r'] - plexi_data['cb']['std1']['rayleigh_r']
    raised = plexi_data['cb']['raised']['rayleigh_r'] - plexi_data['cb']['std1']['rayleigh_r']
    raised_plexi = plexi_data['cb']['raised_plexi']['rayleigh_r'] - plexi_data['cb']['std1']['rayleigh_r']
    std2 = plexi_data['cb']['std2']['rayleigh_r'] - plexi_data['cb']['std1']['rayleigh_r']

    names = ['std1']*len(plexi) + ['plexi']*len(plexi) + ['raised']*len(raised) + ['raised plexi']*len(raised_plexi) + ['std2']*len(std2)

    vals = np.concatenate((plexi,raised,raised_plexi,std2))
    names = ['plexi']*len(plexi) + ['raised']*len(raised) + ['raised plexi']*len(raised_plexi) + ['std2']*len(std2)
    
    data_df = pd.DataFrame({'val':vals,'label':names})
    
    fig = plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue'],clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_ylim([-.45,.45])
    ax.set_yticks([-.4,-.2,0,.2,.4])
    ax.set_ylabel('delt MVL')
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.set_xlabel('')
    
    median_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()  # "X" or "Y"
        median_val = data_df[data_df['label']==var].val.mean()
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=10,alpha=0.8)
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    
    plt.tight_layout()
    
    fig.savefig(figdir + '/MVL_comparisons.png',dpi=400)
    plt.close()
    
    
    
    ''' figure 7 dist slope comparisons '''

    plexi_file = os.getcwd() + '/figures/plexi_data.pickle'
    with open(plexi_file,'rb') as f:
        plexi_data = pickle.load(f)
        
    figdir = os.getcwd() + '/figures/fig7'

    ids = ['plexi','raised','raised_plexi','std2']

    for i in ids:
    
        std1 = plexi_data['cd']['std1']['slope']
        exp = plexi_data['cd'][i]['slope']
        
        min_val = 1.2*np.min([np.min(exp),np.min(std1)])
        max_val = 1.2*np.max([np.max(exp),np.max(std1)])
        if i=='std2':
            slope, intercept, r_value,p_value,std_err = linregress(std1,exp)
            fit_y = []
            fit_x = np.sort(np.array(std1))
            for j in range(len(np.sort(fit_x))):
                fit_y.append(slope*np.sort(fit_x)[j] + intercept)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(std1,exp,color='green',s=60,clip_on=False)
        if i=='std2':
            ax.plot(np.sort(std1),fit_y,color='green',linestyle='--',linewidth=2.5,alpha=.7)
        ax.set_ylim([min_val,max_val])
        ax.set_xlim([min_val,max_val])
        ax.set_xlabel('Std1 slope (Hz/cm)')
        ax.set_ylabel('%s slope' % i)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.axis('square')
        ax.set_xlim([-.2,.2])
        ax.set_ylim([-.2,.2])
        ax.set_xticks([-.2,0,.2])
        ax.set_yticks([-.2,0,.2])
        if i==1:
            ax.set_xticks([-.2,0,.2])
            ax.set_yticks([-.2,0,.2])
        ax.set_xlim([-.22,.22])
        ax.set_ylim([-.22,.22])

        plt.tight_layout()
        
        fig.savefig(figdir + '/dist_slopes_%s_vs_std1.png' % i,dpi=400)
        plt.show()
    
        plt.close()



def stacked_ego_curves(celldict,destination):
    
    ymax = np.max([celldict['pre']['cb_rates'],celldict['exp']['cb_rates'],celldict['post']['cb_rates'],
                   celldict['pre']['wb_rates'],celldict['exp']['wb_rates'],celldict['post']['wb_rates']])
    
    
    for measurement in ['cb_rates','wb_rates']:

        pre_rates = celldict['pre'][measurement]
        pre_rates = np.append(pre_rates,pre_rates[0])
        exp_rates = celldict['exp'][measurement]
        exp_rates = np.append(exp_rates,exp_rates[0])
        post_rates = celldict['post'][measurement]
        post_rates = np.append(post_rates,post_rates[0])
        
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if 'cb' in measurement:
            ax.plot(pre_rates,color='darkblue',linewidth=3.0)
            ax.plot(exp_rates,color='blue',linestyle='--',linewidth=3.0,zorder=100)
            ax.plot(post_rates,color='lightblue',linestyle='-',linewidth=3.0)
            ax.set_xlabel('Center bearing (deg)')
            
        elif 'wb' in measurement:
            ax.plot(pre_rates,color='black',linewidth=3.0)
            ax.plot(exp_rates,color='black',linestyle='--',linewidth=3.0,zorder=100)
            ax.plot(post_rates,color='gray',linestyle='-',linewidth=3.0)
            ax.set_xlabel('Wall bearing (deg)')
    
        ax.set_xlim([0,30])
        ax.set_xticks([0,7.5,15,22.5,30])
        ax.set_xticklabels([0,90,180,270,360])
        ax.set_ylim([0,1.1*ymax])
            
        yticks = ax.get_yticks()
        for j in yticks:
            # print(np.float(j))
            if np.float(j) == np.float(7.5):
                ax.set_yticks([0,4,8])

        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
 
        plt.tight_layout()
        fig.savefig(destination + '/' + measurement + '.png',dpi=400)
        
        plt.close()
        
        
def stacked_dist_curves(celldict,destination):
    
    ymax = np.nanmax([celldict['pre']['cd_rates'],celldict['exp']['cd_rates'],celldict['post']['cd_rates'],
                   celldict['pre']['wd_rates'],celldict['exp']['wd_rates'],celldict['post']['wd_rates']])
        
    for measurement in ['cd_rates','wd_rates']:

        pre_rates = celldict['pre'][measurement]
        exp_rates = celldict['exp'][measurement]
        post_rates = celldict['post'][measurement]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if 'cd' in measurement:
            ax.plot(pre_rates,color='darkgreen',linewidth=3.0)
            ax.plot(exp_rates,color='green',linestyle='--',linewidth=3.0,zorder=100)
            ax.plot(post_rates,color='lightgreen',linestyle='-',linewidth=3.0)
        elif 'wd' in measurement:
            ax.plot(pre_rates,color='black',linewidth=3.0,alpha=.8)
            ax.plot(exp_rates,color='black',linestyle='--',linewidth=3.0,zorder=100)
            ax.plot(post_rates,color='gray',linestyle='-',linewidth=3.0)

        ax.set_xlim([0,d_bins])
        if measurement == 'cd_rates':
            ax.set_xlabel('Center distance (bins)')
        elif measurement == 'wd_rates':
            ax.set_xlabel('Wall distance (bins)')

        ax.set_ylim([0,1.1*ymax])
            
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        
        yticks = ax.get_yticks()
        for j in yticks:
            # print(np.float(j))
            # print(np.float(j) == np.float(7.5))
            if np.float(j) == np.float(7.5):
                ax.set_yticks([0,4,8])
                
        ylim = ax.get_ylim()[1]
        if ylim > 19 and ylim < 20:
            ax.set_ylim([0,20])
            ax.set_yticks([0,10,20])
        
        fig.savefig(destination + '/' + measurement + '.png',dpi=400)
        
        plt.close()
        
        
        
def plexi_stacked_rates(celldict,cell_type,destination):
    
    if cell_type == 'cd':
        colors = ['gray','darkgreen','green','lightgreen','black']
        linestyles = ['-','-','--','-','--']
        xlabel = 'Center distance (cm)'
        xvals = np.linspace(0,81,d_bins+1,endpoint=True)
    elif cell_type == 'cb':
        colors = ['gray','darkblue','blue','lightblue','black']
        linestyles = ['-','-','--','-','--']
        xlabel = 'Center bearing (deg)'
        xvals = np.linspace(0,360,hd_bins+1,endpoint=True)
    elif cell_type == 'hd':
        colors = ['gray','darkred','red','red','black']
        linestyles = ['-','-','--','-','--']
        xlabel = 'Head direction (deg)'
        xvals = np.linspace(0,360,hd_bins+1,endpoint=True)

        
    if cell_type == 'cb':
        std1_rates = celldict['std1']['cb_rates']
        std1_rates = np.append(std1_rates,std1_rates[0])
        plexi_rates = celldict['plexi']['cb_rates']
        plexi_rates = np.append(plexi_rates,plexi_rates[0])
        raised_rates = celldict['raised']['cb_rates']
        raised_rates = np.append(raised_rates,raised_rates[0])
        raised_plexi_rates = celldict['raised_plexi']['cb_rates']
        raised_plexi_rates = np.append(raised_plexi_rates,raised_plexi_rates[0])
        std2_rates = celldict['std2']['cb_rates']
        std2_rates = np.append(std2_rates,std2_rates[0])
        
    elif cell_type == 'cd':
        std1_rates = celldict['std1']['cd_rates']
        plexi_rates = celldict['plexi']['cd_rates']
        raised_rates = celldict['raised']['cd_rates']
        raised_plexi_rates = celldict['raised_plexi']['cd_rates']
        std2_rates = celldict['std2']['cd_rates']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xvals,std1_rates,color=colors[0],linestyle=linestyles[0],linewidth=3.0)
    ax.plot(xvals,plexi_rates,color=colors[1],linestyle=linestyles[1],linewidth=3.0)
    ax.plot(xvals,raised_rates,color=colors[2],linestyle=linestyles[2],linewidth=3.0)
    ax.plot(xvals,raised_plexi_rates,color=colors[3],linestyle=linestyles[3],linewidth=3.0)
    ax.plot(xvals,std2_rates,color=colors[4],linestyle=linestyles[4],linewidth=3.0)
    ax.set_xlim([np.min(xvals),np.max(xvals)])
    if cell_type == 'cb' or cell_type == 'hd':
        ax.set_xticks([0,90,180,270,360])
    elif cell_type == 'cd':
        ax.set_xticks([0,20,40,60,80])
    ax.set_xlabel(xlabel)
    ax.set_ylim([0,None])
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    fig.savefig(destination + '/' + cell_type + '.png',dpi=400)
    
    plt.close()

            
if __name__ == '__main__':

    make_all_figures()