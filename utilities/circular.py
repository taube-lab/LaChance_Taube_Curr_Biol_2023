# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:24:25 2020

circular statistical tests

@author: Patrick
"""
import os
import numpy as np
from scipy.stats import norm, chi2
from scipy.stats import f as statf
import matplotlib.pyplot as plt


def closest_difference(a,b):
    return (a - b + 180) % 360 - 180

def closest_difference_180(a,b):
    return (a - b + 90) % 180 - 90

def rayleigh_r(angles):
    ''' from Batschelet 1981 
    takes angles IN DEGREES '''
    
    mr = np.nansum(np.exp(1j*np.deg2rad(angles)))/len(angles)
    mean = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))
    r = np.abs(mr)
    
    return r, mean

def angular_dev(r):
    ''' from Batschelet 1981 '''
    
    return np.sqrt(2.*(1. - r))

def concentration(r):
    ''' from Mardia & Jupp 2000 '''
    
    if r < 0.53:
        k = 2.*r + r**3 + (5./6.)*r**5
    elif r >= 0.53 and r < 0.85:
        k = -0.4 + 1.39 * r + 0.43/(1-r)
    elif r >= 0.85:
        k = 1./(2*(1-r))
        
    return k

def v_test(angles,expected):
    ''' from Batschelet 1981 '''
    
    r, mean = rayleigh_r(angles)
    n = len(angles)
    
    v = n * r * np.cos(np.deg2rad(mean) - np.deg2rad(expected))
    u = v * np.sqrt(2./np.float(n))
    p = 1. - norm.cdf(u)
    
    return mean, r, u, v, p

def correlation(angles1,angles2):
    ''' from SenGupta 2001 '''
    
    r1,mean1 = rayleigh_r(angles1)
    r2,mean2 = rayleigh_r(angles2)
    
    mean1 = np.deg2rad(mean1)
    mean2 = np.deg2rad(mean2)
    
    angles1 = np.deg2rad(angles1)
    angles2 = np.deg2rad(angles2)
    
    R = np.sum(np.sin(angles1 - mean1) * np.sin(angles2 - mean2)) / np.sqrt(np.sum((np.sin(angles1 - mean1)**2)) * np.sum(np.sin(angles2 -  mean2)**2 ))

    lamb20 = (1. / len(angles1)) * np.sum(np.sin(angles1 - mean1)**2)
    lamb02 = (1. / len(angles1)) * np.sum(np.sin(angles2 - mean2)**2)
    lamb22 = (1. / len(angles1)) * np.sum((np.sin(angles1 - mean1)**2) * (np.sin(angles2 - mean2)**2))

    #calculate z score
    z = R * np.sqrt(len(angles1) * lamb20 * lamb02 / lamb22)
    
    #calculate p-value from z score
    p = 2. * (1. - norm.cdf(abs(z)))
    
    return R, z, p


def concentration_test(angles):
    ''' from Mardia & Jupp 2000 
    test for different concentrations between TWO groups '''
    
    #all angles concatenated
    all_angles = np.concatenate(angles)
    #total number of angles (measurements)
    n = len(all_angles)
    #get r and mean for all measurements
    grand_r, grand_mean = rayleigh_r(all_angles)
    #get r and mean for two groups
    r1, mean1 = rayleigh_r(angles[0])
    r2, mean2 = rayleigh_r(angles[1])
    n1 = len(angles[0])
    n2 = len(angles[1])

    
    #it actually makes more sense to use the lowest group r-value in deciding
    #which approximation to use -- a low grand_r might mask high group r-values,
    #which the low grand_r approximation cannot handle (i.e. taking the arcsin of
    #values > 1)
    #UPDATE 3/24/22 -- ACTUALLY I DISAGREE WITH THIS!
#    grand_r = np.min([r1,r2])
    
    #constant for variance stabilization for r < 0.45
    a = np.sqrt(3./8.)
    #function for variance stabilization for r < 0.45
    g1 = lambda x: np.arcsin(a*x)
    
    #constants for variance stabilization for 0.45 <= r <= 0.70
    c1 = 1.089
    c2 = 0.258
    c3 = 0.893
    #function for variance stabilization for r < 0.45
    g2 = lambda x: np.arcsinh((x - c1)/c2)
    
    #this approximation does not play well with high group r-values
    if grand_r < 0.45:
        #calc test statistic
        t = (2./np.sqrt(3.)) * (g1(2.*r1) - g1(2.*r2)) / np.sqrt(1./(n1-4.) + 1./(n2-4.))
        #take the two-tailed p-value
        p = 2. * (1. - norm.cdf(abs(t)))
        
        ttype = 't'
        stat = t
    
    #for medium group r-values, this works fine
    elif 0.45 <= grand_r and grand_r <= 0.70:
        #calc test statistic
        t = (g2(r1) - g2(r2)) / (c3 * np.sqrt(1./(n1 - 3.) + 1./(n2 - 3.)))
        #take the two-tailed p-value
        p = 2. * (1. - norm.cdf(abs(t)))
        
        ttype = 't'
        stat = t
        
    #most (if not all) of our data will fall under this category, which essentially
    #amounts to a two-tailed F test
    elif grand_r > 0.70:
        
        #calc F statistic
        F = ((n1 - n1 * r1)/(n1 - 1.))/((n2 - n2 * r2)/(n2 - 1))
        
        #put the dist with higher variance on top so we only have to look at the
        #higher tail of the F distribution
        F = np.max([F,1./F])
        
        #calc degrees of freedom
        df1 = n1 - 1.
        df2 = n2 - 1.
        
        #2 sided p-value
        p = 2. * (1. - statf.cdf(abs(F),df1,df2))
        
        ttype = 'F'
        stat = F
        
    return ttype, stat, p


def multi_concentration_test(angles):
    ''' from Mardia & Jupp 2000 
    test for different concentrations between MORE THAN 2 groups 
    we will use the equation for high r values (section 3) '''
    
    #number of groups
    q = len(angles)
    #all angles concatenated
    all_angles = np.concatenate(angles)
    #total number of angles (measurements)
    n = len(all_angles)
    #get r and mean for all measurements
    grand_r, grand_mean = rayleigh_r(all_angles)
    #multiply r by n to get R
    R = n * grand_r
    
    #calculate Ri for each group i
    Ri = []    
    for angles_i in angles:
        Ri.append(len(angles_i) * rayleigh_r(angles_i)[0])
        
    #make an array
    Ri = np.array(Ri)
    
    v = n - q
    vi = []
    ni = []
    for angles_i in angles:
        vi.append(len(angles_i) - 1)
        ni.append(len(angles_i))
    vi = np.array(vi)
    ni = np.array(ni)
    
    #calculate d
    d = (1./(3.*(q - 1.))) * np.sum((1./vi) - (1./v))
    
    #calculate u
    u = (1./(1. + d)) * (v * np.log((n - np.sum(Ri))/v) - np.sum(vi * np.log((ni - Ri)/vi)))
    
    #u is chi-squared distributed (df = q - 1)
    df = q - 1
    
    #chi square is always one-sided - who knew?
    p = 1. - chi2.cdf(u,df)
    
    ttype = 'chi2'
    stat = u
    
    return ttype, stat, p, df



def watson_williams(angles):
    ''' from Mardia & Jupp 2000 
    basically ANOVA for circular data - takes angles in DEGREES
    
    angles = tuple of angle collections 
    
    returns F statistic, p-value, df1 and df2 '''
    
    #number of groups
    q = len(angles)
    #all angles concatenated
    all_angles = np.concatenate(angles)
    #total number of angles (measurements)
    n = len(all_angles)
    #get r and mean for all measurements
    grand_r, grand_mean = rayleigh_r(all_angles)
    #multiply r by n to get R
    R = n * grand_r
    
    #calculate Ri for each group i
    Ri = []    
    for angles_i in angles:
        Ri.append(len(angles_i) * rayleigh_r(angles_i)[0])
        
    #make an array
    Ri = np.array(Ri)
    
    #calculate F statistic -- MSb/MSw
    Fww = ((np.sum(Ri) - R)/(q - 1)) / ((n - np.sum(Ri))/(n - q))
    
    #if concentration > 1 (i.e. r > .45), apply a correction
    if grand_r >= 0.45:
        k = concentration(grand_r)
        F = (1. + 3./(8.*k)) * Fww
    else:
        F = Fww
        
    #degrees of freedom
    df1 = q - 1
    df2 = n - q
    
    #calc p-value (one-sided)
    p = 1. - statf.cdf(F, df1, df2)
    
    return F, p, df1, df2



def polar_dot_plot(angles,savefile=None,labels=None,order=None,pretty=None,labelcolors=None):
    ''' make a polar dot plot '''
    
    shift_bins = 60.
    
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