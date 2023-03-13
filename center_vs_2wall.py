# -*- coding: utf-8 -*-
"""

runs the GLM either using:
    1. center bearing, center distance, head direction, linear speed
    2. 2-wall bearing, 2-wall distance, head direction, linear speed
    3. head direction, linear speed ('no ego' model)
    4. uniform mean firing rate
    
saves results as a pickle file. parameter vectors need to be exponentiated
to turn them into 'response profiles'

one example cell included, recorded in l-shaped environment

NOTE: collect_data function 'calc_wall_ego_lshape' is built specifically to account
for some warping from our camera, so edits will need to be made if used on l-shape data
not from this study

@author: Patrick
"""
import os
import numpy as np
import math
from utilities import collect_data
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from itertools import chain, combinations
import pickle

hd_bins = 30
dist_bins = 10

framerate = 30.


def objective(params,X,spike_train):
    
    ''' compute objective and gradient with optional smoothing (to prevent overfitting) '''
        
    u = X * params
    rate = np.exp(u)
    
    f = np.sum(rate - spike_train * u)
    grad = X.T * (rate - spike_train)
    
    print(f)
    return f,grad


def wall_objective(params,Xw1,Xw2,Xd1,Xd2,Xa,Xs,spike_train):
    
    wall_params = params[:hd_bins]
    dist_params = params[hd_bins:hd_bins+dist_bins]
    allo_params = params[hd_bins+dist_bins:hd_bins*2+dist_bins]
    speed_params = params[hd_bins*2+dist_bins:hd_bins*2+2*dist_bins]

    wall_lamb = (Xd1 * np.exp(dist_params)) * np.exp(Xw1 * wall_params) + (Xd2 * np.exp(dist_params)) * np.exp(Xw2 * wall_params)

    allo_lamb = np.exp(Xa * allo_params)
    speed_lamb = np.exp(Xs * speed_params)
    
    rate = wall_lamb * allo_lamb * speed_lamb
    
    rate[rate<1e-3] = 1e-3
    
    f = np.sum(rate - spike_train*np.log(rate))

    print(f)
    return f


def make_X(trial_data,fdir):
    
    ''' make base design matrix '''
    
    center_ego_angles = np.asarray(trial_data['center_ego_angles'])
    center_ego_dists = np.asarray(trial_data['radial_dists'])
    angles = np.asarray(trial_data['angles'])
    speeds = np.asarray(trial_data['speeds'])

    center_ego_bins = np.digitize(center_ego_angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    center_dist_bins = np.digitize(center_ego_dists,np.linspace(0,np.max(center_ego_dists),dist_bins,endpoint=False))-1
    angle_bins = np.digitize(angles,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    speed_bins = np.digitize(speeds,np.linspace(0,40,dist_bins,endpoint=False)) - 1

    Xe = np.zeros((len(center_ego_angles),hd_bins))
    Xd = np.zeros((len(center_ego_angles),dist_bins))
    Xa = np.zeros((len(center_ego_angles),hd_bins))
    Xs = np.zeros((len(center_ego_angles),dist_bins))

    for i in range(len(angle_bins)):
        Xe[i][center_ego_bins[i]] = 1.
        Xd[i][center_dist_bins[i]] = 1.
        Xa[i][angle_bins[i]] = 1.
        Xs[i][speed_bins[i]] = 1.
        
    X = np.concatenate((Xe,Xd,Xa,Xs),axis=1)
    X = csr_matrix(X)
    
    Xnoego = np.concatenate((Xa,Xs),axis=1)
    Xnoego = csr_matrix(Xnoego)

    return X,csr_matrix(Xe),csr_matrix(Xd),csr_matrix(Xa),csr_matrix(Xs),Xnoego


def make_wall_X_two(trial_data):
    
    ''' make design matrices for two walls '''
    
    all_dists = trial_data['all_wall_dists'].T
    all_bearings = trial_data['all_wall_angles']

    closest_dists = np.take_along_axis(all_dists,np.argsort(all_dists),axis=-1)
    closest_bearings = np.take_along_axis(all_bearings,np.argsort(all_dists),axis=-1)
    
    closest_dists = closest_dists / np.nanmax(closest_dists[:,1])
    
    distance_bins = np.digitize(closest_dists,np.linspace(0,1,dist_bins,endpoint=False)) - 1
    bearing_bins = np.digitize(closest_bearings,np.linspace(0,360,hd_bins,endpoint=False)) - 1
    
    Xw1 = np.zeros((len(distance_bins),hd_bins))
    Xw2 = np.zeros((len(distance_bins),hd_bins))
    
    Xd1 = np.zeros((len(distance_bins),dist_bins))
    Xd2 = np.zeros((len(distance_bins),dist_bins))
    
    for i in range(len(distance_bins)):
        Xw1[i][bearing_bins[i,0]] = 1.
        Xw2[i][bearing_bins[i,1]] = 1.
        
        Xd1[i][distance_bins[i,0]] = 1.
        Xd2[i][distance_bins[i,1]] = 1.

    return csr_matrix(Xw1), csr_matrix(Xw2), csr_matrix(Xd1), csr_matrix(Xd2)
    


def split_data_four(X,Xe,Xd,Xa,Xs,Xw1,Xw2,Xd1,Xd2,spike_train,fold):
    
    ''' split data into 10 parts for our cross-validation '''

    break_points = np.linspace(0,len(spike_train),51).astype(np.int)


    slices = np.r_[break_points[fold]:break_points[fold + 1],break_points[fold + 10]:break_points[fold + 11],break_points[fold + 20]:break_points[fold + 21],
                          break_points[fold + 30]:break_points[fold + 31],break_points[fold + 40]:break_points[fold + 41]]
    
    test_spikes = spike_train[slices]
    test_Xe = csr_matrix(Xe.todense()[slices])
    test_Xd = csr_matrix(Xd.todense()[slices])
    test_Xa = csr_matrix(Xa.todense()[slices])
    test_Xs = csr_matrix(Xs.todense()[slices])
    
    test_Xw1 = csr_matrix(Xw1.todense()[slices])
    test_Xw2 = csr_matrix(Xw2.todense()[slices])

    test_Xd1 = csr_matrix(Xd1.todense()[slices])
    test_Xd2 = csr_matrix(Xd2.todense()[slices])
    
    train_spikes = np.delete(spike_train,slices,axis=0)
    train_X = csr_matrix(np.delete(X.todense(),slices,axis=0))
    train_Xe = csr_matrix(np.delete(Xe.todense(),slices,axis=0))
    train_Xd = csr_matrix(np.delete(Xd.todense(),slices,axis=0))
    train_Xa = csr_matrix(np.delete(Xa.todense(),slices,axis=0))
    train_Xs = csr_matrix(np.delete(Xs.todense(),slices,axis=0))
    
    train_Xw1 = csr_matrix(np.delete(Xw1.todense(),slices,axis=0))
    train_Xw2 = csr_matrix(np.delete(Xw2.todense(),slices,axis=0))
    
    train_Xd1 = csr_matrix(np.delete(Xd1.todense(),slices,axis=0))
    train_Xd2 = csr_matrix(np.delete(Xd2.todense(),slices,axis=0))

    return test_spikes,test_Xe,test_Xd,test_Xa,test_Xs,test_Xw1,test_Xw2,test_Xd1,test_Xd2,train_spikes,train_X,train_Xe,train_Xd,train_Xa,train_Xs,train_Xw1,train_Xw2,train_Xd1,train_Xd2


def run_final(model,scale_factor,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train):
    
    ''' run the model and collect the results '''
    
    if model != 'uniform':
    
        u = np.zeros(len(spike_train))
        
        if 'center_ego' in model:
            u += Xe * center_ego_params
        if 'center_dist' in model:
            u += Xd * center_dist_params
        if 'allo' in model:
            u += Xa * allo_params
        if 'speed' in model:
            u += Xs * speed_params
        
        rate = scale_factor * np.exp(u)
        
    else:
        
        rate = np.full(len(spike_train),np.mean(spike_train))
    
    f = -np.sum(rate - spike_train*np.log(rate))

    
    lgammas = np.zeros(len(spike_train))
    for h in range(len(spike_train)):
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    f -= np.sum(lgammas)
    
    #change from nats to bits
    f = f/np.log(2)

    llps = f/np.sum(spike_train)
    
    
    print('-----------------------')
    print(model)
    print(' ')
    print('log-likelihood: %f' % f)
    print('llps: %f' % llps)
    print(' ')
    print('-----------------------')


    cdict = {}
    #add relevant variables to the appropriate dictionary
    cdict['ll'] = f
    if np.sum(spike_train) > 0:
        cdict['llps'] = float(f/np.sum(spike_train))
    else:
        cdict['llps'] = f
        
    cdict['lambda'] = rate
    cdict['test_spikes'] = spike_train
    cdict['tot_spikes'] = np.sum(spike_train)
    cdict['scale_factor'] = scale_factor
    cdict['center_params'] = center_ego_params
    cdict['dist_params'] = center_dist_params
    cdict['allo_params'] = allo_params
    cdict['speed_params'] = speed_params
    cdict['rate'] = rate

    return cdict

def run_wall_final(model,scale_factor,params,Xw1,Xw2,Xd1,Xd2,Xa,Xs,spike_train):
    
    ''' run the model and collect the results '''
        
    wall_params = params[:hd_bins]
    dist_params = params[hd_bins:hd_bins+dist_bins]
    allo_params = params[hd_bins+dist_bins:hd_bins*2+dist_bins]
    speed_params = params[hd_bins*2+dist_bins:hd_bins*2+2*dist_bins]

    wall_lamb = (Xd1 * np.exp(dist_params)) * np.exp(Xw1 * wall_params) + (Xd2 * np.exp(dist_params)) * np.exp(Xw2 * wall_params)

    allo_lamb = np.exp(Xa * allo_params)
    speed_lamb = np.exp(Xs * speed_params)
    
    rate = wall_lamb * allo_lamb * speed_lamb
    
    rate[rate<1e-3] = 1e-3
    
    f = -np.sum(rate - spike_train*np.log(rate))
    
    lgammas = np.zeros(len(spike_train))
    for h in range(len(spike_train)):
        lgammas[h] = np.log(math.gamma(spike_train[h]+1))
        
    f -= np.sum(lgammas)
    
    #change from nats to bits
    f = f/np.log(2)

    llps = f/np.sum(spike_train)
    
    print('-----------------------')
    print(model)
    print(' ')
    print('log-likelihood: %f' % f)
    print('llps: %f' % llps)
    print(' ')
    print('-----------------------')


    cdict = {}
    #add relevant variables to the appropriate dictionary
    cdict['ll'] = f
    if np.sum(spike_train) > 0:
        cdict['llps'] = float(f/np.sum(spike_train))
    else:
        cdict['llps'] = f
    cdict['lambda'] = rate

    cdict['test_spikes'] = spike_train
    cdict['tot_spikes'] = np.sum(spike_train)
    cdict['scale_factor'] = scale_factor

    cdict['wall_params'] = wall_params
    cdict['dist_params'] = dist_params
    cdict['allo_params'] = allo_params
    cdict['speed_params'] = speed_params
    cdict['rate'] = rate

    return cdict


def get_all_models(variables):
    ''' convenience function for calculating all possible
    combinations of nagivational variables '''
    
    def powerset(variables):
        return list(chain.from_iterable(combinations(variables, r) for r in range(1,len(variables)+1)))
    
    all_models = powerset(variables)
    
    for i in range(len(all_models)):
        all_models[i] = frozenset(all_models[i])
    
    return all_models


if __name__ == '__main__':
    
    variables = [('allo'),('center_ego'),('center_dist'),('speed')]
    all_models = get_all_models(variables)
    
    tracking_fdir = os.getcwd() + '/example_cell l_shape s2'
    cell = 'TT1_SS_10.txt'

    timestamps,center_x,center_y,angles = collect_data.read_video_file(tracking_fdir + '/tracking_data.txt')
    trial_data = {'timestamps':timestamps,'center_x':center_x,'center_y':center_y,'angles':angles}
    spike_timestamps = np.arange(timestamps[0],timestamps[len(timestamps)-1],1000.)

    trial_data['spike_timestamps'] = spike_timestamps
    
    if 'rect' in tracking_fdir:
        gr1 = 20
        gr2 = 10
    elif '1.2m' in tracking_fdir:
        gr1 = 20
        gr2 = 20
    elif '.6m' in tracking_fdir:
        gr1 = 10
        gr2 = 10
    
    save_dir = tracking_fdir + '/wall_vs_center'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    angles=np.array(trial_data['angles'])
    
    if 'l_shape s' in tracking_fdir:
        trial_data = collect_data.calc_center_ego_lshape(trial_data)
        trial_data = collect_data.calc_wall_ego_lshape(trial_data)
    else:
        trial_data = collect_data.ego_stuff(trial_data)
        
    trial_data = collect_data.speed_stuff(trial_data)
        
    #make design matrices
    X,Xe,Xd,Xa,Xs,Xnoego = make_X(trial_data,[])
    Xw1,Xw2,Xd1,Xd2 = make_wall_X_two(trial_data)

    fname = tracking_fdir+ '/' + cell
    cluster_data={}
    cluster_data['spike_list'] = collect_data.ts_file_reader(fname)
    spike_data,cluster_data = collect_data.create_spike_lists(trial_data,cluster_data)
    spike_train = spike_data['ani_spikes']
    
    savefile = save_dir+'/%s_cdict.pickle' % cell

    params = np.zeros(hd_bins*2 + dist_bins*2)
    result = minimize(wall_objective,params,args=(Xw1,Xw2,Xd1,Xd2,Xa,Xs,spike_train),method='L-BFGS-B')
    wall_base_cdict = run_wall_final([],1.,result.x,Xw1,Xw2,Xd1,Xd2,Xa,Xs,spike_train)
    
    params = np.zeros(np.shape(X)[1])
    result = minimize(objective,params,args=(X,spike_train),jac=True,method='L-BFGS-B')
    params = result.x
    center_ego_params = params[:hd_bins]
    center_dist_params = params[hd_bins:(hd_bins+dist_bins)]
    allo_params = params[(hd_bins+dist_bins):(2*hd_bins+dist_bins)]
    speed_params = params[(2*hd_bins+dist_bins):]
    center_base_cdict = run_final(frozenset(variables),1.,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)

    params = np.zeros(np.shape(Xnoego)[1])
    result = minimize(objective,params,args=(Xnoego,spike_train),jac=True,method='L-BFGS-B')
    params = result.x
    allo_params = params[:hd_bins]
    speed_params = params[hd_bins:]
    noego_base_cdict = run_final(frozenset([('allo'),('speed')]),1.,[],[],allo_params,speed_params,[],[],Xa,Xs,spike_train)
    

    uniform_base_cdict = run_final('uniform',1.,center_ego_params,center_dist_params,allo_params,speed_params,Xe,Xd,Xa,Xs,spike_train)

    base_nspikes = np.sum(spike_train)

    cdict = {}
    cdict['wall_base'] = wall_base_cdict
    cdict['center_base'] = center_base_cdict
    cdict['uniform_base'] = uniform_base_cdict
    cdict['noego_base'] = noego_base_cdict
    cdict['base_nspikes'] = base_nspikes
    
    with open(savefile,'wb') as f:
        pickle.dump(cdict,f,protocol=2)
