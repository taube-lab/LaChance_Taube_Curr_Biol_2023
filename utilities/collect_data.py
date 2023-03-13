# -*- coding: utf-8 -*-
"""

spatial/admin stuff for glm

@author: Patrick
"""

import csv
import bisect
import numpy as np

adv = {}
adv['framerate'] = 30.

ops = {}
ops['acq'] = 'neuralynx'

 
def read_video_file(video_file):
    
    timestamps = []
    xpos = []
    ypos = []
    hdangles = []
    
    with open(video_file,'r') as f:
        reader = csv.reader(f,dialect='excel-tab')
        for row in reader:
            timestamps.append(float(row[0]))
            xpos.append(float(row[1]))
            ypos.append(float(row[2]))
            hdangles.append(float(row[3]))
            
    return np.array(timestamps), np.array(xpos), np.array(ypos), np.array(hdangles)

def ego_stuff(trial_data):
    ''' calculate ego bearings/distances for center and walls '''
    
    trial_data = calc_center_ego(trial_data)
    trial_data = calc_wall_ego(trial_data)
    
    return trial_data
        
def speed_stuff(trial_data):
    ''' calculate speed and ahv info from tracking data '''
    
    trial_data['speeds'] = []
    
    print('processing speed data...')
    #calculate running speeds for each frame
    trial_data = calc_speed(trial_data)
        
    return trial_data

def calc_speed(trial_data):
    """calculates 'instantaneous' linear speeds for each video frame"""
    
    #grab appropriate tracking data
    center_x=np.array(trial_data['center_x'])
    center_y=np.array(trial_data['center_y'])
    
    #make an array of zeros to assign speeds to
    speeds = np.zeros(len(center_x),dtype=np.float)
    #for every frame from 2 to total - 2...
    for i in range(2,len(center_x)-2):
        #grab 5 x and y positions centered on that frame
        x_list = center_x[(i-2):(i+3)]
        y_list = center_y[(i-2):(i+3)]
        #find the best fit line for those 5 points (slopes are x and y components
        #of velocity)
        xfitline = np.polyfit(range(0,5),x_list,1)
        yfitline = np.polyfit(range(0,5),y_list,1)
        #total velocity = framerate * sqrt(x component squared plus y component squared)
        speeds[i] = adv['framerate']*np.sqrt(xfitline[0]**2 + yfitline[0]**2)
    #set unassigned speeds equal to closest assigned speed
    speeds[0] = speeds[2]
    speeds[1] = speeds[2]
    speeds[len(speeds)-1] = speeds[len(speeds)-3]
    speeds[len(speeds)-2] = speeds[len(speeds)-3]
    
    #return calculated speeds
    trial_data['speeds'] = speeds
    return trial_data


def calc_center_ego(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2

    radial_dists = np.sqrt(center_x**2 + center_y**2)
    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_ego_angles = (center_ego_angles-angles)%360
    
    trial_data['radial_dists'] = radial_dists
    trial_data['center_ego_angles'] = center_ego_angles
    
    return trial_data


def calc_wall_ego(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= (np.max(center_x) - np.min(center_x))/2
    center_y -= (np.max(center_y) - np.min(center_y))/2
    
    w1 = center_x - (np.min(center_x)-1.)
    w2 = -(center_x - (np.max(center_x)+1.))
    w3 = center_y - (np.min(center_y)-1.)
    w4 = -(center_y - (np.max(center_y)+1.))
    
    all_dists = np.stack([w1,w2,w3,w4])
    wall_dists = np.min(all_dists,axis=0)
    wall_ids = np.argmin(all_dists,axis=0)
    wall_angles = np.array([180.,0.,270.,90.])
    wall_ego_angles = wall_angles[wall_ids]
    wall_ego_angles = (wall_ego_angles - angles)%360
    
    trial_data['wall_dists'] = wall_dists
    trial_data['wall_ego_angles'] = wall_ego_angles
    
    all_wall_angles = (np.stack([wall_angles]*len(angles)) - np.stack((angles,angles,angles,angles)).T)%360
    
    trial_data['all_wall_dists'] = all_dists
    trial_data['all_wall_angles'] = all_wall_angles
    
    
    return trial_data


def calc_wall_ego_lshape(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= np.min(center_x)
    center_y -= np.min(center_y)
    center_x -= 60.
    center_y -= 60.
    
    
    #left wall
    w1 = center_x - (np.min(center_x)-1.)
    w1_angles = np.tile(180.,len(center_x))
    #bottom wall
    w3 = center_y - (np.min(center_y)-1.)
    w3_angles = np.tile(270.,len(center_x))


    #rightmost wall    
    w2 = -(center_x - (np.max(center_x)+1.))
    w2_angles = np.tile(0.,len(center_x))
    w2[center_y > 0.] = np.sqrt((-(center_x[center_y > 0.] - (np.max(center_x)+1.)))**2 + (-(center_y[center_y > 0.] - 1.))**2)
    w2_angles[center_y>0.] = np.rad2deg(np.arctan2(-(center_y[center_y > 0.] - 1.),-(center_x[center_y > 0.] - (np.max(center_x)+1.))))
    
    #topmost wall
    w4 = -(center_y - (np.max(center_y)+1.))
    w4_angles = np.tile(90.,len(center_x))
    w4[center_x > 0.] = np.sqrt((-(center_y[center_x > 0.] - (np.max(center_y)+1.)))**2 + (-(center_x[center_x > 0.] - 1.))**2)
    w4_angles[center_x > 0.] = np.rad2deg(np.arctan2(-(center_y[center_x > 0.] - (np.max(center_y)+1.)), -(center_x[center_x > 0.] - 1.)))

    
    #mid vertical wall    
    #center y < 5 instead of < 0 bc of warping from camera
    w5 = abs(center_x - 1.)
    w5_angles = np.tile(0.,len(center_x))
    w5[center_y < 5.] = np.sqrt((-(center_x[center_y < 5.] - 1.))**2 + (-(center_y[center_y < 5.] - 6.))**2)
    w5_angles[center_y < 5.] = np.rad2deg(np.arctan2(-(center_y[center_y < 5.] - 6.), -(center_x[center_y < 5.] - 1.)))
    #only take samples where the wall is "accessible" to the rat -- e.g., when the rat could go in a straight line to the wall
    #without hitting any other walls (in this case, when center_x < 0) - but this actually doesn't matter for the 2-wall model
    w5[center_x > 0.] = np.nan
    w5_angles[center_x > 0.] = np.nan
    
    #mid horizontal wall
    #-6 instead of -1 because of warping from the camera
    w6 = abs(center_y - 6.)
    w6_angles = np.tile(90.,len(center_x))
    w6[center_x < 0.] = np.sqrt((-(center_y[center_x < 0.] - 6.))**2 + (-(center_x[center_x < 0.] - 6.))**2)
    w6_angles[center_x < 0.] = np.rad2deg(np.arctan2(-(center_y[center_x < 0.] - 6.), -(center_x[center_x < 0.] - 6.)))
    #only take samples where the wall is "accessible" to the rat -- e.g., when the rat could go in a straight line to the wall
    #without hitting any other walls (in this case, when center_y < 0) - but this actually doesn't matter for the 2-wall model
    w6[center_y > 5.] = np.nan
    w6_angles[center_y > 5.] = np.nan

    all_dists = np.stack([w1,w2,w3,w4,w5,w6])
    all_wall_angles = (np.stack((w1_angles,w2_angles,w3_angles,w4_angles,w5_angles,w6_angles)).T - np.stack((angles,angles,angles,angles,angles,angles)).T)%360
    
    trial_data['all_wall_dists'] = all_dists
    trial_data['all_wall_angles'] = all_wall_angles
    
    wall_dists = np.nanmin(all_dists,axis=0)
    wall_ids = np.nanargmin(all_dists,axis=0)
    wall_ego_angles = np.zeros_like(wall_dists)
    for i in range(len(wall_dists)):
        wall_ego_angles[i] = all_wall_angles[i,wall_ids[i]]
    
    trial_data['wall_dists'] = wall_dists
    trial_data['wall_ego_angles'] = wall_ego_angles%360
    
    
    return trial_data


def calc_center_ego_lshape(trial_data):
    
    center_x = np.array(trial_data['center_x'])
    center_y = np.array(trial_data['center_y'])
    angles = np.array(trial_data['angles'])
    
    center_x -= 52.5
    center_y -= 52.5

    radial_dists = np.sqrt(center_x**2 + center_y**2)
    center_ego_angles = (np.rad2deg(np.arctan2(-center_y,-center_x)))%360
    center_ego_angles = (center_ego_angles-angles)%360
    
    trial_data['radial_dists'] = radial_dists
    trial_data['center_ego_angles'] = center_ego_angles
    
    return trial_data

def ts_file_reader(ts_file):
    """reads the spike ASCII timestamp file and assigns timestamps to list"""
    
    #make a list for spike timestamps
    spike_list = []
    #read txt file, assign each entry to spike_list
    reader = csv.reader(open(ts_file,'r'),dialect='excel-tab')

    for row in reader:
        spike_list.append(int(row[0]))
                
    #return it!
    return spike_list

def create_spike_lists(trial_data,cluster_data):
    """makes lists of spike data"""

    #dictionary for spike data
    spike_data = {}

    #creates array of zeros length of spike_timestamps to create spike train
    spike_train = np.zeros(len(trial_data['spike_timestamps']))
    #array of zeros length of video timestamps for plotting/animation purposes
    ani_spikes = np.zeros(len(trial_data['timestamps']),dtype=np.int)
    timestamps = np.array(trial_data['timestamps'])
    #for each spike timestamp...
    for i in cluster_data['spike_list']:
        diff = np.abs(timestamps-i)
        ani_spikes[np.where(diff==np.min(diff))] += 1

        #find closest entry in high precision 'spike timestamps' list
        spike_ind = bisect.bisect_left(trial_data['spike_timestamps'],i)

        if spike_ind < len(spike_train):
            #add 1 to spike train at appropriate spot
            spike_train[spike_ind] += 1
    #find the video timestamp at the halfway point
    halfway_ind = bisect.bisect_left(cluster_data['spike_list'],trial_data['timestamps'][np.int(len(trial_data['timestamps'])/2)]) - 1

    spike_data['ani_spikes'] = ani_spikes
    spike_data['spike_train'] = spike_train
    cluster_data['halfway_ind'] = halfway_ind
    
    return spike_data, cluster_data