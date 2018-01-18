#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:03:13 2018

@author: me
"""
import numpy as np
import math
import networkx as nx
import networkx.algorithms.matching as matching
import random
import candidate_names

event_names = ['Anatomy and Physiology','Astronomy','Chemistry Lab','Disease Detectives','Dynamic Planet','Ecology','Experimental Design','Fermi Questions','Forensics','Game On','Helicopters','Herpetology','Hovercraft','Materials Science','Microbe Mission','Mission Possible','Mousetrap Vehicle','Optics','Remote Sensing','Rocks and Minerals','Thermodynamics','Towers','Write It Do It']
people_per_event = [2,2,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
#built assuming we are team C-38. All self-schedule events given their own block
event_conflicts = [['Disease Detectives','Fermi Questions'],
                   ['Anatomy and Physiology','Dynamic Planet','Rocks and Minerals'],
                   ['Chemistry Lab','Ecology','Remote Sensing'],
                   ['Herpetology','Optics'],
                   ['Astronomy','Game On','Microbe Mission'],
                   ['Experimental Design','Forensics'],
                   ['Materials Science','Thermodynamics', 'Write It Do It'],
                   ['Helicopters'],
                   ['Hovercraft'],
                   ['Mission Possible'],
                   ['Mousetrap Vehicle'],
                   ['Towers']]
people_names = candidate_names.names  #keep real names out of git repo

team_size = 15

max_prelim_test_scores = [190,132,150,162,100,160,105,500,200,98,56.83,133,100,100,120,750.6,0,100,135,140,100,1291.71,175]
#columns are events, rows are people
raw_prelim_test_scores = [[0,0,0,0,0,0,0,174,110,0,0,0,15.77319588,63,0,0,0,0,0,0,35.0257732,0,119],
[68,0,0,0,0,0,0,136,0,70,0,0,0,21.5,0,100,49.3,0,0,0,0,0,132],
[0,0,0,0,0,56,0,0,0,0,0,0,0,0,51.5,0,0,60.38918919,54,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,56.83,0,0,39,0,0,0,0,0,0,0,499.66,141.5],
[0,39,0,69.5,0,0,0,158,0,0,0,33.5,0,0,0,575.5,0,0,48.5,0,0,0,0],
[61.5,0,0,0,0,0,0,0,104,0,17.64,47.5,0,0,54.5,0,551.1,0,0,0,24.3556701,675.5,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,23.5,0,0,0,0,61,51,0,670.81,0],
[0,0,0,0,0,80.5,0,0,0,0,0,72.5,0,49,0,0,0,0,0,49,0,0,0],
[82,0,0,0,0,0,0,152,0,0,0,0,0,0,0,0,0,68.13513514,0,0,30.6185567,0,0],
[0,0,72.5,82.5,0,0,0,169,0,0,0,0,32.4742268,0,0,0,0,82.7,0,0,39.89690722,0,0],
[99.5,0,0,0,0,82,0,0,135.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,46,0,0,0,0,162,0,0,0,0,0,43,0,0,0,0,0,45,0,0,139.5],
[0,0,0,0,0,0,69.5,213,0,0,20.93,0,0,0,0,750.6,0,0,0,0,0,1291.71,122.5],
[0,0,0,94.5,0,0,62.5,142,0,68,0,0,23.65979381,0,0,0,0,0,0,0,0,0,128.5],
[0,85,0,0,0,80,0,174,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,65,53.21,0,0,0,0,0,0,45.35675676,0,0,0,0,141],
[108.5,0,0,95,0,76,0,247,66,0,0,52,0,0,75.5,0,198.3,0,0,0,0,0,0],
[112,0,0,0,0,0,0,0,0,0,0,0,0,0,73.5,0,0,0,0,26.5,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[131.5,0,81.5,0,0,0,0,0,139,0,0,58.5,0,0,60.5,0,0,0,0,0,0,0,8],
[0,0,0,0,30,0,0,130,116.5,0,0,0,0,51.5,0,0,0,0,0,50,0,0,0],
[0,0,66.5,77.5,0,0,57.5,0,0,0,0,0,0,56.5,0,0,0,0,0,0,72.78618557,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,99.433,94.75,0,750.6,0,0,0,0,45,0,0],
[0,0,0,0,64,0,0,153,0,0,0,51,0,42.5,0,0,0,0,80,0,0,0,122.5],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[55,0,0,0,49,0,0,0,0,0,0,0,0,45.5,0,0,0,0,0,41,87.57731959,0,0],
[110.25,0,0,0,60,79,0,0,0,0,0,0,0,0,73.5,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,57,0,0,0,0,63,0,0,76.23712371,0,0,0,29.31,0,0,0,0,0,0]]

def checkInputData(scores, max_scores):
    num_ppl, num_events = scores.shape
    if(len(people_per_event) != num_events):
        raise ValueError('number of events mismatch between people per event list and max score array')
    if(len(event_names) != num_events):
        raise ValueError('number of events mismatch between event name list and max score list')
    if(len(max_scores) != num_events):
        raise ValueError('number of events mismatch between max score list and score array')
    if(recursive_len(event_conflicts) != num_events):
        raise ValueError('number of events mismatch between event conflict list and score array')
    
    if(len(people_names) != num_ppl):
        raise ValueError('number of people mismatch between name list and score array')

def recursive_len(item):  #total items in list of lists. From https://stackoverflow.com/questions/27761463/how-can-i-get-the-total-number-of-elements-in-my-arbitrarily-nested-list-of-list
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
       
def getColMax(np_array, column):
    return np.max(np_array[:,column])

def getCol(np_array, column):
    return np_array[:, column]
    
def normalizeData(scores, max_scores):
    normalized_array = np.full(scores.shape, 0.0)
    
    for event in range(0, len(max_scores)):
        if max_scores[event] == 0:   #low score wins event
            participation_ability = 0.3 #if worst person on event
            max_score = getColMax(scores, event) * (1 + participation_ability)
            #normalize with the highest score getting 0
            num_people = len(getCol(scores, event))
            for person in range(0, num_people):
                if scores[person, event] == 0:
                    normalized_array[person, event] = 0 #didn't participate
                else:
                    #ability goes from participation (weakest person) to 1 (perfect 0 score)
                    normalized_array[person, event] = 1 - scores[person, event]/max_score
        else:  #normal event
            normalized_array[:,event] = getCol(scores, event)/max_scores[event]
    return normalized_array
       
def personNumToName(person_number):
    return people_names[person_number]

def blockedPersonNumToName(person_number):
    person_index = math.floor(event_conflicts/num_blocks)
    return people_names[person_index]

def personNameToNum(person_name):
    return people_names.index(person_name)

def personNameToBlockedNum(person_name):
    return people_names.index(person_name*num_blocks)

def eventNumToName(event_number):
    return event_names(event_number)
        
def eventNameToNum(event_name):
    return event_names.index(event_name)

def splitScoreArray(unblocked_scores):  #splits scores up into one row per person per block
    num_ppl, num_events = unblocked_scores.shape
    blocked_array = np.full((num_ppl*num_blocks, num_events), 0.0)
    
    block_number = 0  #records which block we're on
    for block in event_conflicts:
        for event in block:
            event_num = eventNameToNum(event) #what column number is this event
            
            person_number = 0  #records which person we're on
            for score in getCol(unblocked_scores, event_num):
                blocked_array[person_number*num_blocks + block_number, event_num] = score
                person_number+= 1
        block_number += 1
    
    return blocked_array

def genRandomTeam():
    team_list = random.sample(range(0, num_people), team_size)
    return team_list.sort()
   
def numericalTeamToBlockedNames(numerical_team):
    if !isinstance(numerical_team[0], int):
        raise TypeError('Team must be ints')
        
    blocked_team_list = []
    for member in numerical_team:
        member_name = personNameToNum(member)
        for block in range(0, num_blocks):
            blocked_team_list.append(member_name + '_block_' + str(block))
    return blocked_team_list
    
def assignTeam(unassigned_numerical_team_list, blocked_score_list):
    B = nx.Graph() #create bipartate graph
    size_of_hungarian_array = max(len(unassigned_numerical_team_list)*num_blocks, num_events)
    
    blocked_team_list = numericalTeamToBlockedNames(unassigned_numerical_team_list)
    
    #expand team list to required length
    ieterator = 0
    while(len(blocked_team_list) < size_of_hungarian_array):
        blocked_team_list.append('NULL_PERSON_' + str(ieterator))
        
    
    B.add_nodes_from([1,2,3,4], bipartite=0)
    for person_num in range(0, size_of_hungarian_array):
        for event_num in range(0, size_of_hungarian_array):
            
    
    
        
#begin processing
num_people = len(people_names)
max_person_num = num_people - 1
num_events = len(event_names)
num_blocks = len(event_conflicts)
raw_prelim_test_scores = np.asarray(raw_prelim_test_scores) #convert to numpy array

checkInputData(raw_prelim_test_scores, max_prelim_test_scores)

processed_prelim_test_scores = normalizeData(raw_prelim_test_scores, max_prelim_test_scores)

#generate combo array of all data
scores = processed_prelim_test_scores

#split people into blocks for Hungarian algorithim processing
scores_blocked = splitScoreArray(scores)




