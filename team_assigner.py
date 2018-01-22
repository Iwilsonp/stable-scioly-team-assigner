#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:03:13 2018

@author: me
"""
import numpy as np
import math
import networkx as nx
from networkx.algorithms import bipartite
import random
import csv
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
def checkInputData(scores, max_scores):
    num_ppl, num_events = scores.shape
    if(len(people_per_event) != num_events):
        raise ValueError('number of events mismatch between people per event list and score array')
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

def strListToNumList(str_list, num_type = int):
    return_list = []
    for string in str_list:
        if isinstance(string, list) == True:
            return_list.append(strListToNumList(string, num_type = num_type))
        else:
            if num_type == float:
                return_list.append(float(string))
            elif num_type == int:
                return_list.append(int(string))
    return return_list
        
      
def personNumToName(person_number):
    if person_number < 0:
        return dummy_person_name
    return people_names[person_number]

def personNameToNum(person_name):
    return people_names.index(person_name)

def getMinBlockedPersonNum():
    return len(event_names) * max(people_per_event)

def personNameToBlockedNum(name, block):
    if isinstance(name, int) == True:
        name_num = name
    elif isinstance(name, str) == True:
        name_num = personNameToNum(name)
    else:
        raise TypeError('Name must be a string or an int')
        
    return name_num*num_blocks + block + getMinBlockedPersonNum()

#turns a whole team list into a blocked list
def teamToBlockedNames(numerical_team):        
    blocked_team_list = []
    for member in numerical_team:
        for block in range(0, num_blocks):
            blocked_team_list.append(personNameToBlockedNum(member, block))
    return blocked_team_list

def unpackBlockedPersonNum(number):
    if number < 0:  #dummy person
        return -1, 0
    number = number - getMinBlockedPersonNum()
    person_number = math.floor(number/num_blocks)
    block_number = number%num_blocks
    return person_number, block_number

def eventNumToName(event_number):
    if event_number < 0:
        return dummy_event_name
    return event_names(event_number)
        
def eventNameToNum(event_name):
    return event_names.index(event_name)

#takes event and a number identifying if this is the ith person on that event
def eventToSinglePersonEvent(event, event_person_num):
    if isinstance(event, int) == True:
        return event*max_people_per_event + event_person_num
    elif isinstance(event, str) == True:
        return eventNameToNum(event)*max_people_per_event + event_person_num
    else:
        raise TypeError('Event must be a string or an int')
        
#takes a list of events and turns them into single person events
def eventListToSinglePersonEvents(list_of_events):
    blocked_event_list = []
    for event_iterator in range (0, len(list_of_events)):
        event = list_of_events[event_iterator]
        for person_slot in range(0, people_per_event[event_iterator]):
            blocked_event_list.append(eventToSinglePersonEvent(event, person_slot))
    return blocked_event_list

def singlePersonEventToEvent(single_person_event):
    return math.floor(single_person_event/max_people_per_event)

#splits scores up into one row per person per block
def splitScoreArray(unblocked_scores):
    num_ppl, num_events = unblocked_scores.shape
    blocked_array = np.full((num_ppl*num_blocks, num_events*max_people_per_event), -1.0)
    
    block_number = 0  #records which block we're on
    for block in event_conflicts:
        for event in block:
            event_num = eventNameToNum(event) #what column number is this event          
            person_number = 0  #records which person we're on
            for score in getCol(unblocked_scores, event_num):
                mono_person_event_base_column = eventToSinglePersonEvent(event, 0)
                blocked_array[person_number*num_blocks + block_number, 
                              mono_person_event_base_column:mono_person_event_base_column+max_people_per_event] = score
                person_number+= 1
        block_number += 1
    
    return blocked_array

def getTestScore(blocked_person, blocked_event):
    if blocked_person < 0 or blocked_event < 0:  #fake person or event
        return 0
    else:
        return scores_blocked[blocked_person - getMinBlockedPersonNum(), blocked_event]

#fill out a list with dummy items for Hungarian algo
def makeListAsLongAs(short_list, long_list, starting_val = -1):
    fake_thing_appended = starting_val
    while len(short_list) < len(long_list):
        short_list.append(fake_thing_appended)
        fake_thing_appended -= 1
    return short_list

def genRandomTeam(size):
    team_list = random.sample(range(0, num_people), size)
    return team_list

#takes in a list of unassigned team names and creates a graph of them
def createGraphFromTeam(team_list):
    newTeamGraph = nx.Graph()
    blocked_team_list = teamToBlockedNames(team_list)
    
    local_mono_person_event_list = list(mono_person_event_list)  #local copy
    #fill up lists with dummies if needed
    blocked_team_list =  makeListAsLongAs(blocked_team_list, local_mono_person_event_list)
    local_mono_person_event_list = makeListAsLongAs(local_mono_person_event_list, blocked_team_list)

    newTeamGraph.add_nodes_from(blocked_team_list, bipartite=0)
    newTeamGraph.add_nodes_from(local_mono_person_event_list, bipartite=1)
    
    for person in blocked_team_list:
        for mono_event in local_mono_person_event_list:
            test_score = getTestScore(person, mono_event)
            newTeamGraph.add_edge(person, mono_event, weight=test_score)
    return newTeamGraph, blocked_team_list
    
   
def assignTeam(graph_of_team, blocked_team_list):
    messy_team = bipartite.matching.maximum_matching(graph_of_team, top_nodes = blocked_team_list)
    return filterAssignedTeamDict(messy_team)

def predictPlace(test_score):
    return math.sqrt(test_score)

#eliminates fake people and duplicate matchings. 
#Returns dict with blocked people matched to blocked events.
def filterAssignedTeamDict(messy_team_dict):
    assigned_team_dict = {}
    for node in messy_team_dict:
        matched_node = messy_team_dict[node]
        if node > 0 and matched_node > 0: #not a fake person or event
            if node > getMinBlockedPersonNum(): #it's a person and not an event
                assigned_team_dict[node] = matched_node
    return assigned_team_dict

def evalTeam(assigned_team):
    event_test_scores = np.zeros(len(event_names))
    for blocked_person in assigned_team:
        test_score = scores_blocked
    
    
def assignedTeamToHumanReadableTeam(assigned_team):
    #initialize empty list of lists. Each entry corresponds to a person in the person list
    humanTeam = [[] for i in range(num_people)]
    for person in assigned_team:
        person_num, block = unpackBlockedPersonNum(person)
        single_person_event = assigned_team[person]#a real event, not null
        event = singlePersonEventToEvent(single_person_event)
        humanTeam[person_num].append(event)
        
    return humanTeam
        
    
dummy_event_name = 'DUMMY_EVENT'
dummy_person_name = 'DUMMY_PERSON'    
        
#read in data
first_column = []
prelim_data = []
with open('prelim_results.csv') as prelim_test_file:
    prelim_test_reader = csv.reader(prelim_test_file)
    for row in prelim_test_reader:
        first_column.append(row[0])
        prelim_data.append(row[1:])

people_names = tuple(first_column[4:])
event_names = tuple(prelim_data[0])
people_per_event = tuple(strListToNumList(prelim_data[1]))
event_weight = tuple(strListToNumList(prelim_data[3], num_type = float))

max_prelim_test_scores = tuple(strListToNumList(prelim_data[2], num_type = float))
raw_prelim_test_scores = strListToNumList(prelim_data[4:], num_type = float)
team_size = 15

#begin processing
num_people = len(people_names)
max_person_num = num_people - 1
num_events = len(event_names)
num_blocks = len(event_conflicts)
max_people_per_event = max(people_per_event)

np_raw_prelim_test_scores = np.asarray(raw_prelim_test_scores) #convert to numpy array

checkInputData(np_raw_prelim_test_scores, max_prelim_test_scores)

processed_prelim_test_scores = normalizeData(np_raw_prelim_test_scores, max_prelim_test_scores)

#generate combo array of all data
scores = processed_prelim_test_scores

#split people into blocks for Hungarian algorithim processing
scores_blocked = splitScoreArray(scores)
mono_person_event_list = tuple(eventListToSinglePersonEvents(event_names))

randTeam = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 25]
print(randTeam)
graph, people = createGraphFromTeam(randTeam)
team = assignTeam(graph, people)
print(team)
print(len(team))
human_readable_team = assignedTeamToHumanReadableTeam(team)
print(human_readable_team)
print(recursive_len(human_readable_team))

