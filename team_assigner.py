#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:03:13 2018

@author: me
"""
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
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

def personNameToBlockedNum(name, block):
    if isinstance(name, int) == True:
        name_num = name
    elif isinstance(name, str) == True:
        name_num = personNameToNum(name)
    else:
        raise TypeError('Name must be a string or an int')
        
    return name_num*num_blocks + block

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
    person_number = math.floor(number/num_blocks)
    block_number = number%num_blocks
    return person_number, block_number

def eventNumToName(event_number):
    if event_number < 0:
        return dummy_event_name
    return event_names[event_number]
        
def eventNameToNum(event_name):
    return event_names.index(event_name)

def eventNumListToNameList(event_number_list):
    name_list = []
    for event_number in event_number_list:
        name_list.append(eventNumToName(event_number))
    return name_list

#takes event and a number identifying if this is the ith person on that event
def eventToSinglePersonEvent(event, event_person_num):
    if isinstance(event, int) == True:
        numerical_event = event
    elif isinstance(event, str) == True:
        numerical_event = eventNameToNum(event)
    else:
        raise TypeError('Event must be a string or an int')
    single_person_event = 0
    for x in range(0, numerical_event):
        single_person_event += people_per_event[x]
    return single_person_event + event_person_num
        
#takes a list of events and turns them into single person events
def eventListToSinglePersonEvents(list_of_events):
    blocked_event_list = []
    for event_iterator in range (0, len(list_of_events)):
        event = list_of_events[event_iterator]
        for person_slot in range(0, people_per_event[event_iterator]):
            blocked_event_list.append(eventToSinglePersonEvent(event, person_slot))
    return blocked_event_list

def singlePersonEventToEvent(single_person_event):
    event_num = 0
    while single_person_event >= people_per_event[event_num]:
        single_person_event -= people_per_event[event_num]
        event_num += 1
    return event_num

#splits scores up into one row per person per block
def splitScoreArray(unblocked_scores):
    num_ppl, num_events = unblocked_scores.shape
    num_single_person_events = len(mono_person_event_list)
    num_blocked_people = num_ppl*num_blocks
    blocked_array = np.full((num_blocked_people,num_single_person_events), -1.0)
    
    block_number = 0  #records which block we're on
    for block in event_conflicts:
        for event in block:
            event_num = eventNameToNum(event) #what column number is this event          
            person_number = 0  #records which person we're on
            for score in getCol(unblocked_scores, event_num):
                mono_person_event_base_column = eventToSinglePersonEvent(event, 0)
                blocked_array[person_number*num_blocks + block_number, 
                              mono_person_event_base_column:mono_person_event_base_column+people_per_event[event_num]] = score
                person_number+= 1
        block_number += 1
    
    return blocked_array

def getTestScore(blocked_person, blocked_event):
    return scores_blocked[blocked_person, blocked_event]

def predictPlace(test_score):
    return 60*(1-test_score)   
        
def scoreTeam(assigned_team):
    sum_test_scores_per_event = np.zeros(num_events)
    for person in range(0, len(assigned_team)):
        persons_events = assigned_team[person]
        for event in persons_events:
            sum_test_scores_per_event[event] += scores[person, event]
    placing = 0
    for x in range(0, len(sum_test_scores_per_event)):
        score = sum_test_scores_per_event[x]/people_per_event[x]
        placing += predictPlace(score)
    return placing

def genRandomTeam(size):
    team_list = random.sample(range(0, num_people), size)
    return team_list

#makes a matrix square
def makeSquare(numpy_array):
    height, width = numpy_array.shape
    if height < width:
        added_array = np.zeros((width-height, width), dtype = numpy_array.dtype)
        square_array = np.concatenate((numpy_array, added_array), axis = 0)
    elif width < height:
        added_array = np.zeros((height, height-width), dtype = numpy_array.dtype)
        square_array = np.concatenate((numpy_array, added_array), axis = 1)
    else:
        square_array = numpy_array
    return square_array

#takes in a list of unassigned team names and creates a graph of them
def assignTeam(team_list):
    blocked_team_list = teamToBlockedNames(team_list)
    
    scores_blocked_of_team = []
    #make array with only the people in the team
    for blocked_person_unit in blocked_team_list:
        scores_blocked_of_team.append(scores_blocked[blocked_person_unit])
    scores_blocked_of_team = np.asarray(scores_blocked_of_team)
    
    num_real_ppl, num_real_events = scores_blocked_of_team.shape
    #expand to square array
    hungarian_matrix = np.negative(makeSquare(scores_blocked_of_team))
    assigned_blocked_ppl, blocked_event_assignments = linear_sum_assignment(hungarian_matrix)
    assigned_team_list = cleanAssignedTeamList(assigned_blocked_ppl, blocked_event_assignments, blocked_team_list, num_real_events)
    return assigned_team_list

def cleanAssignedTeamList(assigned_people, blocked_events, list_blocked_ppl, num_real_events):
    team_assigned = [[] for j in range(len(people_names))]
    num_real_ppl = len(list_blocked_ppl)
    
    for x in range(0, len(assigned_people)):
        assigned_person = assigned_people[x]
        blocked_event = blocked_events[x]
        if assigned_person < num_real_ppl and blocked_event < num_real_events:
            blocked_person = list_blocked_ppl[x]
            person, block = unpackBlockedPersonNum(blocked_person)
            event = singlePersonEventToEvent(blocked_event)
            team_assigned[person].append(event)
    return team_assigned

def getTeamScore(unassigned_team_list):
    return scoreTeam(assignTeam(unassigned_team_list))
#takes in an unassigned list and returns it sorted by least contribution to greatest
def findListOfPersonContributions(team_list):
    people_vs_score_list = []
    for person in team_list:
        team_list_with_person_removed = team_list
        team_list_with_person_removed.remove(person)
        people_vs_score_list.append([person, getTeamScore(team_list)])
    people_vs_score_list.sort(key=lambda x: x[1]) #sorts by score
    return people_vs_score_list

def teamToHumanReadableTeam(clean_assigned_team):
    team_dict = {}
    for person in range(0, len(clean_assigned_team)):
        person_name = personNumToName(person)
        events = eventNumListToNameList(clean_assigned_team[person])
        if len(events) > 0:  #has some events assigned
            team_dict[person_name] = events
    return team_dict

def humanPrintTeam(human_readable_team):
    for person in human_readable_team:
        print(str(person) + ': ' + str(human_readable_team[person]))
        

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
mono_person_event_list = tuple(eventListToSinglePersonEvents(event_names))
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

randTeam = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 25]

assigned_team = assignTeam(randTeam)
print(assigned_team)
humanPrintTeam(teamToHumanReadableTeam(assigned_team))
print(scoreTeam(assigned_team))
print(findListOfPersonContributions(randTeam))

