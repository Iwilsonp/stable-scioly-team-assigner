#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:03:13 2018

@author: me
"""
try:
    import numpy as np
except ImportError:
    raise ImportError('You must have Numpy installed')
import math
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError('You must have Scipy installed')
import random
import csv
import copy
import time
import sys
import pickle

#sorted list of teams
file_to_save_to = 'team_config.txt'
csv_of_people_who_must_be_on = 'forced_on_people.csv'
event_conflicts_csv = 'event_conflicts.csv'

list_of_files = sys.argv[1:]
if list_of_files == []:
    print('No score files provided on the command line, using examples')
    list_of_files = ['example_scores.csv']

team_size = 15
max_num_seniors = 7

dummy_person_name = 'FAKE PERSON'
dummy_person_number = -1
invalid_team_score = 1000000

def predictPlace(test_score):
    return 60*(1-math.sqrt(test_score))   

def multiEventPenalty(events_one_person):
    return 0

def isBool(test_bool):
    return (test_bool == 0 or test_bool == 1)

def isNumber(number):
    try:
        float(number)
        return True
    except ValueError:
        return False
    
def removeFromList(the_list, value):
    return_list = []
    for element in the_list:
        if isinstance(element, list):
            return_list.append(removeFromList(element, value))
        else:
            if element != value:
                return_list.append(element)
    return return_list      

def checkInputData(scores, max_scores, people_per_event, event_names, people_names, is_senior):
    num_ppl, num_events = scores.shape
    if(len(people_per_event) != num_events):
        raise ValueError('number of events mismatch between people per event list and score array')
    if(len(event_names) != num_events):
        raise ValueError('number of events mismatch between event name list and max score list')
    if(len(max_scores) != num_events):
        raise ValueError('number of events mismatch between max score list and score array')
    if(recursive_len(event_conflicts) != num_events):
        print(recursive_len(event_conflicts))
        raise ValueError('number of events mismatch between event conflict list and score array')
    if(len(people_names) != num_ppl):
        raise ValueError('number of people mismatch between name list and score array')
    if(len(is_senior)!=num_ppl):
        raise ValueError('number of people is not the same as the length of the senior bool list')
    for senior_status in is_senior:
        if isBool(senior_status) == False:
            raise ValueError('numbers in senior status column must be 0 or 1, not ' + str(senior_status))

def checkListEquality(list1, list2):
    if list1 == list2:
        return True
    else:
        return False

def flattenList(list_of_lists):
    return_list = []
    for element in list_of_lists:
        if isinstance(element, list):
            return_list = return_list + flattenList(element)
        else:
            return_list.append(element)
    return return_list
    
def checkUnsortedListEquality(list1, list2):
    print(flattenList(list1))
    flat_list_1 = sorted(flattenList(list1))
    flat_list_2 = sorted(flattenList(list2))
    return flat_list_1 == flat_list_2
       
def checkAllAreEqual(list_of_data):
    if len(list_of_data) == 1:
        return True #only one data point in the list
    for x in range(0, len(list_of_data) - 1):
        if checkListEquality(list_of_data[x], list_of_data[x+1]) == False:
            return False
    return True 

def recursive_len(item):  #total items in list of lists. From https://stackoverflow.com/questions/27761463/how-can-i-get-the-total-number-of-elements-in-my-arbitrarily-nested-list-of-list
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
       
def getColMax(np_array, column):
    return np.max(np_array[:,column])

def getCol(np_array, column):
    try:
        return np_array[:, column]
    except TypeError:
        return[row[column] for row in np_array]
    
def normalizeData(scores, max_scores):
    normalized_array = np.full(scores.shape, 0.0)
    
    for event in range(0, len(max_scores)):
        if max_scores[event] == 0:   #low score wins event
            participation_ability = 0.1 #if worst person on event
            max_score = getColMax(scores, event) * (1 + participation_ability)
            #normalize with the highest score getting 0
            num_people = len(getCol(scores, event))
            for person in range(0, num_people):
                if scores[person, event] == 0:
                    normalized_array[person, event] = 0 #didn't participate
                else:
                    #ability goes from participation (weakest person) to 1 (perfect 0 score)
                    #this array call cannot be replaced by getTestScore
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
    try:
        if person_number < 0 or person_number >=num_people:
            return dummy_person_name
        else:
            return people_names[person_number]
    except NameError:   #num_people not defined
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
            blocked_person_num = personNameToBlockedNum(member, block)
            if blocked_person_num in person_block_matching:
                blocked_team_list.append(blocked_person_num)
    return blocked_team_list

def unpackBlockedPersonNum(number):
    if number < 0:  #dummy person
        return -1, 0
    person_number = math.floor(number/num_blocks)
    block_number = number%num_blocks
    return person_number, block_number

def eventNumToName(event_number):
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
           blocked_person_number = eventToSinglePersonEvent(event, person_slot)
           if blocked_person_number in person_block_matching: #we didn't get rid of this row from the score array
               blocked_event_list.append(blocked_person_number)
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
    num_single_person_events = sum(people_per_event)
    num_blocked_people = num_ppl*num_blocks
    blocked_array = np.full((num_blocked_people,num_single_person_events), 0.0)
    person_per_row_in_score_array = np.zeros(num_blocked_people)
    block_number = 0  #records which block we're on
    for block in event_conflicts:
        for event in block:
            event_num = eventNameToNum(event) #what column number is this event          
            person_number = 0  #records which person we're on
            for score in getCol(unblocked_scores, event_num):
                person_per_row_in_score_array[personNameToBlockedNum(person_number, block_number)] = personNameToBlockedNum(person_number, block_number)
                mono_person_event_base_column = eventToSinglePersonEvent(event, 0)
                blocked_array[personNameToBlockedNum(person_number, block_number), 
                              mono_person_event_base_column:mono_person_event_base_column+people_per_event[event_num]] = score
                person_number+= 1
        block_number += 1
    
    #filter score array. from stackoverflow.
    blocks_with_nothing_in_them = np.where(~blocked_array.any(axis=1))[0]

    blocked_people_with_nonzero_scores = [x for i,x in enumerate(person_per_row_in_score_array) if i not in blocks_with_nothing_in_them]
    return blocked_array, blocked_people_with_nonzero_scores

def isSenior(person):
    if isinstance(person, str):
        person = personNameToNum(person)
    if who_are_seniors[person] == 1:
        return True
    elif who_are_seniors[person] == 0:
        return False
    else:
        raise ValueError('person not a 0 or a 1, is a ' + str(person))     

def getNumSeniors(team_list):
    if isAssigned(team_list):
        team_list = unassignTeam(team_list)
    num_seniors = 0
    for person in team_list:
        if isSenior(person):
            num_seniors += 1
    return num_seniors

def numSeniorsOK(team_list):
    if getNumSeniors(team_list) > max_num_seniors:
        return False
    else:
        return True

def forcedOnPeopleOn(team_list):
    if isAssigned(team_list):
        team_list = unassignTeam(team_list)
    try:
        for forced_on_person in forced_on_people:
            team_list.index(forced_on_person)
    except ValueError:  #person not in team list
        return False
    #if everybody on who must be in team
    return True
        
    

def getTestScoreBlocked(blocked_person, blocked_event):
    return scores_blocked[blocked_person, blocked_event]

def getTestScore(person, event):
    if person < 0: #a fake person
        return 0
    try:
        return scores[person, event]
    except IndexError:  #a fake person
        return 0
        
def scoreTeam(assigned_team):
    #check if the number of seniors is OK
    if numSeniorsOK(assigned_team) == False:
        return invalid_team_score
    if forcedOnPeopleOn(assigned_team) == False:
        return invalid_team_score
    #get the test score for each event.
    #an event is one entry in the vector (its number is its index)
    sum_test_scores_per_event = np.zeros(num_events)
    events_per_person_penalty = 0
    for person in range(0, len(assigned_team)):
        persons_events = assigned_team[person]
        events_per_person_penalty += multiEventPenalty(len(persons_events))
        for event in persons_events:
            sum_test_scores_per_event[event] += getTestScore(person, event)
    #convert scores to placings and sum them to get the total team score
    placing = 0
    for x in range(0, len(sum_test_scores_per_event)):
        score = sum_test_scores_per_event[x]/people_per_event[x]
        placing += predictPlace(score)
    return placing + events_per_person_penalty

def genRandomTeam(size):
    num_guesses = 0
    already_warned = False
    
    if size < len(forced_on_people):
        raise ValueError('the team must be at least as many people as are forced on')
    num_free_slots = size - len(forced_on_people)
    
    #list of candidates without the must have people
    list_of_possible_people = list(range(0, num_people))
    for person in forced_on_people:
        list_of_possible_people.remove(person)
    while True:
        num_guesses += 1
        team_list = random.sample(list_of_possible_people, num_free_slots)
        team_list = team_list + forced_on_people
        if numSeniorsOK(team_list):
            return team_list
        #warn if having trouble guessing team
        if num_guesses > 10000 and already_warned == False:
            print('Guessing a team with few enough seniors is taking longer than expected. Are there enough non-seniors?')
            already_warned == True
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

#takes in a list of unassigned team names and assigns them
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
    team_assigned = [[] for j in range(len(people_names) + 1)]
    num_real_ppl = len(list_blocked_ppl)
    
    for x in range(0, len(assigned_people)):
        assigned_person = assigned_people[x]
        blocked_event = blocked_events[x]
        if assigned_person < num_real_ppl and blocked_event < num_real_events:
            blocked_person = list_blocked_ppl[x]
            person, block = unpackBlockedPersonNum(blocked_person)
            event = singlePersonEventToEvent(blocked_event)
            if getTestScoreBlocked(blocked_person, blocked_event) > 0:
                team_assigned[person].append(event)
            else:
                team_assigned[dummy_person_number].append(event)
    return team_assigned

#turns an assigned team into a numerical list of members
def unassignTeam(assigned_team):
    unassigned_team = []
    #iterate over people's numbers
    for x in range(0, num_people):
        if len(assigned_team[x]) != 0: #events assigned to this person
            unassigned_team.append(x)
    return unassigned_team

#tells if team is assigned or not
def isAssigned(team):
    if isinstance(team[0], list):
        return True
    elif isNumber(team[0]):
        return False
    else:
        raise TypeError('Wrong type of list passed. Member 0: ' + str(team[0]))
        

def getTeamScore(unassigned_team_list):
    if numSeniorsOK(unassigned_team_list) == False:
        return invalid_team_score
    if forcedOnPeopleOn(unassigned_team_list) == False:
        return invalid_team_score
    return scoreTeam(assignTeam(unassigned_team_list))
#takes in an unassigned list and returns it sorted by least contribution to greatest
def findListOfPersonContributions(team_list):
    people_vs_score_list = []
    for person in team_list:
        team_list_with_person_removed = copy.deepcopy(team_list)
        team_list_with_person_removed.remove(person)
        team_list_with_person_removed.sort()
        people_vs_score_list.append([person, getTeamScore(team_list_with_person_removed)])
        del team_list_with_person_removed
    people_vs_score_list.sort(key=lambda x: x[1]) #sorts by score
    return people_vs_score_list

def findBestAdditionList(team_list):
    people_vs_score_list = []
    possible_people = [x for x in range(0, num_people) if x not in team_list]
    for person in possible_people:
        team_list_with_person_added = copy.deepcopy(team_list)
        team_list_with_person_added.append(person)
        people_vs_score_list.append([person, getTeamScore(team_list_with_person_added)])
        del team_list_with_person_added
    people_vs_score_list.sort(key=lambda x: x[1]) #sorts by score
    return getCol(people_vs_score_list, 0)

def findBestAddition(team_list):
    return findBestAdditionList(team_list)[0]

#returns the new team list and True (if it could replace) or False (if already optimized)
def stepTeam(team_list):
    print('')
    person_vs_score_list = np.asarray(findListOfPersonContributions(team_list))
    people_booting_order = getCol(person_vs_score_list, 0)
    for booted_person in people_booting_order:
        new_team = copy.deepcopy(team_list)
        new_team.remove(booted_person)
        added_person = findBestAddition(new_team)
        if added_person != booted_person:
            print('Booted person: ' + personNumToName(int(booted_person)))
            print('Added person: ' + personNumToName(int(added_person)))
            new_team.append(added_person)
            new_team.sort()
            print('Current team:')
            print(new_team)
            print('New score: ' + str(getTeamScore(new_team)))
            return new_team, True
        del new_team
    #if can't optimize further
    return team_list, False

def optimizeTeam(team_list):
    can_be_optimized = True
    while can_be_optimized == True:
        team_list, can_be_optimized = stepTeam(team_list)
    return team_list

def teamToHumanReadableTeam(clean_assigned_team):
    team_dict = {}
    for person in range(0, len(clean_assigned_team)):
        person_name = personNumToName(person)
        events = eventNumListToNameList(clean_assigned_team[person])
        if len(events) > 0:  #has some events assigned
            team_dict[person_name] = sorted(events)
    return team_dict

def humanPrintAssignedTeam(assigned_team, to_file = sys.stdout):
    if isinstance(assigned_team, list):
        human_readable_team = teamToHumanReadableTeam(assigned_team)
    #need a mutable copy so we can add in the fake person
    names_to_try = list(people_names)
    names_to_try.append(dummy_person_name)
    for person in names_to_try:
        try:
            print(str(person) + ': ' + str(human_readable_team[person]), file = to_file)
        except KeyError:
            pass
    if isinstance(assigned_team, list):
        print('Score: ' + str(scoreTeam(assigned_team)), file = to_file)
    print('', file = to_file)
    
def prettyPrintList(list_of_lists):
    for element in list_of_lists:
        print(element)

def humanPrintTeamList(team_list):
    for person in team_list:
        print(personNumToName(person))

def addTeamToListOfTeams(assigned_team):
    global list_of_best_teams
    score = scoreTeam(assigned_team)
    for team_and_score in list_of_best_teams:
        if team_and_score[0] == score:  #this team already exists
            return  #break out
    list_of_best_teams.append([score, assigned_team])
    
    list_of_best_teams = sorted(list_of_best_teams)

def getScoreAndTeam(team_from_list_of_teams):
    assigned_team = team_from_list_of_teams[1]
    score = team_from_list_of_teams[0]
    return score, assigned_team

def printTeams(to_file):
    open(to_file, 'w').close()  #wipes file of previous teams. From stackoverflow
    with open(to_file, "a") as save_file:
        print('Number of teams: ' + str(len(list_of_best_teams)), file = save_file)
        print('', file = save_file)
        for team in list_of_best_teams:
            people = team[1]
            humanPrintAssignedTeam(people, to_file = save_file)

def dumpTeams(was_interrupted):
    finished_optimizing, current_team_list = loadTeams()
    for team in current_team_list:
        addTeamToListOfTeams(team[1])
    printTeams(file_to_save_to)
    
    list_of_best_teams.insert(0, was_interrupted)
    with open('outfile', 'wb') as fp:
        pickle.dump(list_of_best_teams, fp)

def loadTeams():
    try:
        with open ('outfile', 'rb') as fp:
            data = pickle.load(fp)
            was_interrupted = data[0]
            team_list = data[1:]
            return was_interrupted, team_list
    except FileNotFoundError:
        return False, []

def fuseScoresAcrossInvites(list_of_scores, list_of_weights):
    fused_score = 0
    total_weight_from_invites_with_scores = 0
    total_weight_from_invites_without_scores = 0
    
    for x in range(0, len(list_of_scores)):
        score = list_of_scores[x]
        weight = list_of_weights[x]
        if score > 0: #did this event at this invite
            fused_score += score*weight
            total_weight_from_invites_with_scores += weight
        else: #didn't do this event at this invite
            total_weight_from_invites_without_scores += weight
    if total_weight_from_invites_with_scores == 0:  #never participated
        return 0
    #the sqrt is so that non-participation is a small penalty, but not insurmountable.
    return fused_score/math.sqrt(total_weight_from_invites_with_scores)
    

def fuseScoreMatrix(scores_from_all_invites, invite_weights, event_weights):
    num_ppl, num_events = np.shape(scores_from_all_invites[0])
    scores = np.zeros((num_ppl, num_events))
    
    scores_from_all_invites = np.asarray(scores_from_all_invites)
    for person in range(0, num_ppl):
        for event in range(0, num_events):
            scores[person, event] = fuseScoresAcrossInvites(
                    scores_from_all_invites[:, person, event], invite_weights)
    
    #deal with event weights
    for x in range(0, len(event_weights)):
        weight = event_weight[x]
        scores[:,x] = scores[:,x]*weight
        
    return scores

#ensure invite weights sum to 1
def normalizeInviteWeights(invite_weight_list):
    return tuple(np.asarray(invite_weight_list)/sum(invite_weight_list))

def loadAndCheckForcedOnPeople():
    people_who_must_be_on = []
    with open(csv_of_people_who_must_be_on) as data_file:
        data_reader = csv.reader(data_file)
        for row in data_reader:
            for person in row:
                if person != '':
                    try:
                        person_num = personNameToNum(person)
                    except ValueError:
                        raise ValueError('People in the forced on team list must be in the team list')
                    people_who_must_be_on.append(person_num)
    print('')
    print('People on every team:')
    humanPrintTeamList(people_who_must_be_on)
    return people_who_must_be_on

def loadEventConflicts():
    event_conflicts = []
    with open(event_conflicts_csv) as data_file:
        data_reader = csv.reader(data_file)
        for row in data_reader:
            event_conflicts.append(row)
    event_conflicts = removeFromList(event_conflicts, '')
    print('')
    print('The schedule:')
    prettyPrintList(event_conflicts)
    return event_conflicts
            

def loadFile(file_name):  
    print('loading file: ' + str(file_name))     
    #read in data
    #for people names
    first_column = []
    #for is a senior
    second_column = []
    #for everything else
    prelim_data = []
    with open(file_name) as data_file:
        data_reader = csv.reader(data_file)
        for row in data_reader:
            first_column.append(row[0])
            second_column.append(row[1])
            prelim_data.append(row[2:])
    
    people_names = tuple(first_column[4:])
    event_names = tuple(prelim_data[0])
    people_per_event = tuple(strListToNumList(prelim_data[1]))
    event_weight = tuple(strListToNumList(prelim_data[3], num_type = float))
    data_weight = float(first_column[0])
    
    max_data_scores = tuple(strListToNumList(prelim_data[2], num_type = float))
    raw_data_scores = strListToNumList(prelim_data[4:], num_type = float)
    is_senior = tuple(strListToNumList(second_column[4:], num_type = int))
    np_raw_data_scores = np.asarray(raw_data_scores) #convert to numpy array
    
    checkInputData(np_raw_data_scores, max_data_scores, people_per_event, event_names, people_names, is_senior)
    processed_data_scores = normalizeData(np_raw_data_scores, max_data_scores)
    return processed_data_scores, people_names, is_senior, event_names, people_per_event, event_weight, data_weight


#start by importing event blocks
event_conflicts = loadEventConflicts()


#import data from file
processed_scores_list = []
people_names_list = []
seniors_list = []
event_names_list = []
people_per_event_list = []
event_weight_list = []
invite_weight_list = []
for file in list_of_files:
    processed_data_scores, people_names, are_seniors, event_names, people_per_event, event_weight, data_weight = loadFile(file)
    processed_scores_list.append(processed_data_scores)
    people_names_list.append(people_names)
    seniors_list.append(are_seniors)
    event_names_list.append(event_names)
    people_per_event_list.append(people_per_event)
    event_weight_list.append(event_weight)
    invite_weight_list.append(data_weight)

#begin checks to see if csv file headers are consistent
if checkAllAreEqual(people_names_list) == False:
    raise ValueError('people names must be consistent across files')
if checkAllAreEqual(seniors_list) == False:
    raise ValueError('the people who are seniors must be consistent across files')
if checkAllAreEqual(event_names_list) == False:
    raise ValueError('event names must be consistent across files')
if checkAllAreEqual(people_per_event_list) == False:
    raise ValueError('people per event must be consistent across files')
if checkAllAreEqual(event_weight_list) == False:
    raise ValueError('event weights must be consistent across files')
    
#normalize weights
invite_weight_list = normalizeInviteWeights(invite_weight_list)

#nobody should be modifying these    
people_names = tuple(people_names_list[0])
who_are_seniors = tuple(seniors_list[0])

event_names = tuple(event_names_list[0])
people_per_event = tuple(people_per_event_list[0])
event_weight = tuple(event_weight_list[0])

#people who must be on. Must be after we create the list of people name
forced_on_people = loadAndCheckForcedOnPeople()
print('Numbers of must be present people:' + str(forced_on_people))

#generate combo array of all data
scores = fuseScoreMatrix(processed_scores_list, invite_weight_list, event_weight)

#optimize code by compressing the event schedule
#compressSchedule()
    
#begin processing
num_people = len(people_names)
max_person_num = num_people - 1
num_events = len(event_names)
num_blocks = len(event_conflicts)
max_people_per_event = max(people_per_event)


#split people into blocks for Hungarian algorithim processing
scores_blocked, person_block_matching = splitScoreArray(scores)

#debugging
#this team caused an infinite loop
prob_team = [0, 3, 4, 9, 11, 13, 14, 15, 16, 17, 20, 21, 23, 24, 27]
test_team = [0, 1, 3, 7, 10, 11, 12, 13, 15, 16, 18, 22, 27, 28]


#prior file data will be loaded when we dump the data
finished_optimizing_previously, list_of_prior_teams = loadTeams()
list_of_best_teams = []
try:
    print('Checking saved teams to see if any can be further optimized. Do not parallelize this step by opening multiple windows.')
    for x in range(0, len(list_of_prior_teams)):
        assigned_team_and_score = list_of_prior_teams[x]
        score, assigned_team = getScoreAndTeam(assigned_team_and_score)
        if score != scoreTeam(assigned_team) or finished_optimizing_previously == False:
            print('Found a team that could be potentially optimized')
            start_time = time.time()
            team_list = unassignTeam(assigned_team)
            optimized_team = optimizeTeam(team_list)
            
            new_assigned_team = assignTeam(optimized_team)
            humanPrintAssignedTeam(new_assigned_team)
            print('Optimization took ' + str(time.time() - start_time) + 's')
            print('If nobody was booted, the team list remained the same, but people may have been rearranged')
  
        addTeamToListOfTeams(assigned_team)
except KeyboardInterrupt:
    print('WARNING: Interrupted in the middle of optimizing prior teams! Strange behavior may result.')
    for y in range(x, len(list_of_prior_teams)):
        assigned_team_and_score = list_of_prior_teams[y]
        score, assigned_team = getScoreAndTeam(assigned_team_and_score)
        
        #the unassignment and reassignment is so the team is assigned optimally with the new input data
        addTeamToListOfTeams(assignTeam(unassignTeam(assigned_team)))
    sorted_list = sorted(list_of_best_teams)
    dumpTeams(False)
    
    print('The SystemExit exception is normal. It exits the program so we do not go on to random teams.')
    sys.exit()  #so we don't go on to random teams
    
print('Finished checking old teams for optimizations. Starting guessing random teams. Parallelization by running multiple copies of this program is fine (and recommended).')
num_tried = 0
try:
    while True:
        start_time = time.time()
        randTeam = genRandomTeam(team_size)

        real_team = optimizeTeam(randTeam)
        assigned_team = assignTeam(real_team)
        humanPrintAssignedTeam(assigned_team)
        addTeamToListOfTeams(assigned_team)
        num_tried += 1
        print('Assignment took ' + str(time.time() - start_time) + 's')
        print('Number of random teams tried:' + str(num_tried))
except KeyboardInterrupt:
    sorted_list = sorted(list_of_best_teams)
    
    dumpTeams(True)