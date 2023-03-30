
# roster format: lastname, firstname, id
# rostername = "astr" + userclass + "_roster_conferences_csv.csv"
# homework format: lastname, firstname, id, excluded
#   excluded = 0 if full credit
#   excluded = 12 if no credit for questions 1 or 2
#   etc.
#   append x if not present, ex. 0x or 12x
# homeworkfile = (rosterpath + "astr" + userclass + "_hw" + str(homeworkcounter) + ".csv")
# conference format: lastname, firstname, id, group, question

# tempering depth and annealing depth specified around line 262

# import relevant libraries
import math
import random
from matplotlib import pyplot # for plotting mcmc convergence
import json
from copy import deepcopy

# function to merge two dictionaries
def merge_dictionaries(dict1, dict2):
    dict3 = dict1.copy()
    dict3.update(dict2)
    return dict3

# function to count how many students per group
def invert_groups(assignments, groups):
    inverse_groups = {} # key = groupid, value = all studentids
    for groupid in groups:
        inverse_groups[groupid] = []
    for studentid in assignments:
        groupid = assignments[studentid]
        inverse_groups[groupid].append(studentid)
    return inverse_groups

# penalty function for mcmc
def mcmc_penalty(priorassociations, excludehw, pastdict,
                 new_assignments, new_groups):
    penalty = 0
    max_exponent = 19 # maximum penalty
    inverse_groups = invert_groups(new_assignments, new_groups)
    # penalize for:
    #   having worked together in previous group
    #   having worked on same question in previous group
    #   not having completed the assigned question
    #   groups not having 4 or 5 members
    for studentid in new_assignments:
        proposed_group = new_assignments[studentid]
        proposed_question = new_groups[proposed_group]
        for partnerid in inverse_groups[proposed_group]:
            if priorassociations[studentid]:
                group_exponent = 2 * priorassociations[studentid].count(partnerid)
                # note x2 multiplier, compared to question exponent (below)
                if (group_exponent > 0):
                    penalty = penalty + 2 ** group_exponent
        question_exponent = pastdict[studentid].count(proposed_question)
        if (question_exponent > 0):
            penalty = penalty + 2 ** question_exponent
        if (str(proposed_question) in excludehw[studentid]):
            penalty = penalty + 2 ** max_exponent
    for groupid in inverse_groups:
        n_members = len(inverse_groups[groupid])
        if ((n_members < 4) or (n_members > 5)):
            penalty = penalty + 2 ** max_exponent
    return penalty

# annealing function
def anneal(temperature, previous_energy, open_assignments, open_groups, depth,
           open_studentids, open_groupids, group_numbers,
           protected_assignments, protected_groups, priorassociations,
           excludehw, pastdict, track_penalties):
    new_assignments = merge_dictionaries(protected_assignments,
                                         open_assignments)
    new_groups = merge_dictionaries(protected_groups, open_groups)
    penalty_initial = mcmc_penalty(priorassociations, excludehw, pastdict,
                                   new_assignments, new_groups)
    n_open_groups = len(open_groups)
    iteration_history = []
    if (track_penalties == True):
        iteration_history.append(penalty_initial)
    for n_iteration in range(int(depth)):
        if (penalty_initial == 0):
            break # no better solution can be found
        if ((random.random() > 0.8) and (n_open_groups > 1)):
            # swap two groups' questions
            groupid1, groupid2 = random.sample(open_groupids, 2)
            question1, question2 = (open_groups[groupid1],
                                    open_groups[groupid2])
            (open_groups[groupid1],
             open_groups[groupid2]) = question2, question1
            new_assignments = merge_dictionaries(protected_assignments,
                                                 open_assignments)
            new_groups = merge_dictionaries(protected_groups, open_groups)
            penalty_new = mcmc_penalty(priorassociations, excludehw, pastdict,
                                       new_assignments, new_groups)
            #print("question {:f}".format(-(penalty_new - penalty_initial) /
            #                             temperature))
            try:
                acceptance_threshold = min(1.0, math.exp(-(penalty_new -
                                                           penalty_initial) /
                                                         temperature))
            except OverflowError:
                acceptance_threshold = 1.0
            if (random.random() < acceptance_threshold): # accept the move
                penalty_initial = penalty_new
            else: # reject the move
                (open_groups[groupid1],
                 open_groups[groupid2]) = question1, question2
        else:
            # swap one student's groups
            studentid1 = random.choice(open_studentids)
            group1, group2 = (open_assignments[studentid1],
                              random.choice(group_numbers))
            open_assignments[studentid1] = group2
            new_assignments = merge_dictionaries(protected_assignments,
                                                 open_assignments)
            new_groups = merge_dictionaries(protected_groups, open_groups)
            penalty_new = mcmc_penalty(priorassociations, excludehw, pastdict,
                                       new_assignments, new_groups)
            #print("group {:f}".format(-(penalty_new - penalty_initial) /
            #                          temperature))
            try:
                acceptance_threshold = min(1.0, math.exp(-(penalty_new -
                                                           penalty_initial) /
                                                         temperature))
            except OverflowError:
                acceptance_threshold = 1.0
            if (random.random() < acceptance_threshold): # accept the move
                penalty_initial = penalty_new
            else: # reject the move
                open_assignments[studentid1] = group1
        if (track_penalties == True):
            iteration_history.append(penalty_initial)
        #if (n_iteration % depth / 10 == 0):
        #    print("Finished {:f} annealing iteration {:d} with energy {:d}.".format(temperature, n_iteration + 1, penalty_initial))
    return [temperature, penalty_initial, open_assignments, open_groups,
            depth, iteration_history]

# mcmc
def updategroups(studentslist, priorassociations, includegroup, excludehw,
                 pastdict):
    # list of students (lastname, firstname, id)
    # dictionary of prior associations
    #   key = student id, value = list including multiples
    # dictionary of existing groups and questions (cannot change these)
    #   key = student id, value = (string, string)
    # dictionary of which questions not to completed in hw
    #   key = student id, value = string (detailed above)
    # dictionary of which questions assigned in past conferences
    #   key = student id, value = list
    track_penalties = True # for graphing convergence
    n_credit = 0
    protected_assignments, open_assignments = {}, {}
    protected_groups, open_groups, existing_groups = {}, {}, {}
    for line in studentslist: # lastname, firstname, id
        studentid = line[2]
        if ((excludehw[studentid] != "") and
            ("x" not in excludehw[studentid])):
            # count students who turned in hw and are present
            n_credit = n_credit + 1
            if (studentid in includegroup):
                groupid = int(includegroup[studentid][0])
                questionid = int(includegroup[studentid][1])
                protected_assignments[studentid] = groupid
                # student already assigned to group
                if (groupid not in existing_groups):
                    existing_groups[groupid] = [studentid]
                    # keep track of which group numbers are used
                    # key is groupid, value is list of all studentids
                    protected_groups[groupid] = questionid
                    # question already assigned to group
                else:
                    existing_groups[groupid].append(studentid)
            else:
                open_assignments[studentid] = 0 # placeholder
                # student not yet assigned to group
    n_open_assignments = len(open_assignments)
    n_groups_ideal = math.ceil(float(n_credit) / 5)
    # create list of group numbers to be assigned
    if (len(existing_groups) >= n_groups_ideal):
        group_numbers = list(existing_groups.keys())
    else:
        if (len(existing_groups) > 0):
            group_numbers = list(existing_groups.keys())
        else:
            group_numbers = []
        group_counter = 0
        while (len(group_numbers) < n_groups_ideal):
            group_counter = group_counter + 1
            if (group_counter not in group_numbers):
                group_numbers.append(group_counter)
                open_groups[group_counter] = 0 # placeholder
    n_groups = len(group_numbers)
    # create list of question numbers to be assigned
    if (n_groups % 3 == 2):
        # allocate extra to q1 and q2
        n_groups_q1 = math.ceil(n_groups / 3)
        n_groups_q2 = math.ceil(n_groups / 3)
        n_groups_q3 = math.floor(n_groups / 3)
    elif (n_groups % 3 == 1):
        # allocate extra to q1
        n_groups_q1 = math.ceil(n_groups / 3)
        n_groups_q2 = math.floor(n_groups / 3)
        n_groups_q3 = math.floor(n_groups / 3)
    else: # n_groups % 3 == 0)
        n_groups_q1 = n_groups / 3
        n_groups_q2 = n_groups / 3
        n_groups_q3 = n_groups / 3
    n_groups_q1_protected = list(protected_groups.values()).count(1)
    n_groups_q2_protected = list(protected_groups.values()).count(2)
    n_groups_q3_protected = list(protected_groups.values()).count(3)
    q_excess = ((n_groups_q1_protected - n_groups_q1) +
                (n_groups_q2_protected - n_groups_q2) +
                (n_groups_q3_protected - n_groups_q3))
    # if more than ideal already assigned, reallocate as necessary
    if (q_excess > 0):
        q1_excess = n_groups_q1_protected - n_groups_q1
        q2_excess = n_groups_q1_protected - n_groups_q1
        q3_excess = n_groups_q1_protected - n_groups_q1
        excess_counter = 0
        for item in [q1_excess, q2_excess, q3_excess]:
            if (item > 0):
                excess_counter = excess_counter + 1
        if (excess_counter == 1):
            excess_correction = math.ceil(float(q_excess / 2))
            if (q3_excess > 0):
                # take half (round up) from q2, rest from q1
                n_groups_q3 = n_groups_q3_protected
                n_groups_q2 = n_groups_q2 - excess_correction
                n_groups_q1 = n_groups - n_groups_q2 - n_groups_q3
            else:
                # take half (round up) from q3, rest from q1 or q2
                n_groups_q3 = n_groups_q3 - excess_correction
                if (q2_excess > 0):
                    n_groups_q2 = n_groups_q2_protected
                    n_groups_q1 = n_groups - n_groups_q2 - n_groups_q3
                else: # q1_excess > 0
                    n_groups_q1 = n_groups_q1_protected
                    n_groups_q2 = n_groups - n_groups_q1 - n_groups_q3
        else: # at least one must be over, but all three can't be
            # take all excess from only q not already over-allocated
            if (q1_excess > 0):
                n_groups_q1 = n_groups_q1_protected
                if (q2_excess > 0):
                    n_groups_q2 = n_groups_q2_protected
                    n_groups_q3 = n_groups_q3 - q_excess
                else: # q3_excess > 0
                    n_groups_q3 = n_groups_q3_protected
                    n_groups_q2 = n_groups_q2 - q_excess
            else: # q2_excess > 0 and q3_excess > 0
                n_groups_q1 = n_groups_q1 - q_excess
                n_groups_q2 = n_groups_q2_protected
                n_groups_q3 = n_groups_q3_protected
    #print(n_groups_q1, n_groups_q1_protected)
    open_questions = [1] * (int(n_groups_q1) - n_groups_q1_protected)
    open_questions.extend([2] * (int(n_groups_q2) - n_groups_q2_protected))
    open_questions.extend([3] * (int(n_groups_q3) - n_groups_q3_protected))
    # finally, the mcmc
    # choose whether to change one student's group or two groups' questions
    # calculate penalty
    # remember to break if penalty = 0
    open_studentids = list(open_assignments.keys())
    open_groupids = list(open_groups.keys())
    tempering_depth, annealing_depth = int(1e+1), int(1e+3)
    # initialize annealing states
    temperatures = [10 ** i for i in range(-2, 7, 2)]
    #temperatures = [1.0e+10 * (0.25 * i) + 1.0e+5 for i in range(5)]
    annealing_assignments = [{}, {}, {}, {}, {}]
    annealing_groups = [{}, {}, {}, {}, {}]
    annealed_output = [None, None, None, None, None]
    acceptances = [None, None, None, None, None]
    # note to self: "[[]] * 5" produces only shallow copies
    if (track_penalties == True):
        penalty_history = {}
        for temp_index in range(5):
            penalty_history[temperatures[temp_index]] = []
    for anneal_index in range(5):
        # randomly assign open students to open groups
        for studentid in open_assignments:
            annealing_assignments[anneal_index][studentid] = random.choice(group_numbers) # key = studentid, value = groupid
            # may result in groups with fewer than 4 or more than 5 students
        # randomly assign open questions to open groups
        random.shuffle(open_questions)
        for pair in zip(list(open_groups.keys()), open_questions):
            annealing_groups[anneal_index][pair[0]] = pair[1] # key = groupid, value = questionid
        annealed_output[anneal_index] = [temperatures[anneal_index],
                                         None, # placeholder for energy
                                         annealing_assignments[anneal_index],
                                         annealing_groups[anneal_index],
                                         annealing_depth]
    for n_iteration in range(int(tempering_depth)):
        for temp_index in range(5):
            annealed_output[temp_index] = anneal(annealed_output[temp_index][0],
                                                 annealed_output[temp_index][1],
                                                 annealed_output[temp_index][2],
                                                 annealed_output[temp_index][3],
                                                 annealed_output[temp_index][4],
                                                 open_studentids,
                                                 open_groupids,
                                                 group_numbers,
                                                 protected_assignments,
                                                 protected_groups,
                                                 priorassociations, excludehw,
                                                 pastdict,
                                                 track_penalties)
            # returns: temp, energy, assignments, groups, depth, history
            if (track_penalties == True):
                iteration_history = annealed_output[temp_index][5]
                iteration_temp = annealed_output[temp_index][0]
                penalty_history[iteration_temp].extend(iteration_history)
                trunc_history = annealing_depth - len(iteration_history)
                if (trunc_history > 0):
                    iteration_history = [float("nan")] * trunc_history
                    penalty_history[iteration_temp].extend(iteration_history)
        annealed_temps = [output[0] for output in annealed_output]
        annealed_energies = [output[1] for output in annealed_output]
        if (min(annealed_energies) == 0):
            print("Optimal solution found after tempering iteration {:d}.".format(n_iteration + 1))
            break # no better solution to be found
        #for temp_index in range(5):
        #    try:
        #        acceptances[temp_index] = min(1.0, math.exp((1.0 / annealed_temps[temp_index] - 1.0 / annealed_temps[(temp_index + 1) % 5]) * -(annealed_energies[temp_index] - annealed_energies[(temp_index + 1) % 5])))
        #    except OverflowError:
        #        acceptances[temp_index] = 1.0
        #for accept_index in range(5):
        #    if (random.random() < acceptances[accept_index]):
        #        temp1 = annealed_output[accept_index][0]
        #        temp2 = annealed_output[(accept_index + 1) % 5][0]
        #        annealed_output[accept_index][0] = temp2
        #        annealed_output[(accept_index + 1) % 5][0] = temp1
        #        break # don't exchange more than two chains at once
        for temp_index in range(5):
            try:
                acceptance_value = min(1.0, math.exp((1.0 / annealed_temps[temp_index] - 1.0 / annealed_temps[(temp_index + 1) % 5]) * -(annealed_energies[temp_index] - annealed_energies[(temp_index + 1) % 5])))
            except OverflowError:
                acceptance_value = 1.0
            if (random.random() < acceptance_value):
                # swap temperature with next adjacent chain
                temp1 = annealed_output[temp_index][0]
                temp2 = annealed_output[(temp_index + 1) % 5][0]
                annealed_output[temp_index][0] = temp2
                annealed_output[(temp_index + 1) % 5][0] = temp1
                # swap energy with next adjacent chain
                energy1 = annealed_output[temp_index][1]
                energy2 = annealed_output[(temp_index + 1) % 5][1]
                annealed_output[temp_index][1] = energy2
                annealed_output[(temp_index + 1) % 5][1] = energy1
                # update quick views
                annealed_temps = [output[0] for output in annealed_output]
                annealed_energies = [output[1] for output in annealed_output]
        if (n_iteration % (tempering_depth / 10) == 0):
            print("Finished tempering iteration {:d}.".format(n_iteration + 1))
    tempered_energies = [output[1] for output in annealed_output]
    min_energy = min(tempered_energies)
    min_index = tempered_energies.index(min_energy)
    open_assignments = annealed_output[min_index][2]
    open_groups = annealed_output[min_index][3]
    # structure return data
    return_assignments = merge_dictionaries(protected_assignments,
                                            open_assignments)
    return_groups = merge_dictionaries(protected_groups, open_groups)
    if (track_penalties == True):
        pyplot.ion()
        for n_chain in range(5):
            chain_temp = temperatures[4 - n_chain] # high to low
            chain_history = penalty_history[chain_temp]
            annealing_steps = [i for i in range(len(chain_history))]
            n_alpha = n_chain * 0.2 + 0.1 # 0.1, 0.3, 0.5, 0.7, 0.9
            legend_str = "{:d}".format(4 - n_chain + 1)
            pyplot.plot(annealing_steps, chain_history, "-", alpha=n_alpha,
                        label=legend_str)
            # solid lines, no markers
        pyplot.legend()
        pyplot.show()
        print("Final MCMC penalty: {:d}".format(min_energy))
        inverse_groups = invert_groups(return_assignments, return_groups)
        for groupid in inverse_groups:
            print("Students in Group {:d}: {:d}".format(groupid, len(inverse_groups[groupid])))
    return (return_assignments, return_groups)

# get path data from user
userclass = input("Please enter the class name. \nastr")
rosterpath = "astr" + userclass + "/"
rostername = "astr" + userclass + "_roster_conferences_csv.csv"
print("The default roster path is: " + rosterpath)
userpath = input("Specify the roster path if different from default. ")
if (userpath != ""):
    rosterpath = userpath
if (not (rosterpath.endswith("/"))): # append "/" to path if necessary
    rosterpath = rosterpath + "/"
print("The default roster file is: " + rostername)
userfile = input("Specify the roster file if different from default. ")
if (userfile != ""):
    rostername = userfile
rosterfile = rosterpath + rostername
print("The roster file to be checked is: " + rosterfile)

# check whether roster exists
#   if it doesn't, generate message and exit
#   if it does, read in the information
try:
    students = []
    with open(rosterfile, "r") as rosterobject:
        for line in rosterobject:
            if (not (line.startswith("#")) and not (line == "")):
                if (line[-1] == "\n"):
                    linedata = line[:-1].split(",") # without newline character
                else:
                    linedata = line.split(",")
                for index in range(3): # lastname, firstname, id
                    linedata[index] = linedata[index].replace("\"", "").strip()
                    # strip quotation marks, then whitespace
                students.append(linedata[:2]) # lastname, firstname, id
                students[-1].append(int(linedata[2]))
    print("Roster found and read.")
    #print(students)
except:
    students = []
    print("Error reading roster. Exiting.")

# check whether prior conference data exists
#   if it does, append to priorconferences array
if students: # evaluates as false if students array is empty
    priorconferences = deepcopy(students) # will be appended with group numbers
    conferencecounter = 1
    conferencefound = True
    while conferencefound:
        try:
            conferencefile = (rosterpath + "astr" + userclass + "_conf" +
                              str(conferencecounter) + ".csv")
            #print(conferencefile)
            with open(conferencefile, "r") as conferenceobject:
                conferencedict = {}
                posterdict = {}
                for line in conferenceobject:
                    if (not (line.startswith("#")) and not (line == "")):
                        linedata = line[:-1].split(",")
                        # without newline character
                        for index in range(5):
                            # lastname, firstname, id, group, question
                            linedata[index] = linedata[index].replace("\"", "").strip()
                            # strip quotation marks, then whitespace
                        studentid = int(linedata[2])
                        if (linedata[3] != ""):
                            studentgroup = int(linedata[3])
                            questionid = int(linedata[4])
                            conferencedict[studentid] = studentgroup
                            posterdict[studentid] = questionid
                        #else:
                        #    conferencedict[studentid] = ""
                        #    posterdict[studentid] = ""
                for line in priorconferences: # lastname, firstname, id
                    studentid = line[2]
                    if (studentid in conferencedict):
                        line.extend([conferencedict[studentid],
                                     posterdict[studentid]])
                    else:
                        line.extend(["", ""])
                        # lastname, firstname, id, group(s)/question(s)
            print("Conference " + str(conferencecounter) + " found and read.")
            conferencecounter = conferencecounter + 1
        #except Exception as inst:
        #    print(type(inst))
        #    print(inst.args)
        #    print(inst)
        except:
            print("Conference " + str(conferencecounter) +
                  " not read. Continuing.")
            conferencecounter = conferencecounter - 1
            conferencefound = False

# check which homework data exists
# determine whether to create new groups or append existing conference
if students: # evaluates as false if students array is empty
    homeworkcounter = 0
    homeworkfound = True
    while homeworkfound:
        try:
            homeworkcounter = homeworkcounter + 1
            homeworkfile = (rosterpath + "astr" + userclass + "_hw" +
                            str(homeworkcounter) + ".csv")
            with open(homeworkfile, "r") as homeworkobject:
                pass
        except:
            homeworkcounter = homeworkcounter - 1
            homeworkfound = False
    if (homeworkcounter == 0):
        print("No homework found. Exiting.")
        operation = "none"
    else:
        print("Most recent homework found: " + str(homeworkcounter))
        if (homeworkcounter == conferencecounter + 1):
            operation = "create"
            print("New groups will be generated for Conference " +
                  str(homeworkcounter))
        elif (homeworkcounter == conferencecounter):
            operation = "update"
            print("Existing groups will be updated for Conference " +
                  str(homeworkcounter))
        else:
            operation = "none"
            print("No action will be taken.")
else:
    operation = "none"

# generate list of prior associations
if (operation != "none"):
    associationsdict, questionsdict = {}, {}
    # for each student, create a set of prior partners
    for line in students: # lastname, firstname, id
        studentid = line[2]
        associationsdict[studentid] = []
        questionsdict[studentid] = []
    for conferenceiteration in range(conferencecounter):
        groupiteration = 1
        groupfound = True
        while groupfound:
            group = set()
            for line in priorconferences: # lastname, firstname, id, group(s)/question(s)
                studentid = line[2]
                groupid = line[3 + conferenceiteration * 2] # indexed from 0
                if (groupid == groupiteration):
                    group.add(studentid)
                    questionid = line[4 + conferenceiteration * 2]
            for studentid in group:
                associationsdict[studentid] = associationsdict[studentid] + list(group.difference(set((studentid,))))
                questionsdict[studentid] = questionsdict[studentid] + list((questionid,))
            if (len(group) > 0):
                groupiteration = groupiteration + 1
            else:
                groupfound = False
    print("List of prior associations built.")
            
# create placeholder for new group if necessary
#   otherwise, read existing groups
# read homework
if (operation == "create"):
    includeddict = {}
    for line in priorconferences: # lastname, firstname, id, group(s)/question(s)
        line.extend(["", ""])
        #studentid = line[2]
        #includeddict[studentid] = ("", "")
if (operation == "update"):
    includeddict = {}
    for line in priorconferences: # lastname, firstname, id, group(s)/question(s)
        studentid = line[2]
        groupid = line[3 + conferencecounter * 2] # indexed from 0
        questionid = line[4 + conferencecounter * 2]
        includeddict[studentid] = (groupid, questionid)
if (operation != "none"):
    excludeddict = {}
    for line in students: # lastname, firstname, id
        studentid = line[2]
        excludeddict[studentid] = ""
    try:
        homeworkfile = (rosterpath + "astr" + userclass + "_hw" +
                        str(homeworkcounter) + ".csv")
        with open(homeworkfile, "r") as homeworkobject:
            for line in homeworkobject:
                if (not (line.startswith("#")) and not (line == "")):
                    if line.endswith("\n"):
                        linedata = line[:-1].split(",")
                        # without newline character
                    else:
                        linedata = line.split(",")
                        # don't leave out last student if doesn't end with blank line
                    for index in range(4):
                        # lastname, firstname, id, excluded (see top)
                        linedata[index] = linedata[index].replace("\"", "").strip()
                        # strip quotation marks, then whitespace
                    studentid, excludedstr = int(linedata[2]), linedata[3]
                excludeddict[studentid] = excludedstr
        print("Homework " + str(homeworkcounter) + " found and read.")
    except:
        print("Error reading homework. Exiting.")
        operation = "none"

# update groups
if (operation != "none"):
    (updated_assignments,
     updated_questions) = updategroups(students, associationsdict,
                                       includeddict, excludeddict,
                                       questionsdict)
    # assignments: key = student id, value = group number
    # groups: key = group number, value = question number
    for line in priorconferences:
        # lastname, firstname, id, group(s)/question(s)
        studentid = line[2]
        if (studentid in updated_assignments):
            groupid = updated_assignments[studentid]
            questionid = updated_questions[groupid]
            if (studentid not in includeddict):
                line[3 + conferencecounter * 2] = groupid # indexed from 0
                line[4 + conferencecounter * 2] = questionid
    print("Groups updated.")
    conferencefile = (rosterpath + "astr" + userclass + "_conf" +
                      str(conferencecounter + 1) + ".csv")
    with open(conferencefile, "w") as conferenceobject:
        for line in priorconferences:
            #write_str = json.dumps(line)[1:-1].replace(", ", ",").replace("\"\"", "") + "\n"
            # strip brackets
            # replace empty ' ""'  with ''
            write_str = (line[0] + ", " + line[1] + ", " + str(line[2]) + ", " +
                         str(line[3 + conferencecounter * 2]) + ", " +
                         str(line[4 + conferencecounter * 2]) + "\n")
            conferenceobject.write(write_str)
    print("Machine-readable file written.")
    readablefile = (rosterpath + "astr" + userclass + "_conf" +
                    str(conferencecounter + 1) + "_readable.txt")
    lastname_char_limit = 31
    firstname_char_limit = 39 - 5 - lastname_char_limit
    # 5 chars reserved for formatting
    printable_names, identical_names, full_names = {}, {}, {}
    # make sure printable names are unique
    for line in priorconferences:
        lastname = line[0]
        firstname = line[1]
        studentid = line[2]
        full_names[studentid] = (lastname, firstname)
        printable_name = lastname[:lastname_char_limit] + ", " + firstname[0] + "."
        if (printable_name not in identical_names):
            identical_names[printable_name] = [studentid]
        else:
            identical_names[printable_name].append(studentid)
    n_initials = 1
    while (n_initials < firstname_char_limit):
        n_initials = n_initials + 1
        remove_names, add_names = [], {}
        for printable_name in identical_names:
            if (len(identical_names[printable_name]) != 1):
                for studentid in identical_names[printable_name]:
                    lastname, firstname = full_names[studentid]
                    new_name = lastname[:lastname_char_limit] + ", " + firstname[:n_initials]
                    if (len(firstname) > n_initials):
                        new_name = new_name + "."
                    if (new_name not in add_names):
                        add_names[new_name] = [studentid]
                    else:
                        add_names[new_name].append(studentid)
                remove_names.append(printable_name)
        for add_name in add_names:
            identical_names[add_name] = add_names[add_name]
            # can't do this while looping through identical_names
        for removable_name in remove_names:
            del identical_names[removable_name]
            # can't do this while looping through identical_names
        if (len(remove_names) == 0):
            break
    for printable_name in identical_names:
        studentid = identical_names[printable_name][0]
        printable_names[studentid] = printable_name
    with open(readablefile, "w") as readableobject:
        for groupid in sorted(list(updated_questions.keys())):
            if ((conferencecounter == 4) and # indexed from 0
                (updated_questions[groupid] == 3)):
                write_str = "(Group {:d} will choose Question 1 or 2.)".format(groupid).ljust(39, " ") + "\n"
            else:
                write_str = "(Group {:d} will present Question {:d}.)".format(groupid, updated_questions[groupid]).ljust(39, " ") + "\n"
            readableobject.write(write_str)
        for line in priorconferences:
            studentid = line[2]
            groupid = line[3 + conferencecounter * 2]
            if (groupid != ""):
                write_str = "{:s} {:2d}\n".format(printable_names[studentid].ljust(36, "-"), groupid)
                readableobject.write(write_str)
    print("Human-readable file written.")
