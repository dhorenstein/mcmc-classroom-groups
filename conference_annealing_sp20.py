
# roster format: lastname, firstname, id
# rostername = "astr" + userclass + "_roster_conferences_csv.csv"
# homework format: lastname, firstname, id, excluded
#   excluded = 0 if full credit
#   excluded = 12 if no credit for questions 1 or 2
#   etc.
#   append x if not present, ex. 0x or 12x
# homeworkfile = (rosterpath + "astr" + userclass + "_hw" + str(homeworkcounter) + ".csv")
# conference format: lastname, firstname, id, group, question

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
    n_groups_ideal = math.floor(float(n_credit) / 4)
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
    n_open_groups = len(open_groups)
    # randomly assign remaining students to groups
    # remember to penalize groups of fewer than 4 or more than 5
    for studentid in open_assignments:
        open_assignments[studentid] = random.choice(group_numbers)
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
    #print("Groups for Q1: {:d}, Q2: {:d}, Q3: {:d}".format(n_groups_q1,
    #                                                       n_groups_q2,
    #                                                       n_groups_q3))
    open_questions = [1] * (n_groups_q1 - n_groups_q1_protected)
    open_questions.extend([2] * (n_groups_q2 - n_groups_q2_protected))
    open_questions.extend([3] * (n_groups_q3 - n_groups_q3_protected))
    # randomly assign remaining questions to groups
    random.shuffle(open_questions)
    for pair in zip(list(open_groups.keys()), open_questions):
        open_groups[pair[0]] = pair[1]
    # finally, the mcmc
    # choose whether to change one student's group or two groups' questions
    # calculate penalty
    # remember to break if penalty = 0
    open_studentids = list(open_assignments.keys())
    open_groupids = list(open_groups.keys())
    mcmc_depth = 1e+5
    new_assignments = merge_dictionaries(protected_assignments,
                                         open_assignments)
    new_groups = merge_dictionaries(protected_groups, open_groups)
    penalty_initial = mcmc_penalty(priorassociations, excludehw, pastdict,
                                   new_assignments, new_groups)
    if (track_penalties == True):
        penalty_history = [penalty_initial]
    for n_iteration in range(int(mcmc_depth)):
        if (penalty_initial == 0):
            break # no better solution can be found
        temperature = (mcmc_depth - n_iteration) / mcmc_depth * 1.0e+6
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
            #print("questions {:f}, {:f}".format(penalty_new - penalty_initial,
            #                                    temperature))
            try:
                acceptance_threshold = min(1.0, math.exp(-(penalty_new -
                                                           penalty_initial) /
                                                         temperature))
            except OverflowError:
                acceptance_threshold = 1.0
            #if ((penalty_new < penalty_initial) or
            #    (random.random() < penalty_limit)): # accept the move
            if (random.random() < acceptance_threshold):
                penalty_initial = penalty_new
            else: # reject the move
                (open_groups[groupid1],
                 open_groups[groupid2]) = question1, question2
        else:
            # swap one student's groups
            studentid1 = random.choice(open_studentids)
            group1, group2 = (open_assignments[studentid1],
                              random.choice(group_numbers))
            #studentid1, studentid2 = random.sample(open_studentids, 2)
            #group1, group2 = (open_assignments[studentid1],
            #                  open_assignments[studentid2])
            #(open_assignments[studentid1],
            # open_assignments[studentid2]) = group2, group1
            open_assignments[studentid1] = group2
            new_assignments = merge_dictionaries(protected_assignments,
                                                 open_assignments)
            new_groups = merge_dictionaries(protected_groups, open_groups)
            penalty_new = mcmc_penalty(priorassociations, excludehw, pastdict,
                                       new_assignments, new_groups)
            #print("groups {:f}, {:f}".format(penalty_new - penalty_initial,
            #                                 temperature))
            try:
                acceptance_threshold = min(1.0, math.exp(-(penalty_new -
                                                           penalty_initial) /
                                                         temperature))
            except OverflowError:
                acceptance_threshold = 1.0
            #if ((penalty_new < penalty_initial) or
            #    (random.random() < penalty_limit)): # accept the move
            if (random.random() < acceptance_threshold):
                penalty_initial = penalty_new
            else: # reject the move
                open_assignments[studentid1] = group1
                #(open_assignments[studentid1],
                # open_assignments[studentid2]) = group1, group2
        if (track_penalties == True):
            penalty_history.append(penalty_initial)
        if (n_iteration % (mcmc_depth / 10) == 0):
            print("Finished MCMC iteration {:d}.".format(n_iteration + 1))
    # structure return data
    return_assignments = merge_dictionaries(protected_assignments,
                                            open_assignments)
    return_groups = merge_dictionaries(protected_groups, open_groups)
    if (track_penalties == True):
        pyplot.ion()
        pyplot.plot([i for i in range(len(penalty_history))],
                    penalty_history, "-")
        # solid black line, no markers
        pyplot.show()
        print("Final MCMC penalty: {:d}".format(penalty_history[-1]))
        inverse_groups = invert_groups(return_assignments, return_groups)
        for groupid in inverse_groups:
            print("Students in Group {:d}: {:d}".format(groupid, len(inverse_groups[groupid])))
    return (return_assignments, return_groups)

# get path data from user
userclass = input("Please enter the class name. \nastr")
rosterpath = userclass + "/"
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
                    linedata = line[:-1].split(",")
                    # without newline character
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
