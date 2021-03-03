import numpy as np
import pandas as pd
from collections import Counter
import csv
# I think it makes the most sense to cast this in to a relational database for some basic quries.
# Cast as .csv file
#user[0] is rank
# user[1] is user_name

# [2] might not exist, optional 2xmmc or mmc 



class User:
    # I don't think I will use this mehtod.
    def __init__(self,rank, name, mode, corr, mmc, fnc, compute,stake, day_1=None, month_3=None, year_1 =None):
        self.rank = rank
        self.name = name
        self.mode = mode
        self.corr = corr
        self.compute = compute
        self.mmc = mmc
        self.fnc =fnc
        self.stake = stake
        self.day_1 = day_1
        self.month_3 = month_3
        self.year_1 = year_1


def remove_new_lines(file):
    """
    remove all the blank lines from the score.txt and rewrite them
    """
    lines = file.readlines()
    lines_cleaned = [line for line in lines if line !='\n']

    with open('cleaned_scores.txt', 'x') as out:
        for l in lines_cleaned:
            out.write(l)


def parse_user_from_raw_lines(record):
    """
    Old Verison You are not using this
    """
    rank = record[0]
    name = record[1]
    mode = record[2]
    stats = record[3].split('\t')
    corr = stats[0]
    mmc = stats[1]
    fnc = stats[2]
    day_1=None
    month_3=None
    year_1 =None
    try:
        stake = stats[3].split(' ')[0]
    except:
        state =None
    if len(stats)==5:
        day_1 = stats[4]
        month_3 = None
        year_1 = None
    elif len(stats)==6:
        day_1 = stats[4]
        month_3 = stats[5]
        year_1 = None
    elif len(stats)==7:
        day_1 = stats[4]
        month_3 = stats[5]
        year_1 = stats[6]
    
    a_user = (rank, name, mode, corr, mmc, fnc, stake, day_1, month_3, year_1)
    return a_user


def group_users(lines):
    """
    note this excludes the last user in the list. here it excludes 
    
    4861
    NEKOG_COMP
    -0.0998	-0.0998	-0.0997	

    You might want to add this manually later	

    lines in all the lines in cleaned_scores.txt
    you take this and return a list of tuples where each tuple is a unique user. 
    """

    counter =0
    lines_with_rank =[0]
    user_tupules =[]
    start_index =0
    for line in lines[1:]:
        try:
            rank = int(line) # you are gettign groups by the rank as a single line
            if rank == 10001110101: # you are just excluding the only user with a weird name. make a joke about this in teh tableau of this
                raise Exception
            counter+=1
            lines_with_rank.append(counter)
        except:
            counter+=1

    for i in range(len(lines_with_rank)-1):
        tup = (lines_with_rank[i], lines_with_rank[i+1])
        user = lines[tup[0]:tup[1]]
        user = [l.strip() for l in user]
        a = [s.split('\t') for s in user]
        user_tupules.append(a)
    

    # at this point you need to work backwards and get the last element. 
    tup = (lines_with_rank[-1], 0) # might be len lines -1
    user = lines[tup[0]:]
    user = [l.strip() for l in user]
    a = [s.split('\t') for s in user]
    user_tupules.append(a)

    lengths = [len(a) for a in user_tupules]
    len_2_users = [l for l in user_tupules if len(l) ==2]

    # the only element in len_2 users is '10001110101'
    # [[['10001110101'], ['-0.096', '-0.0961', '-0.0962']]]
    len_3_users = [l for l in user_tupules if len(l) ==3]
    len_4_users = [l for l in user_tupules if len(l) ==4]
    len_5_users = [l for l in user_tupules if len(l) ==5]
    len_6_users = [l for l in user_tupules if len(l) ==6]

    user_groups = [len_2_users, len_3_users ,len_4_users ,len_5_users ,len_6_users]
    return user_groups


def parse_stats(stats):
    """
        Parse the stats for corr, mmc, fnc, stake, roi_1_day, roi,_3_month, roi_1_year.

        since all the stats look the same you can store all the methods in a singel space.

        Returns an array of floats

    """
    corr = stats[0]
    mmc = stats[1]
    fnc = stats[2]
    if len(stats) == 3:
        stake = 0
        roi_1_day = 0
        roi_3_months = 0
        roi_1_year = 0
    elif len(stats) == 4:
        stake = stats[3].split(' ')[0] # untested this line removes the NMR
        roi_1_day = 0
        roi_3_months = 0
        roi_1_year = 0
    elif len(stats)==5:
        stake = stats[3].split(' ')[0]
        roi_1_day = stats[4]
        roi_3_months = 0
        roi_1_year = 0
    elif len(stats)==6:
        stake = stats[3].split(' ')[0]
        roi_1_day = stats[4]
        roi_3_months = stats[5]
        roi_1_year = 0
    elif len(stats)==7:
        stake = stats[3].split(' ')[0]
        roi_1_day = stats[4]
        roi_3_months = stats[5]
        roi_1_year = stats[6]

    part_done_stats = [corr, mmc, fnc, stake, roi_1_day, roi_3_months, roi_1_year]
    res =[]
    for r in part_done_stats:
        try:
            r = round(float(r),6)
            res.append(r)
        except:
            try:
                r = r[:-1] # remove the % character
                r = round(float(r)*.01,6)
                res.append(r)
            except:
                res.append(0.0) # when they have a roi but no sake eg user35 UUAZED4
    return res


def parse_3_user_groups(users):
    """
    This method takes in the user group with No modes and No compute and returns
    a list of tupele for each person representign:
    rank, usersname, Mode, if compute, corr, mmc, fnc, stake, roi_1_day, roi_3_months, roi_1_year

    If they dont' have on of these value a 0 in inserted instead. 

    example:
    (3673, 'OTOMARUKANTA', 'No Mode', 'No Compute', -0.096, -0.0958, -0.0967, 0.0, 0.0, 0.0, 0.0)
    """

    #counts = Counter([len(a[2]) for a in users])
    uniform_user_list =[]
    for a in users:
        mode = 'No Mode' # defaults
        compute ='No Compute' # defaults
        rank =a[0]
        name =a[1]
        stats = a[2]
        stat_tup = parse_stats(stats)
        uniform_user = [int(rank[0]), name[0], mode, compute]
        for s in stat_tup:
            uniform_user.append(s)
        uniform_user = tuple(uniform_user)
        #print(uniform_user)
        uniform_user_list.append(uniform_user)
    return uniform_user_list


def parse_4_user_groups(users):
    """
    These users have a method but do not have compute. 

    In the future there might be soem users with compute but no method. Right now there are none.

    I spot checked several users with cleaned_scores.text at N=3 all were right

    """
    uniform_user_list =[]
    for a in users:
        mode = a[2] # defaults
        compute ='No Compute' # defaults
        rank =a[0]
        name =a[1]
        stats = a[3]
        stat_tup = parse_stats(stats)
        uniform_user = [int(rank[0]), name[0], mode[0], compute]
        for s in stat_tup:
            uniform_user.append(s)
        uniform_user = tuple(uniform_user)
        #print(uniform_user)
        uniform_user_list.append(uniform_user)
    return uniform_user_list


def parse_5_user_groups(users):
    uniform_user_list =[]
    for a in users:
        mode = a[2] # defaults
        compute =a[3] # defaults
        rank =a[0]
        name =a[1]
        stats = a[4]
        stat_tup = parse_stats(stats)
        uniform_user = [int(rank[0]), name[0], mode[0], compute[0]]
        for s in stat_tup:
            uniform_user.append(s)
        uniform_user = tuple(uniform_user)
        #print(uniform_user)
        uniform_user_list.append(uniform_user)
    return uniform_user_list



def main():
    # scores.txt was gathered on 3/2/2021
    
    with open('cleaned_scores.txt','r') as fin:
        user_groups = group_users(fin.readlines())
        g=[]
        for s in user_groups:
            g.extend(s)
        cleaned_3_group = parse_3_user_groups(user_groups[1])
        cleaned_4_group = parse_4_user_groups(user_groups[2])
        # some users like NUMERARK have ½MMC this shows up as 'Â½MMC'. Cast this this to somethign more intutive
        # right now it is good engough to just keep that as is
        cleaned_5_group = parse_5_user_groups(user_groups[3])
        all_clean_users = cleaned_3_group
        all_clean_users.extend(cleaned_4_group)
        all_clean_users.extend(cleaned_5_group)
        print(len(all_clean_users))
        print(len(g))
        #I am currently missing 263 from user groups and 1 that was never read in from the file.

        with open('finished_cleaned_users.csv', 'w') as out:
            writer = csv.writer(out, lineterminator='\n')
            for row in all_clean_users:
                writer.writerow(row)
main()
