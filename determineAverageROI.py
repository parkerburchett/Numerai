import numpy as np
import pandas as pd
# I think it makes the most sense to cast this in to a relational database for some basic quries.
# Cast as .csv file
#user[0] is rank
# user[1] is user_name

# [2] might not exist, optional 2xmmc or mmc 



class User:
    # I don't think I will use this mode.
    def __init__(self,rank, name, mode, corr, mmc, fnc, stake, day_1=None, month_3=None, year_1 =None):
        self.rank = rank
        self.name = name
        self.mode = mode
        self.corr = corr
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
    Pass this 4 lines from the scores.txt file, it returns a tuple of that user's data. 
    not rigerously tested
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


# you need to parse by a line being single int. You can use try catch blocks for this. 

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
            rank = int(line)
            if rank ==10001110101: # you are just excluding teh only user with a weird name
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
    
    lengths = [len(a) for a in user_tupules]
    len_2_users = [l for l in user_tupules if len(l) ==2]

    # the only element in len_2 users is '10001110101'
    # [[['10001110101'], ['-0.096', '-0.0961', '-0.0962']]]
    len_3_users = [l for l in user_tupules if len(l) ==3]
    len_4_users = [l for l in user_tupules if len(l) ==4]
    len_5_users = [l for l in user_tupules if len(l) ==5]
    len_6_users = [l for l in user_tupules if len(l) ==6]

    user_groups = [len_2_users, len_3_users ,len_4_users ,len_5_users ,len_6_users ]
    for group in user_groups:
        print(len(group))
    # You should short this by lengths

    print(user_tupules[-1])
    print(len_2_users)
    return user_groups

def parse_3_user_groups(users):
    lengths =[len(a) for a in users]


        





def main():
    # scores was gathered on 3/2/2021

    with open('cleaned_scores.txt','r') as fin:
        users = group_users(fin.readlines())
        cleaned_3_group = parse_3_user_groups
       
main()
