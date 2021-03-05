import numerapi
import json
import pandas as pd
import numpy as np


def open_api_access():
    """
    Read in my private key from creds.json and return the numer.ai api wrapper
    """
    creds  =  open('creds.json','r')
    key = json.load(creds)
    secret_key = key['secret_key']
    my_username = key['username']
    napi = numerapi.NumerAPI(id, secret_key)
    creds.close()
    return napi


def create_df_from_leaderboard(leader_board):
    """
    return a pandas dataframe from leader board, the list of dictionaris.

    """
    return pd.DataFrame(leader_board)

def save_leader_board_df_to_csv(df):
    """
    So you dont' have to query the API a bunch, save the df into a .csv that you just load instead.
    """
    try:
        to_save = open('leaderboard.csv', 'x')
    except:
        to_save = open('leaderboard.csv', 'w')

    df.to_csv(to_save, index=False, line_terminator='')
    to_save.close()


def load_leaderboard():
    with open('leaderboard.csv','r') as fin:
        df = pd.read_csv(fin)
        return df


def query_and_save_leaderboard():
    """
    Ping numerAI and refesh the leaderboard record you have. Only need to do this once per day
    """
    napi = open_api_access()
    leader_board = napi.get_leaderboard(6000)
    df = create_df_from_leaderboard(leader_board)
    save_leader_board_df_to_csv(df)


def query_leaderboard():
    """
    Ping API to get the current leaderboard
    """
    napi = open_api_access()
    leader_board = napi.get_leaderboard(6000)
    df = create_df_from_leaderboard(leader_board)
    return df


def get_all_user_names():
    """
    Query API to get all the usernames of the models.
    """
    leader_board = query_leaderboard()
    usernames = leader_board['username']
    return usernames





# use https://numerapi.readthedocs.io/en/latest/api/numerapi.html#module-numerapi.numerapi


def get_all_user_details(usernames):
    """
    Ping the API with public_user_profile() and daily_user_performances()

    Cast into a  dataframe,
    Save that dataframe to a .csv

    note: daily_user_performances() does not return the username as well. You need to merge that into as well.


    This is a simplified verison I might go back and capture more data later. Right now I just get the average rolling_score_rep for taht uer
    """

    user_details =[]

    napi = open_api_access()


    # untested
    # for some readon daily_user_performances is always in a factor of 5. I dont know why that is so 
    for name in usernames:
        profile = napi.public_user_profile(name) # this is a dict
        profile_df = pd.DataFrame(profile)
        select_profile = profile_df[['username', 'totalStake','startDate']].drop_duplicates().to_dict()
        daily_performace = napi.daily_user_performances(name)
        daily_performace_df = pd.DataFrame(daily_performace)
        print(daily_performace_df.columns)
        # I need to get mmc, and fnc here too.
        averages = daily_performace_df.mean(axis=0, numeric_only=True).to_dict()
        merged = select_profile | averages
        user_details.append(merged)
    details = pd.DataFrame(user_details)
    return details
        

def main():
    usernames = get_all_user_names()
    details = get_all_user_details(usernames)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    #print(details.head())

# learn how to interact with graph ql api. 



def get_submission_results():
    # usernames = get_all_user_names().to_list()

    # a_user = usernames[0] 
    # print(a_user)
    # api = numerapi.SignalsAPI()
    # daily_performances = api.daily_user_performances(username=a_user)
    # print(type(daily_performances))
    # print(len(daily_performances))
    # print(daily_performances[1])

    api.raw_query(False, 'asdf')



# you are going to need to write your own custom Graph QL quries. 

# don't know how to do that yet

get_submission_results()