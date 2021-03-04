import numerapi
import json
import pandas as pd


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
    df = pd.DataFrame(leader_board)
    print(df.head())
    print(df.columns)
    return df


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
    leader_board = napi.get_leaderboard(5000)
    df = create_df_from_leaderboard(leader_board)
    save_leader_board_df_to_csv(df)


def main():  
    leader_board = load_leaderboard()

main()