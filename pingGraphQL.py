import json
import pandas as pd
import requests
import matplotlib.pyplot as plt


def custom_ping_leaderboad():
    """
    Use a custom graph QL query to get the load the leaderboard into a pandas dataframe.
    Source: https://towardsdatascience.com/connecting-to-a-graphql-api-using-python-246dda927840

    """

    batch = 1000
    endpoint = 'https://api-tournament.numer.ai/'

    query = """{ 
	v2Leaderboard { # If you dont' limit it gets 4352. You might want to batch it 
	    returns
        corrRep
        fncRep
        mmcRep
        username
        nmrStaked
        returns # this is the 1d roi. 2.2 is 2.2 percent. It does not *100
        rank
	    }  
    }"""
    # this throws erros when you try and get the returns for weeks13 and weeks52.
    # you should let them know.

    r = requests.post(url=endpoint, json={'query': query})
    json_data = json.loads(r.text)

    df_data = json_data['data']['v2Leaderboard']
    df = pd.DataFrame(df_data)
    df.convert_dtypes()
    df['nmrStaked'] = pd.to_numeric(df['nmrStaked'])
    return df


df = custom_ping_leaderboad()