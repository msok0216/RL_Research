from __future__ import print_function

import os.path
import json
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow,Flow
from google.auth.transport.requests import Request
import os
import pickle
# import numpy

# If modifying these scopes, delete the file token.json.
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

f = open('secret.json')

data = json.loads(f.read())

client_id = data['installed']['client_id']
project_id = data['installed']['project_id']
client_secret = data['installed']['client_secret']

hh_sheet_id = data['sheet']['hh_id']
# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = hh_sheet_id
SAMPLE_RANGE_NAME = 'Study1!A:AI'


SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# here enter the id of your google sheet

def main(range_name):
    global values_input, service
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'secret.json', SCOPES) # here enter the name of your downloaded JSON file
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result_input = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=range_name).execute()
    
    values_input = result_input.get('values', [])

    if not values_input and not values_expansion:
        print('No data found.')

    df=pd.DataFrame(values_input[1:], columns=values_input[0])
    return df

def get_data():
    list = []
    for i in range(9):
        r = 'Study' + str(i+1) + '!A:AP'
        list.append(main(r))
    return list

def main1(range_name, spreadsheet):
    global values_input, service
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'secret.json', SCOPES) # here enter the name of your downloaded JSON file
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    SAMPLE_SPREADSHEET_ID = spreadsheet
    sheet = service.spreadsheets()
    result_input = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=range_name).execute()
    
    values_input = result_input.get('values', [])

    if not values_input and not values_expansion:
        print('No data found.')
    # print(len(values_input[0]))
    # print(len(values_input[1]))
    df=pd.DataFrame(values_input[1:], columns=values_input[0])
    return df

def get_woz_data():
    list = []
    # return main('Study1!A:AP')
    for i in range(1,6):
        r = 'P' + str(i) + ' S!A:AF'
        list.append(main1(r, data['sheet']['woz_id']))
    return list

# print(get_additional_data())
# df=pd.DataFrame(values_input[1:], columns=['Speaker', 'Gender', 'IDE Interactions', 'Combined', 'IDE State', "Dialogue State", 'Delivery', 'Actions', 'Tone', 'Who'])
# print(df)

# for col in df.columns:
#     print(col)

# for index, row in df.iterrows():
#     print(row['Intent'], row['Dialogue State'])
