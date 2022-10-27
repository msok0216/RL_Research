from __future__ import print_function
import pickle
import os.path
import json
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
#from urllib.error import HTTPError

spreadsheet = None #Blank spreadsheet to be used later

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
# SPREADSHEET_ID = "1RmjF0lTJyJUSQa4VRVva2q6L6SQlO7fsytkKmP-Arxs"#'1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms' #this is the spreadsheet I made


f = open('secret.json')

data = json.loads(f.read())
SPREADSHEET_ID = data['sheet']['hh_id']
"""
def findValue(spread, sheet, id): # Finds where an ID is held in the spreadsheet
    # Gets the values held in the spreadsheet from column A from row 2 onwards
    result = spread.values().get(spreadsheetId = SPREADSHEET_ID, 
                            range = sheet + "!A2:A").execute()
    values = result.get('values')
    if not values:
        print('No data found.')
        return None
    i = 2; # Starts at 2 because sheets indices start at 1 and the first index is the name for our categories.
    for entry in values:
        if id == int(entry[0]):
            return i
        i = i + 1

def get_id(sheet):
    if sheet == "Study1":
        return 0
    elif sheet == "Study2":
        return 2047291576
    elif sheet == "Study3":
        return 844307069
    elif sheet == "Study4":
        return 604228491
    elif sheet == "Study5":
        return 915485595
    elif sheet == "Study6":
        return 1817937262
    elif sheet == "Study7":
        return 2117658586
    elif sheet == "Study8":
        return 1351506954
    elif sheet == "Study9":
        return 1744961862
"""
"""def get_sheet(spreadsheet, sheet):
    result = spreadsheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=sheet + "!A2:I").execute()
    values = result.get('values')
    print(values)
    if not values:
        return None
    else:
        return getSheet(spreadsheet, SPREADSHEET_ID, sheet)"""

def get_sheet(spread, SPREADSHEET_ID, sheet_num, state, action):
    
    SPREADSHEET_ID = "1RmjF0lTJyJUSQa4VRVva2q6L6SQlO7fsytkKmP-Arxs"
    # A query to get all the rows of the 'sheet_num'-th sheet
    ROWS_RANGE = 'Study' + str(sheet_num) + '!A:AZ'

    # Apply query
    result = spread.values().get(spreadsheetId=SPREADSHEET_ID, range=ROWS_RANGE).execute()
        
    #print("    Result Found")
        
    # 2D List containing rows
    sheet_values = result.get('values', [])
        
    # Get the labels
    labels = sheet_values.pop(0)
    
    #print(labels)
    # Create DataFrame
    sheet_df = pd.DataFrame.from_records(sheet_values,columns=labels)
    #print(sheet_df)
    #print(labels)
    #print(sheet_df)
    #print(sheet_df["Speaker"])
    #print(sheet_df["Dialogue State"])
    print(sheet_df.head())
    state = sheet_df["Combined"].where(sheet_df["Speaker"] == state).tolist()
    state = [x for x in state if pd.isnull(x) == False and x != 'nan']
    action = sheet_df["Combined"].where(sheet_df["Speaker"] == action).tolist()
    action = [x for x in action if pd.isnull(x) == False and x != 'nan']
    #print(state)
    #print(action)
    return state, action
    #print(sheet_df["Dialogue State"].where(sheet_df["Speaker"] == "P2").notnull())
    #print(type(sheet_df["Speaker"]))

def getSheet(spread, SPREADSHEET_ID, sheet_num, state, action):
    
    print("Querying Study" + str(sheet_num))
    SPREADSHEET_ID = "16lN1GRCacgyDD3M53oC9oUWtIr7R8-w_7mUYjYRlCfc"
    # A query to get all the rows of the 'sheet_num'-th sheet
    ROWS_RANGE = 'P' + str(sheet_num) + ' S' + '!A:AZ'

    # Apply query
    result = spread.values().get(spreadsheetId=SPREADSHEET_ID, range=ROWS_RANGE).execute()
        
    #print("    Result Found")
        
    # 2D List containing rows
    sheet_values = result.get('values', [])
        
    # Get the labels
    labels = sheet_values.pop(0)
    
    #print(labels)
    # Create DataFrame
    sheet_df = pd.DataFrame.from_records(sheet_values,columns=labels)
    
    #print(sheet_df)
    #print(labels)
    #print(sheet_df)
    #print(sheet_df["Speaker"])
    #print(sheet_df["Dialogue State"])
    
    state = sheet_df["Combined"].where(sheet_df["Speaker"] == state).tolist()
    state = [x for x in state if pd.isnull(x) == False and x != 'nan']
    action = sheet_df["Combined"].where(sheet_df["Speaker"] == action).tolist()
    action = [x for x in action if pd.isnull(x) == False and x != 'nan']

    # print(state)
    # print(action)
    return state, action
    #print(sheet_df["Dialogue State"].where(sheet_df["Speaker"] == "P2").notnull())
    #print(type(sheet_df["Speaker"]))


def spread():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    #Call the Sheets API
    sheet = service.spreadsheets()
    return sheet

if __name__ == '__main__':
    spreadsheet = spread()
    test_states = []
    test_actions = []
    for x in range(9):
        #if x is not 7:
        temp_state, temp_action = get_sheet(spreadsheet, SPREADSHEET_ID, x + 1, "P1", "P2")
        test_states.append(temp_state)
        test_actions .append(temp_action)
    # print(test_states)
    # print(test_actions)
