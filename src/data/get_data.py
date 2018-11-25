import pandas as pd
import os


class CFPBdata:
    def __init__(self):
        self.api = 'https://data.consumerfinance.gov/resource/jhzv-w97w.json'
        self.query = ''
        self.dataset_identifier - 'jhzv-w97w'
        self.APP_TOKEN = os.getenv('CFPB_APP_TOKEN')
    
    def limit_data(self, *args, **kwargs):
        '''
        Recommend calling this like:
        limit_data(limit=1e5) # or whatever limit
        '''
        # resetting query below ensures that the query resets every time 
        # this method is run, not every time the class is instantiated
        self.query = '?&'
        if limit in args:
            self.query = self.query+f'$limit={limit}'
        if date in kwargs:
            # TODO
            #$where=date%20between%20%272014-01-01T00:00:00%27%20and%20%272015-01-01T00:00:00%27'
            beginning_date = date[0]
            end_date = date[-1]
            self.query = self.query+f'$where=date%20between%20%27{beginning_date}%27%20and%20%27{end_date}'
        

    def get_data(self):
        '''
        calls down the data from the CFPB API and reads it into memory
        '''
        full_query = self.api+self.query+'?$$app_token='+self.APP_TOKEN
        return pd.read_json(full_query)

