import pandas as pd
import os

# the base API
api = "https://data.consumerfinance.gov/resource/jhzv-w97w.json"

# the number of records we're going to request
query = '?&$limit=100000'

# additional API specifications I'm not using right now
#$where=date%20between%20%272014-01-01T00:00:00%27%20and%20%272015-01-01T00:00:00%27'
    
# I have an app token in case we need that, but I haven't so far
dataset_identifier = 'jhzv-w97w'
APP_TOKEN = os.getenv('CFPB_APP_TOKEN')
token = '?$$app_token='

# total query we're running at the moment
full_query = api+query

# calls down the data from the CFPB API and reads it into memory
# remember that we only requested 100,000 records (total is ~750k)
cfpb = pd.read_json(full_query)

# saves the data as a csv
cfpb.to_csv('test_data/cfpb_sample_data.csv',encoding='utf-8',index=False)

# this squeezes in under the 50M limit at which github yells at you
