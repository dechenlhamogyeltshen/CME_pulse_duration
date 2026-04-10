import json
import os
import datetime
import requests
import errno

import json
import os
import datetime
import requests
import errno
import time
import numpy as np
from astropy.time import Time

   
def getMetOfficeCone():
    """
    Downloads the most recent Cone CME file in a specified time window
    Args:
        startdate: A Datetime object specifying the start date of the window
        enddate: A Datetime object specifying the end date of the window
    Returns:
        success:
        cone_filepath:
        model_time:
    """
    version = 'v1'
    api_key = os.getenv("UKMO_API")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    
    startdate_orig = datetime.datetime(2023, 1, 1, 0, 0 ,0)
    enddate_final = datetime.datetime.now()
    
    # Chunk it in julian days
    start_date = Time(startdate_orig)
    end_date = Time(enddate_final)
    
    start_int = int(np.fix(start_date.jd))
    end_int = int(np.fix(end_date.jd))
    
    for julian_day in range(start_int, end_int):
    
        t = Time(julian_day+0.5, format='jd')
        
        start_date = datetime.datetime(t.datetime.year, t.datetime.month, t.datetime.day, 0, 0, 0)
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        end_date = datetime.datetime(t.datetime.year, t.datetime.month, t.datetime.day, 23, 59, 59)
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        request_url = url_base + "/" + version + "/data/swc-enlil-wsa?from=" + start_date_str + "&to=" + end_date_str
        headers_dict = {"accept": "application/json", "apikey": api_key}
        response = requests.get(request_url, headers=headers_dict)
    
        if response.status_code == 200:
    
            # Convert to json
            js = response.json()
            nfiles = len(js['data'])
            
            for i in range(nfiles):
            
                model_time = js['data'][i]['model_run_time']
                cone_file_name = js['data'][i]['cone_file']
                cone_file_url = url_base + "/" + version + "/" + cone_file_name
                cone_filepath = rename_cone_file(cone_file_name)
                
                if not os.path.exists(cone_filepath):
                                
                    response_cone = requests.get(cone_file_url, headers={"apikey": api_key})
                    time.sleep(2)
                    if response_cone.status_code == 200:
                        print(f"Downloading: {cone_filepath}")
                        with open(cone_filepath, "wb") as f:
                            f.write(response_cone.content)
                            
                    else:
                        print(f"Status code {response_cone.status_code} for {cone_file_url}")
                else:
                    print(f"File already exists: {cone_filepath}")
        else:
            print(f"Status code {response.status_code} for {cone_file_url}")
        
        time.sleep(2)    
    return


def rename_cone_file(file_in):
    """
    Function to rename Cone CME files for a more sensible archive. UKMO API file naming is WILD!
    Args:
        file_in: String name of the file to download
    Returns:
        file_out: String name of the full path to download the file to.
    """

    separator = '%2F'
    parts = file_in.split(separator)

    #  takes the parts of the filename and assigns them to what part of the date they are
    year = int(parts[2])
    month = int(parts[3])
    day = int(parts[4])
    hour = int(parts[5])
    #  combines the parts into a datetime object
    date = datetime.datetime(year, month, day, hour)
    #  converts the date into a string, adding zeroes if the month or day is only one number
    date_str = date.strftime("%Y%m%d%H")

    #  creates the new file name from the date string
    file_out = os.path.join("/data/conecme", "cone_cme_{}.in".format(date_str))

    return file_out
    

if __name__ == "__main__":

    getMetOfficeCone()
