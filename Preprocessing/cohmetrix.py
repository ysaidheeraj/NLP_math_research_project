import requests
from bs4 import BeautifulSoup
import pandas as pd
import warnings
import re
warnings.filterwarnings("ignore")

captcha_mapping ={
    1 : "16OSR",
    2 : "FC85s",
    3 : "6wC1o",
    4 : "Mmbu6"
}

file_name_mapping = {
    "FILE_1" : "cohmetrix_0-500.csv",
    "FILE_2" : "cohmetrix_501-1000.csv",
    "FILE_3" : "cohmetrix_1001-1500.csv",
    "FILE_4" : "cohmetrix_1501-2000.csv",
    "FILE_5" : "cohmetrix_2001-2500.csv",
    "FILE_6" : "cohmetrix_2501-3000.csv",
    "FILE_7" : "cohmetrix_3001-3500.csv",
    "FILE_8" : "cohmetrix_3501-4000.csv",
    "FILE_9" : "cohmetrix_4001.csv"
}

start_indices = {
    1: 0,
    2: 501,
    3: 1001,
    4: 1501,
    5: 2001,
    6: 2501,
    7: 3001,
    8: 3501,
    9: 4001
}

def remove_irregular_characters(text):
    # Define the pattern to match irregular characters
    pattern = r"[^a-zA-Z0-9.,!?:;&()\[\]\"'\-—–@]"

    # Remove irregular characters using regex
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

def get_metrics():
    df = pd.read_csv(r"../Data/math_problems_Exp_1.csv")
    df.drop(columns=['problem', 'Level ', 'modified_problem', 'no_of_equations',
       'no_of_variables', 'has_exponential_or_mod', 'has_logarithm',
       'has_fraction', 'has_equality_or_inequality'], inplace=True)
    return list(df.columns)

def get_cohmetrics(text, cohmetrix):
    soup = BeautifulSoup(text, 'html.parser')

    # Find the tbody element
    tbody = soup.find('tbody')

    if tbody is None:
        return -1

    features = {}

    # Iterate over the rows in the tbody
    for row in tbody.find_all('tr'):
        # Find the columns in each row
        columns = row.find_all('td')
        if(len(columns) > 4):
            # Extract the values from the 2nd and Last but one columns
            colname = columns[1].text.strip()
            if colname in cohmetrix:
                features[columns[1].text.strip()] = columns[-2].text.strip()
    
    return features

def call_request(text, captcha):


    url = 'http://141.225.61.35/CohMetrix2017/'

    cookie_val = 'ASP.NET_SessionId=ipmnh0ipmdossnvulbfhx4gn'
    if captcha == captcha_mapping[2]:
        cookie_val = 'ASP.NET_SessionId=px0204dld2o4k2yph3kblgbf'
    elif captcha == captcha_mapping[3]:
        cookie_val = 'ASP.NET_SessionId=q1lus5u14ata44dl13vpjfpi'
    elif captcha == captcha_mapping[4]:
        cookie_val = 'ASP.NET_SessionId=0omkxddhutce01woxghepflh'
    
    print(captcha, cookie_val)

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Cookie': cookie_val,
        'Origin': 'http://141.225.61.35',
        'Referer': 'http://141.225.61.35/cohmetrix2017',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }

    data = {
        'Text': text,
        'CaptchaText': captcha,
        'submit': 'Submit'
    }

    response = requests.post(url, headers=headers, data=data, verify=False)

    print(response)
    if("Validation Failed! Try Again" in response.text or "Error" in response.text):
        return -1

    return response.text

def save_dataframe(df, FILE):
    df.to_csv(FILE, index=False)



cohmetrix = get_metrics()

file_mapping = {}
for i in range(1, 9):
    file_mapping[i] = "FILE_"+str(i)
print(file_mapping)
out_file = int(input("Enter Output file: "))
output_file = file_name_mapping[file_mapping[out_file]]

print(captcha_mapping)
captcha = captcha_mapping[int(input("Enter which captch to use: "))]
start_index = start_indices[out_file]
end_index = start_indices[out_file+1] if start_indices[out_file+1] is not None else -1

try:
    result_df = pd.read_csv(output_file)
except:
    result_df = pd.DataFrame(columns=cohmetrix)

all_data_problems = pd.read_csv(r"../Data/all_data_compressed.csv")['problem'][0]
for i,problem in enumerate(all_data_problems):
    if i < start_index:
        continue

    if end_index != -1 and i==end_index:
        break
    print("Processing problem number: {}".format(i))

    cleaned_problem = remove_irregular_characters(problem)
    resp = call_request(cleaned_problem, captcha)
    if resp == -1:
        save_dataframe(result_df, output_file)
        print("Error in request! Last stopped at: {}".format(i))
        break

    features = get_cohmetrics(resp, cohmetrix)
    
    if features == -1:
        save_dataframe(result_df, output_file)
        print("Error in parsing! Last stopped at: {}".format(i))
        break

    features['problem'] = problem
    
    result_df = result_df.append(features, ignore_index=True)
    save_dataframe(result_df, output_file)





