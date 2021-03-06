"""
Programmer: Gina Sprint and Brandon Clark
Class: CPSC 322-02, Spring 2021
Description: This file contains the neccessary components to make requests through a url
"""
import requests # lib to make http requests
import json # lib to help with parsing JSON objects

# url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&lang=Java&tweets=yes&phd=yes"
url = "http://127.0.0.0:5000/predict?Scoring_Margin=1&efg=2&spg_bpg=2&rebound_margin=2"


# make a GET request to get the search results back
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url=url)

# first thing... check the response status code 
status_code = response.status_code
print("status code:", status_code)

if status_code == 200:
    # success! grab the message body
    json_object = json.loads(response.text)
    print(json_object)