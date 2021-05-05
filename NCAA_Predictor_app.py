"""
Programmer: Gina Sprint and Brandon Clark
Class: CPSC 322-02, Spring 2021
Description: Creates a flask app to be deployed to a heroku server
"""
# we are going to use Flask, a micro web framework
import os
import importlib
import pickle 
from flask import Flask, jsonify, request 


# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/", methods=["GET"])
def index():
    """ The purpose of this function is to create the home page of the website.
    """
    # return content and a status code
    return "<h1>Welcome to my App</h1>", 200

# Scoring Margin,eFG%,SPG+BPG,Rebound Margin,Win Percentage
# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    """ The purpose of this function is to predict a value based off of the random forest classifer based on entered values.
    Returns:
        - jsonified prediction of the given values or nothing if it does not work
    """
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    Scoring_Margin = request.args.get("Scoring Margin", "")
    efg = request.args.get("eFG%", "")
    spg_bpg = request.args.get("SPG+BPG", "")
    rebound_margin = request.args.get("Rebound Margin", "")
    print("level:", Scoring_Margin, efg, spg_bpg, rebound_margin)
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    prediction = predict_winning_percentage_well([Scoring_Margin, efg, spg_bpg, rebound_margin])
    # if anything goes wrong, predict_interviews_well() is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400


def predict_winning_percentage_well(instance):
    """ The purpose of this function is to read in the pickeled values and return the prediction
    ARGS:
        - instance: the testing instance
    Returns:
        - prediction value or nothing 
    """
    infile = open("best_classifier.p", "rb")
    header, my_rf = pickle.load(infile)
    infile.close()
    # 2. use the tree to make a prediction
    try: 
        return my_rf.predict([instance])[0]# recursive function
    except:
        return None




if __name__ == "__main__":
    # deployment notes
    # two main categories of how to deploy
    # host your own server OR use a cloud provider
    # there are lots of options for cloud providers... AWS, Heroku, Azure, DigtalOcean, Vercel, ...
    # we are going to use Heroku (Backend as a Service BaaS)
    # there are lots of ways to deploy a flask app to Heroku
    # 1. deploy the app directly as a web app running on the ubuntu "stack" 
    # (e.g. Procfile and requirements.txt)
    # 2. deploy the app as a Docker container running on the container "stack"
    # (e.g. Dockerfile)
    # 2.A. build the docker image locally and push it to a container registry (e.g. Heroku's)
    # **2.B.** define a heroku.yml and push our source code to Heroku's git repo
    #  and Heroku will build the docker image for us
    # 2.C. define a main.yml and push our source code to Github, where a Github Action
    # builds the image and pushes it to the Heroku registry

    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000