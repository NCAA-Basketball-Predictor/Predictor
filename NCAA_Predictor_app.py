# we are going to use Flask, a micro web framework
import os
import importlib
import pickle 
from flask import Flask, jsonify, request 
import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils
import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyRandomForestClassifier
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation
import copy
import random

# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/", methods=["GET"])
def index():
    # return content and a status code
    return "<h1>Welcome to my App</h1>", 200

# Scoring Margin,eFG%,SPG+BPG,Rebound Margin,Win Percentage
# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
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
    header, data = myutils.load_from_file("input_data/NCAA_Statistics_24444.csv")
    random.seed(15)

    # Now, we can move to create some decision trees. Let's first create trees over the whole dataset, then
    # test upon our stratisfied k-fold splitting method.

    class_col = myutils.get_column(data, header, "Win Percentage")
    data = myutils.drop_column(data, header, "Win Percentage")
    data = myutils.drop_column(data, header, "Scoring Margin")
    atts = header[1:-1]

    X_indices = range(len(class_col))
    X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(X_indices, class_col, n_splits=10)

    my_rf = MyRandomForestClassifier()
    for fold_index in range(len(X_train_folds)):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        
        for train_index in X_train_folds[fold_index]:
            X_train.append(copy.deepcopy(data[train_index]))
            y_train.append(copy.deepcopy(class_col[train_index]))
            
        for test_index in X_test_folds[fold_index]:
            X_test.append(copy.deepcopy(data[test_index]))
            y_test.append(copy.deepcopy(class_col[test_index]))
            
        # Get a classifier in here...

    # Fitting...
        my_rf.fit(X_train, y_train, n_trees=50, m_trees=10, min_atts=2)
    # ... and predicting!

    # 2. use the tree to make a prediction
    try: 
        return my_rf.predict(instance) # recursive function
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