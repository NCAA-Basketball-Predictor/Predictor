# NCAA Basketball Team Winning Percentage Predictor
* **Team Members:**
    * Brandon Clark
    * Benjamin Comer

## Files
* project-proposal.ipynb: Important details about our project
* classification_eval.ipynb: Evaluation of our three classifers
* best_classifer.p: a pickeled version of our best classifier.
* data_parser.ipynb: where we parse our data
* decision_tree.ipynb: creating variations of decision trees and making pdf versions of viewing them
* Dockerfile: creates connection between docker container and NCAA_predictor_app.py
* EDA: Our exploratory data analysis
* heroku.yml: Allows our docker file to be deployed to heroku
* NCAA_Pickeler.py: pikcles our best classifer and sends it to our app to be predicted upon and allows for it to be depolyed.
* NCAA_Predictor_app.py: Our Flask app that was deployed
* NCAA_Predictor: Creates the server
* norm_and_disc.ipynb: normalizing and discretizing the data
* random_forest.ipynb: testing random forest

## Folders
* **input_data:** 
    * contains multiple variations of the Data that we used discretized and the orginal data
* **mysklearn**
    * myclassifers.py: all of our classifers that we made
    * myevaluation.py: all of the evaluation tools that we made
    * mypytable.py: Data preparation tools
    * myutils.py: reusable utility functions
    * plotutils.py: reusable graphing utility functions
