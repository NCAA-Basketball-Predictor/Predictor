# NCAA Basketball Team Winning Percentage Predictor
* **Team Members:**
    * Brandon Clark
    * Benjamin Comer


## Instructions:
**On Our Heroku App**
1. Navigate to our [homescreen](https://ncaa-predictor-app-bclark.herokuapp.com)
2. Add path for prediction and query string
e.g. /predict?Scoring_Margin=1&efg=2&spg_bpg=2&rebound_margin=2

**Locally**
1. Run command python3 NCAA_Predictor_app.py
2. Navigate to local host url it gives in command line
3. Add path for predictiona and query string
e.g. /predict?Scoring_Margin=1&efg=2&spg_bpg=2&rebound_margin=2

## Files
* final_report.ipynb: The final report.
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
    * contains multiple variations of the Data that we used discretized and the original data
* **mysklearn**
    * myclassifers.py: all of our classifers that we made
    * myevaluation.py: all of the evaluation tools that we made
    * mypytable.py: Data preparation tools
    * myutils.py: reusable utility functions
    * plotutils.py: reusable graphing utility functions
* **img**
    * contains multiple screenshots used in the final report
* **tree_vis**
    * contains tree visualizations
