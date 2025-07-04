# SpamHam
Spam Email Classification using Ensemble Learning

# Tech Stack
Data Handling: numpy and pandas\
Data Preprocessing: scikit-learn\
Boosting: xgboost\
Deployment: streamlit and joblib\
Scripting: Python

# Working
This model works by combining three other models namely; RandomForestClassifer, AdaBoostClassifier, and XGBClassifier. First the training dataset containing examples of spam and ham emails, which is present [here](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv), is loaded and converted in to numerical values so that the model can be trained using it.\
The dataset is spilt into a training and a testing set in a 4:1 ratio. Each model is then trained on the training set individually and then they are combined using majority voting\
Finally the accuracy of the model is evaluated using the test set.

# Deployment
The model and the vectorizer are saved locally using joblib. They are then loaded in the app.py script and are used to classify a user-input message/email as spam or ham.\

# How to run
Download both the transfer.py and app.py scripts.\
Run the transfer.py script.
```
python ensemble.py
```
To run the app.py script use the following command in the terminal:
```
streamlit run app.py
```

