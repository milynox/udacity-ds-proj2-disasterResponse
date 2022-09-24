# Disaster Response Pipeline Project
## Introduction
When a disaster happens, there are millions and millions of communication such as messages, news, chat, etc either direct or via social media. And the disaster response organization with the least capacity can hardly filter and pull out the most important messages. It really is only one in thousands of messages is important. Then, these important messages will be handled by different organizations. For example, one organization will care about water, another care about blocked road and another will care about medical supplies.

As there are millions of messages communicated, spending human effort on filtering these would take a huge of time and the disaster response organization may not have enough human resource to adapt with other priorities in time.

Building a machine learning pipeline to help classifying messeages that are relevant would be very helpful for disaster response organization while a disaster happens.

## Project structure
This projects contains three folders: 
1. `app`: includes the source code of Flask app used to input the message and show the its class.
2. `data`: stores the raw data (csv format) and SQLite database that need for training and testing the model. It also contains the ETL steps used to transform raw data into SQLite database.
3. `models`: stores the source code for the modeling and a model named `classifier.pkl` used as backend to classify messages.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
