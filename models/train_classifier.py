import sys
import re
import nltk
import pickle
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def load_data(database_filepath):
    """
    Load from SQLite database to use in modeling
    params:
        - database_filepath: the path of SQLite database
    return:
        - X: features to be used to train model
        - y: labels for each observation
        - y.columns: categories' names
    """
    # read data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath)).connect()
    df = pd.read_sql_table('cleaned_data', engine)
    
    # drop unnecessary columns and seperate train and test sets
    df = df.drop(columns=['id', 'original', 'genre'])
    X = df.loc[:, 'message']
    y = df.drop(columns=['message'])
    
    return (X, y, y.columns)

def tokenize(text):
    """
    Function to tokenize the text.
    params:
        - text: input text
    return:
        - a list of words from the text but normalized.
    """
    # prepare Stemmer and Lemmatizer
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()

    # normalization
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # stop word removal
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    
    # lemmetization    
    # tag word
    pos_tagged = nltk.pos_tag(tokens)
    
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    simple_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmed = []
    for word, tag in simple_tagged:
        if tag == None:
            lemmed.append(lemmer.lemmatize(word))
        else:
            lemmed.append(lemmer.lemmatize(word, tag))
    
    return lemmed


def build_model():
    """
    Construct the models with attached GridSearch
    params: None
    return: A model with GridSearch inside
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__max_depth': [3, 4]
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model with test data.
    params:
        - model: the trained model to be used to test data
        - X_test: feature on test dataset
        - Y_test: labels on test dataset
        - category_names: As its name.
    return: None
    """
    # predict the model
    Y_predict = model.predict(X_test)

    # iterate through each category to calculate final metrics
    for i, val in enumerate(category_names):
        # prepare category
        y_pred, y_test = Y_predict[:, i], Y_test.iloc[:, i]
        
        # calculate metrics
        result = precision_recall_fscore_support(y_test, y_pred)
        accuracy = (y_pred == y_test).mean()
        precision = result[0]*result[3]/sum(result[3])
        recall = result[1]*result[3]/sum(result[3])
        
        print("{}".format(val))
        print("\tAccuracy: {}\t% Precision: {}\t% Recall: {}\n".format(accuracy, precision, recall))


def save_model(model, model_filepath):
    """
    Serialize the model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()