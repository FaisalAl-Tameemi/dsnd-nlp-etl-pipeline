import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import _pickle as pkl

sys.path.append('../models')

from train_classifier import tokenize


nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/figure8etl.db')
df = pd.read_sql_table('tweets', engine)
results = pd.read_sql_table('results', engine)

# load model
model = pkl.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    class_names = [c for c in df.columns if c not in ['message', 'original', 'genre']]
    class_counts = df[class_names].sum()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Class Name"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=results['feature'],
                    y=results['precision']
                )
            ],

            'layout': {
                'title': 'Metrics',
                'yaxis': {
                    'title': "Precision"
                },
                'xaxis': {
                    'title': "Feature"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=results['feature'],
                    y=results['recall']
                )
            ],

            'layout': {
                'title': 'Metrics',
                'yaxis': {
                    'title': "Recall"
                },
                'xaxis': {
                    'title': "Feature"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=results['feature'],
                    y=results['f1_score']
                )
            ],

            'layout': {
                'title': 'Metrics',
                'yaxis': {
                    'title': "F1 Score"
                },
                'xaxis': {
                    'title': "Feature"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()