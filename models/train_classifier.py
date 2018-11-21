import sys

from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
import _pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


""""
A custom extractor to be added to the NLP pipeline.
This could help extend the list of features.
"""
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath, tablename='tweets'):
    """
    Given a path to a database file (sqlite), and an optional table name.
    Load all the data from that table and turn it into a dataframe.

    Split the data into features (X) and outputs (Y)
    """
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table(tablename, engine)

    # pick only direct tweets (i.e. ignore things like news)
    # df = df[df['genre'] == 'direct']

    X = df['message']
    Y = df[[col for col in df.columns if col not in ['message', 'original', 'genre']]]

    return X, Y, Y.columns


def tokenize(text, token_type='word', lemmatize=True, stem=False):
    """
    Given a string of text. Tokenize the string into an list of strings.
    Remove any stop words. Apply lemmatization and/or stemming.
    """
    tokens = []

    if len(text) == 0:
        return ['']
    
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    if token_type is 'sentence':
        tokens = sent_tokenize(text)
    else:
        # default to words tokenization
        tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    if lemmatize:
        tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    
    if stem:
        tokens = [PorterStemmer().stem(w) for w in tokens]
    
    return tokens


def build_model(gridsearch=True):
    """
    Builds an NLP pipeline to do teh following:
    1. Tokenize
    2. Vectorize (count then tfidf)
    3. other custom extractors
    4. finally, a classifier

    The pipeline will also support methods such as .fit and .predict

    Will also apply a grid search optionally.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            # ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    if gridsearch == True:
        parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
            'features__text_pipeline__vect__max_features': (None, 5000, 10000),
            'features__text_pipeline__tfidf__use_idf': (True, False)
        }

        cv = GridSearchCV(pipeline, param_grid=parameters)

        return cv

    return pipeline


def evaluate_model(Y_test, Y_preds, category_names, binarize=False):
    """
    A method to score a given model based on its predictions.
    Scores are calculated per each prediction category.
    """
    true_df = pd.DataFrame(Y_test, columns=category_names)
    preds_df = pd.DataFrame(Y_preds, columns=category_names)

    results = []

    for c in preds_df.columns:
        precision, recall, fsore, support = \
            precision_recall_fscore_support(true_df[c], preds_df[c], average='weighted')

        results.append([c, precision, recall, fsore])


    results_df = pd.DataFrame(results, columns=['feature', 'precision', 'recall', 'f1_score'])
    
    return results_df


def save_model(model, model_filepath):
    """"
    Saves the classifier/pipeline to the disk
    """
    with open(model_filepath, 'wb') as f:
        pkl.dump(model, f)

    return True


def save_results(results, database_filename, database_tablename='results'):
    """
    Save a dataframe into the 'results' table of a database
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    results.to_sql(database_tablename, engine, index=False, if_exists='replace')
    return True


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
        results = evaluate_model(model, X_test, Y_test, category_names)
        save_results(results, database_filepath)
        
        print("\n\n")
        print(results)
        print("\n\n")

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