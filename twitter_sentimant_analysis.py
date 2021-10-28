# %%
from enum import Enum
from typing import List, AnyStr, Callable, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers.pipelines.base import Pipeline
import tweepy as tw
import plotly.express as px
import os
import json
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, pipeline)
import torch
from icecream import ic
import numpy as np
import pandas as pd
import textwrap

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import IPython.display
from IPython.display import display, clear_output

import plotly.graph_objects as go

import datetime

import dash
from dash import dcc
from dash import html
import plotly
from dash.dependencies import Input, Output, State, ALL, ALLSMALLER, MATCH, DashDependency

# %%
# help(go.Stream)
# %%
# %%
api_key = os.environ['TWITTER_API_KEY']
api_secret = os.environ['TWITTER_API_KEY_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_token_secret = os.environ['TWITTER_ACCESS_TOKEN_SECRET']
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
# %%
#  API Setup
# auth = tw.OAuthHandler(api_key, api_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tw.API(auth, wait_on_rate_limit=True)
# hashtag = "#VenomReleasesInMumbai"
# query = tw.Cursor(api.search_tweets, q=hashtag).items()
# tweets = [{'Tweet': tweet.text, 'Timestamp': tweet.created_at} for tweet in query]
# df = pd.DataFrame.from_dict(tweets)
# df.head()

# %%
print('Loading pipeline...')
classifier = pipeline('sentiment-analysis', model=model_name)
print('Pipeline load completed.')

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %%


def classify_sentiment(tweets: Union[AnyStr, Sequence[AnyStr]], classifier: Pipeline) -> Sequence[int]:

    return list(
        pd.DataFrame.from_dict(classifier(tweets))['label']
        .apply(lambda x: x.split()[0])
        .astype(int)
        .values
    )


print(classify_sentiment(["This is worse", "This is fun"], classifier))


# %%

'''
def get_sentiment_score(tweet_text: AnyStr, model, tokenizer):
    classification_result = model(tokenizer.encode(tweet_text, return_tensors='pt'))
    return int(torch.argmax(classification_result.logits)) + 1


class SentimentAnalyzer():
    def get_scores(tweet: AnyStr, func: Callable[[AnyStr, Union[Pipeline, (AutoModelForSequenceClassification, AutoTokenizer)]], int]) -> List[int]:
        return func(tweet)

class PlotType(Enum):
    Current = 1
    Cumulative = 2


class PeriodType(Enum):
    Current = 1
    Day = 2
    Week = 3
    ByWeek = 4
    Month = 5
    Quarter = 6
    HalfYear = 7
    Year = 8
    BiYear = 9
    ThreeYear = 10
    FiveYear = 11


class Plotter:
    def plot_sentiments(plot_type: PlotType, period_type: PeriodType):
        pass



class LayoutGenerator:
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def get_layout(self):
        LayoutGenerator.app.layout = html.Div(
            html.Div([
                html.H4('Live Twitter Sentiment analysis'),
                html.Div(id='live-update-tweet'),
                dcc.Graph(id='live-update-graph')
            ])
        )

    @app.callback(Output('live-update-tweet', 'children'))
    def update_tweet(tweets):
        style = {'padding': '5px', 'fontSize': '16px'}
        return html.Span(tweets, style=style)
'''

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True,prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])


app.layout = html.Div(
    html.Div([
        html.H4('Live Twitter Sentiment analysis'),
        html.Div(id='live-update-tweet'),
        # dcc.Graph(id='live-update-graph'),
        html.Div(id='1', style={'display': 'none'}, children=None)
    ])
)


@app.callback(Output('live-update-tweet', 'children'), Input('1', 'children'), prevent_initial_call=True)
def update_tweet(tweets):
    print("update_tweet called")
    style = {'padding': '5px', 'fontSize': '16px'}
    return html.Span(tweets, style=style)


class Listener(tw.Stream):
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, *, chunk_size=512, daemon=False, max_retries=10, proxy=None, verify=True):
        super().__init__(consumer_key, consumer_secret, access_token, access_token_secret, chunk_size=chunk_size, daemon=daemon, max_retries=max_retries, proxy=proxy, verify=verify)

        self.df_cols = ["text", "created_at"]
        self.tweets_df = pd.DataFrame(columns=[*self.df_cols, "score"])

    def on_data(self, raw_data):
        data = json.loads(raw_data)
        data = [data.get(field) for field in self.df_cols]
        temp_df = pd.DataFrame(data=data).T
        temp_df.columns = self.df_cols
        tweets = list(temp_df['text'].apply(lambda x: str(x)[:512]).values)
        temp_df["score"] = classify_sentiment(tweets, classifier)
        # app.callback(Output('live-update-tweet', 'children'))(update_tweet)
        # update_tweet("<br>".join(tweets))
        self.tweets_df = self.tweets_df.append(temp_df, ignore_index=True)

        del temp_df
        return super().on_data(raw_data)

    def on_status(self, status):
        return super().on_status(status)

    def on_request_error(self, status_code):
        if status_code == 420:
            return False
        return super().on_request_error(status_code)

    def on_connection_error(self):
        self.disconnect()

    def start(self, keywords):
        return super().filter(track=keywords)


# %%
listener = Listener(api_key, api_secret, access_token, access_token_secret)
print("Startting server")
_ = app.run_server(debug=True)
print("Streaming tweets")
_ = listener.start(['Python'])
# %%
