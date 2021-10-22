# %% [markdown]
'''
<h1>Tripadvisor Review : Scrapping, Sentiment Analysis using BERT</h1>

<h2 style="color: red;">Scrapping may not be allowed in many websites. This tutorial is educational purpose only. Confirm leagal issues before scrapping</h2>
'''
# %% [markdown]
'''
## Environment and Library installation
 1. Python Version 3.8
 1. pip install transformers requests beautifulsoup4 pandas numpy iteration_utilities
 1. pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
'''

# %% [markdown]
'''
<h2>Imports and variables</h2>
'''
# %%
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from selenium import webdriver
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, pipeline)
from iteration_utilities import flatten
base_url = "https://www.tripadvisor.com"
review_class_names = {"div": "pIRBV _T"}
url_class_name_in_cat_pages = "fLhRg b S7 W o q"
link_class_name = "iPqaD _F G- ddFHE eKwUx ecmMI"
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'


# %% [markdown]
'''
<h2>Scrap reviews</h2>
'''
# %%


def get_soup(url) -> BeautifulSoup:
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    driver.maximize_window()
    return BeautifulSoup(driver.page_source, 'html.parser')


# f"{base_url}{result['href']}"
# %%
soup_home = get_soup(base_url)
results = soup_home.find_all('a', {'class': link_class_name})
review_urls = [result['href'] for result in results if result['href'].startswith("/Hotel_Review-")]
category_urls = [result['href'] for result in results if result['href'].startswith("/VacationRentals-")]

# %%


def get_review_text(review_class_names, url) -> np.ndarray:
    return [
        review.text
        for review in
        np.array(
            [
                get_soup(url).find_all(el, {'class': review_class_names[el]})
                for el in review_class_names
            ], dtype='object').ravel()
    ]


# %%
reviews = [get_review_text(review_class_names, f"{base_url}{review_url}") for review_url in review_urls]
# %%
reviews = list(flatten(reviews))
# %%
reviews_df = pd.DataFrame(reviews, columns=["reviews"])
reviews_df.head()


# %%
reviews_df['reviews_clipped'] = reviews_df['reviews'].apply(lambda x: x[:512])

# %% [markdown]
'''
<h2>Using pipeline</h2>
'''
# %%
classifier = pipeline('sentiment-analysis', model=model_name)
# %%
reviews_df_classifier = reviews_df.copy()
# %%
reviews_df_classifier.head()

# %%


def classify_review(review):
    res = classifier(review)[0]
    return res.get('label'), res.get('score')


# %%
reviews_df_classifier['label'], reviews_df_classifier['score'] = zip(*reviews_df_classifier['reviews_clipped'].map(classify_review))

# %%
reviews_df_classifier.head()
# %% [markdown]
'''
<h2>Using classes</h2>
'''
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# %%


def get_sentiment_score(review_text):
    classification_result = model(tokenizer.encode(review_text, return_tensors='pt'))
    return int(torch.argmax(classification_result.logits)) + 1


# %%
reviews_df_model = reviews_df.copy()

# %%
reviews_df_model['score'] = reviews_df_model['reviews_clipped'].map(get_sentiment_score)
reviews_df_model.head()
# %%
