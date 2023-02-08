# # Import necessary dependencies 
import pandas as pd
import nltk
import numpy as np
import re  #regular expressions
from nltk.stem import wordnet  # for lemmatization
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words (bow)
from sklearn.feature_extraction.text import TfidfVectorizer  #for tfidf
from nltk import pos_tag  # for parts of speech
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances  # cosine similarity
from nltk import word_tokenize
from nltk.corpus import stopwords 
nltk.download('omw-1.4')  # this seems to be a requirement for the .apply() function to work 

# %%
df_context= pd.read_excel('dialog_talk_agent.xlsx')  # read the database into a data frame
df_context.head()  # see first 5 lines

# %%
# check the Context length
length1 = df_context['Context'].map(lambda x:len(str(x)))
length1.hist(figsize=(8,6))
plt.show()

# %% [markdown]
# Why not add another dataset?

# %%
# load train data
df_question= pd.read_json('train.json')  # read the database into a data frame
df_question.head()  # see first 5 lines

# %%
# check the question length
length2 = df_question['question'].map(lambda x:len(str(x)))
length2.hist(figsize=(8,6))
plt.show()

# %%
import re

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub(' ',line)
    return ' '.join(line.split())

def extract_english(text:str):
    # remove non-alphabet characters
    words = text.lower().split(' ')
    words = [re.sub(r'[^a-z]', '', w.lower()).strip() for w in words]
    return ' '.join(words)


def remove_stopwords(words,stopwords = None):
    """
    # define a function that removes stopwords
    words: list or str
    stopwords: stopword list or str
    """
    return_list = True
    if type(words) is str:
        words = words.split()
        return_list = False
    if stopwords:
        words = [word for word in words if word not in stopwords]
    if return_list:
        return words
    else:
        return ' '.join(words)

def plot_wordcloud(content,file_prefix =None):
    import wordcloud
    import matplotlib.pyplot as plt
    w = wordcloud.WordCloud(
    width = 1600,height=1200,background_color='white',collocations=False
    )
    if type(content) == dict:
        w.generate_from_frequencies(content)
    elif type(content) == str:
        w.generate(content)
    else:
        raise ValueError("content should be frequent dict or text str")
    w.to_file(f"{file_prefix} - " if file_prefix else '' + 'word_cloud.png')
    plt.figure(dpi=400)
    plt.imshow(w,interpolation='catrom')
    plt.axis('off')
    plt.show()

