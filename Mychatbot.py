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
STOPWORDS_LIST = stopwords.words('english')


questions = df_question['question'].map(lambda x: remove_stopwords(
                                                    extract_english(
                                                    remove_punctuation(x)
                                                    ),
                                                    stopwords=STOPWORDS_LIST))
questions = ' '.join(questions)
word_frequency(questions.split(' '))
plot_wordcloud(questions)

# %% [markdown]
# Now, let's make this dataset look similar to our original one, it is: two columns with the same headings

# %%
# delete columns other than question and nq_answer
df_question = df_question.drop(
            columns=['viewed_doc_titles', 'used_queries', 'annotations', 'id', 'nq_doc_title']
)
df_question = df_question.reindex(columns=['question', 'nq_answer'])  # swap the order for questions to be first
df_question = df_question.rename(
    columns={'question': 'Context', 'nq_answer': 'Text Response'}
)
print(df_question.head())

# %%
# there are brackets included in the Text Responses
def remove_brackets(text):
    new_text = str(text).replace('[', '')  # replace left square bracket character with nothing
    new_text = str(new_text).replace(']', '')  # do the same with the right one
    return new_text

df_question['Text Response'] = df_question['Text Response'].apply(remove_brackets)  # remove all the brackets for the selected column
df_question.head()

# %%
# now we have some apostrophes at the start and end of our responses...
def remove_first_and_last_character(text):
    return str(text)[1:-1]  # slice from the second character till the last one (excluding the last one)

df_question['Text Response'] = df_question['Text Response'].apply(remove_first_and_last_character)  # execute it for the whole column
df_question.head()

# %%
# compare the number of rows for each dataset
print("Number of rows in the original dataset: ", df_context.shape[0])
print("Number of rows in the new dataset: ", df_question.shape[0])

# %%
df = pd.DataFrame()  # create blank dataframe'
column1 = [*df_context['Context'].tolist(), *df_question['Context'].tolist()]  # make a list out of the first columns of both datasets
column2 = [*df_context['Text Response'].tolist(), * df_question['Text Response'].tolist()]  # make a second column by combining both
df.insert(0, 'Context', column1, True)  # insert first column
df.insert(1, 'Text Response', column2, True)  # insert second one
print("Number of rows in the combined dataset: ", df.shape[0])

# %%
df.head() 

# %% [markdown]
# Null values are present for the same type of questions whose response can be almost similar and in that similar group of questions, the response is given to the first and the rest filled with null. So, what we can do is use `ffill()` which returns the value of previous response in place of null values as below.

# %%
df.ffill(axis = 0, inplace = True)   # fill the null value with the previous value
df.head()  # see first 5 lines

# %% [markdown]
# # Preprocess data

# %% [markdown]
# ### Convert text into lower cases and remove special characters and numbers

# %%
def cleaning(x):
    cleaned_array = list()
    for i in x:
        # convert to all lower letters
        a = str(i).lower()  
        
        # remove any special characters but keep numbers
        p = re.sub(r'[^a-z0-9]', ' ', a)  
        
        # add variable p to our array names cleaned_array
        cleaned_array.append(p)  
    return cleaned_array


# Create extra column in our dataset just for fun and to see how the cleaned text looks like 
# in comparison with the original
df.insert(1, 'Cleaned Context', cleaning(df['Context']), True)
# first argument indicates position we want this column to be slotted in, second is the name of the new column,
# third is the array we want to you to fill the rows with and 
# the last boolean indicates whether to allow duplicates
df.head()


