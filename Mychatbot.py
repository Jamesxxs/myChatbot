# %%
# Import necessary dependencies 
import pandas as pd
import nltk # natural language toolkit
import numpy as np
import matplotlib.pyplot as plt
import re  #regular expressions
import wordcloud
from textblob import TextBlob
from os import path
from PIL import Image
from nltk.stem import wordnet  # for lemmatization
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words (bow)
from sklearn.feature_extraction.text import TfidfVectorizer  #for tfidf
from nltk import pos_tag  # for parts of speech
from sklearn.metrics import pairwise_distances  # cosine similarity
from nltk import word_tokenize
from nltk.corpus import stopwords
# Dowloads necessary resources
nltk.download('omw-1.4')  
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# %%
# Load a Excel file of dialog dataset into a pandas DataFrame and displays the first 5 rows.
df_context= pd.read_excel('dialog_talk_agent.xlsx')  
df_context.head() 

# %%
# Plot histogram of string length vs frequency based on the dialog dataset
# Map each elements to its string length
length1 = df_context['Context'].map(lambda x:len(str(x)))
# Define the figure size
length1.hist(figsize=(8,6))
# Define the font
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
# Set title and labels
plt.title("Dialog_dataset", fontdict = font1)
plt.xlabel("string length", fontdict = font2)
plt.ylabel("frequency", fontdict = font2)
plt.show()

# %% [markdown]
# Load another dataset (Tv show dataset)

# %%
# Load json file of another training data (tv show related dataset) into a pandas DataFrame
# and displays the first 5 rows.
df_question= pd.read_json('train.json') 
df_question.head() 

# %%
# Plot the histogram of string length vs frequency based on the new training dataset
# Map each elements to its string length
length2 = df_question['question'].map(lambda x:len(str(x)))
# Set figure size
length2.hist(figsize=(8,6))
# Set title and labels
plt.title("Training_dataset", fontdict = font1)
plt.xlabel("string length", fontdict = font2)
plt.ylabel("frequency", fontdict = font2)
plt.show()

# %%
def remove_punctuation(line):
    """
    This function removes any punctuation characters from a given string, keeping only alphanumeric characters
    and Chinese characters. It then returns the resulting string with extra whitespace removed.

    Args:
    line (str): the input string to remove punctuation from

    Returns:
    str: the input string with all punctuation removed and extra whitespace removed
    """
    line = str(line)
    # If input is empty or all whitespace, return an empty string
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub(' ',line) 
    return ' '.join(line.split()) 

def extract_english(text:str):
    """
    A function that takes in a string and extracts only the English alphabet characters from it.
    
    Args:
    - text: A string containing text that may include non-alphabet characters
    
    Returns:
    - A string containing only the English alphabet characters (in lowercase), separated by a single space.
    """
   # Split the text into individual words
    words = text.split(' ')

    # Use a regular expression to remove non-alphabet characters from each word
    english_words_only = [re.sub(r'[^a-z]', '', w.lower()).strip() for w in words]
    return ' '.join(english_words_only)


# Stopwords downloaded from the dependency nltk
STOPWORDS_LIST = stopwords.words('english')

def remove_stopwords(words,stopwords = None):
    """
    Function to remove stopwords from a list or string of words. and return it as its original
    type
    
    Args:
    words: A list or str of words to remove stopwords from.
    stopwords: A list or str of stopwords to remove. If None, use default stopwords.
    
    Returns:
    A list of words with stopwords removed or a string of words without stopwords.
    """
    return_list = True
    # Check if words is a string or list, and set return_list flag accordingly.
    if type(words) is str:
        words = words.split()
        return_list = False
    # If stopwords is provided, remove stopwords from words.
    if stopwords:
        words = [w for w in words if w not in stopwords]
     # If return_list flag is True, return list of words. Otherwise, return string of words.
    if return_list:
        return words
    else:
        return ' '.join(words)

# %%
# Get current working directory
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# Load a bubble mask for the wordcloud 
bubble_mask = np.array(Image.open(path.join(d, "bubble_mask.png")))

def plot_wordcloud(content,file_prefix =None):
    """
    Plots a image which shows some frequent words in a given dataset and stores the image as 'word_cloud.png'
    Arguments:
    content: dict or str, the content to be plotted
        if dict, it should be a frequency dictionary with keys as words and values as their corresponding frequencies
        if str, it should be the raw text to be plotted
    file_prefix: str, optional, the prefix to be added to the image file name
    
    Returns: 
    None
    """ 
    wc = wordcloud.WordCloud(
    width = 1600,height=1200,background_color='white',collocations=False,mask=bubble_mask
    )
    if type(content) == dict:
        wc.generate_from_frequencies(content)
    elif type(content) == str:
        wc.generate(content)
    else:
        raise ValueError("content should be frequent dict or text str")
        
    # store the generated image
    wc.to_file(f"{file_prefix} - " if file_prefix else '' + 'word_cloud.png')
    
    plt.figure(dpi=400)
    plt.imshow(wc,interpolation='catrom')
    plt.axis('off')
    plt.show()

# %%
# Applying preprocessing steps to the 'questuon' of the main training dataset (TV_show dataset)
questions = df_question['question'].map(lambda x: remove_stopwords(
                                                    extract_english(
                                                    remove_punctuation(x)
                                                    ),
                                                    stopwords=STOPWORDS_LIST))
print(questions)
print(type(questions))
# Convert series to str
questions = ' '.join(questions)
# Plot word cloud image, the size of each word is proportional to its frequency in the dataset
plot_wordcloud(questions)

# %% [markdown]
# Get rid of these unnecessary columns, stop words and brackets in the training dataset, and combine
# two datasets into one

# %%
# Delete all columns other than 'question' and 'nq_answer'
df_question = df_question.drop(
            columns=['viewed_doc_titles', 'used_queries', 'annotations', 'id', 'nq_doc_title']
)
# Swap the order of 'questions' and 'nq_anwer'.
df_question = df_question.reindex(columns=['question', 'nq_answer'])  
# Rename the column
df_question = df_question.rename(
    columns={'question': 'Context', 'nq_answer': 'Text Response'}
)
# Check first 5 rows
df_question.head()

# %%
# Remove brackets in the Text Responses
def remove_brackets(text):
    """
    Removes square brackets from a string.
    
    Args:
        text (str): The input text to remove brackets from.
        
    Returns:
        str: The input text with square brackets removed.
    """
    # Replace opening & closing bracket with empty space
    new_text = str(text).replace('[', '')  
    new_text = str(new_text).replace(']', '') 
    # Return modified text 
    return new_text

# Apply remove_brakets function to the 'Text Response' column and check first 5 rows
df_question['Text Response'] = df_question['Text Response'].apply(remove_brackets) 
df_question.head()

# %%
def remove_first_and_last_character(text):
    """
    A function that removes the first and last characters (ie. apostrophes) from a string.

    Parameters:
    text (str): The input string from which the first and last characters will be removed.

    Returns:
    str: The input string with the first and last characters removed.
    """
    return str(text)[1:-1]  # slice from the second character till the last one (excluding the last one)

# Remove apostrophes in theText Response' column
df_question['Text Response'] = df_question['Text Response'].apply(remove_first_and_last_character) 
df_question.head()

# %%
# Compare the number of rows (data) in each dataset
print("Number of rows in the original dataset: ", df_context.shape[0]) #(1592, 2)
print("Number of rows in the new dataset: ", df_question.shape[0]) #(10036, 2)

# %%
# Create blank dataframe
df = pd.DataFrame() 
# Combine two dataset 
# Unpack two lists and concatenate them into one and store it in 'column1/2'
column1 = [*df_context['Context'].tolist(), *df_question['Context'].tolist()]  
column2 = [*df_context['Text Response'].tolist(), * df_question['Text Response'].tolist()] 
# Create a new dataframe with only 2 columns(context & text response)
df.insert(0, 'Context', column1, True)  # insert the first column
df.insert(1, 'Text Response', column2, True)  # insert the second one
# Show the row number of the combined dataset
print("Number of rows in the combined dataset: ", df.shape[0]) # （11628，2）
df.head() 


# %%
# Fill missing values with the most recent non-missing value in the same row
df.ffill(axis = 0, inplace = True)  
# Check firts 5 rows
df.head()  

# %% [markdown]
# # Preprocess the combined data, including tokenization and lemmatization 

# %%
def cleaning(text):
    """
    A function to clean a list of strings by converting them to lowercase, removing non-alphanumeric characters, 
    and returning the cleaned list.
    
    Args:
    text (list): A list of strings to be cleaned.
    
    Returns:
    list: A list of cleaned strings.
    """
    # Create empty list for cleaned data
    cleaned_lst = list()
    for i in text:
        # convert input to string and make it all lowercase letters
        lc_word = str(i).lower()  
        # remove any non-alphanumeric characters
        cleaned_word = re.sub(r'[^a-z0-9]', ' ', lc_word)  
        # append cleaned str to the list
        cleaned_lst.append(cleaned_word)  

    return cleaned_lst


# Create extra column in our dataset to see how the cleaned text looks like in comparison with the original
df.insert(1, 'Cleaned Context', cleaning(df['Context']), True)
df.head()

# %%
def text_normalization(text):
    """
    A function to perform text normalization on input text by converting it to lowercase, removing 
    non-alphabetic characters, tokenizing it, lemmatizing the tokens based on their parts of speech, 
    and returning the normalized text as a string.

    Args:
    text (str): The input text to be normalized.

    Returns:
    str: The normalized text as a string.
    """
    # Convert input text to all lowercase letters
    text = str(text).lower()  
    # Remove any special characters including numbers
    spl_char_text = re.sub(r'[^a-z]', ' ', text)  
    # Tokenize text
    tokens = word_tokenize(spl_char_text) 
    # Lemmatizer initiation
    lema = wordnet.WordNetLemmatizer()  
    # Parts of speech (default: Penn Treebank tagset)
    tags_list = pos_tag(tokens, tagset = None)  
    lema_words = []

    for token, pos_token in tags_list:
        # If the tag from tag_list is a verb, assign 'v' to it's pos_val
        if pos_token.startswith('V'):  
            pos_val = 'v'
        # Adjective
        elif pos_token.startswith('J'):  
            pos_val = 'a'
        # Adverb
        elif pos_token.startswith('R'):  
            pos_val = 'r'
        # Noun
        else: 
            pos_val = 'n'
        # Performing lemmatization
        lema_token = lema.lemmatize(token, pos_val)  
        # Add the lemmatized words to the list
        lema_words.append(lema_token) 

    return " ".join(lema_words)  

# %%
# Apply the normalization function to the 'context' column in the combined dataset
normalized = df['Context'].apply(text_normalization)
df.insert(2, 'Normalized Context', normalized, True)
df.head()


# %%
def removeStopWords(text):
    """
    A function that takes in a sentence and removes all the stop words from it using the stopwords module of the 
    nltk library. 

    Args:
    text (str): The input sentence to remove stop words from.

    Returns:
    str: The cleaned sentence with stop words removed.
    """
    lst= []
    # split the input sentence into individual words
    words = text.split()  
    # obtain the list of English stop words from nltk
    stop = stopwords.words('english')
    # for every word in the given sentence, if the word is a stop word ignore it,
    # else add it to the end of list
    for w in words:
        if w in stop:
            continue
        else:  
            lst.append(w)
    # join the remaining words in the list to form a sentence        
    return " ".join(lst) 

# %% [markdown]
# Bag of words

# %%
def text_extract(texts:list,vocabulary=None,model_type = 'tfidf',n_features = 1000):
    """
    Extracts features from a list of texts using a vectorizer (either TfidfVectorizer or CountVectorizer).

    Args:
        texts (list): A list of strings, where each string represents a document.
        vocabulary (list or None): A list of feature names to be used for the vectorizer. If None, the vectorizer will create 
                                   its own vocabulary.
        model_type (str): The type of vectorizer to use. Must be either 'tfidf' or 'bow'. Defaults to 'tfidf'.
        n_features (int): The maximum number of features to extract. Defaults to 1000.

    Returns:
        df_array (pd.DataFrame): A Pandas DataFrame containing the extracted features, where each row represents a document
                                 and each column represents a feature.
        vectorizer: The vectorizer object used for feature extraction.
    """
    # Prints the model type that is being used for feature extraction
    print(f"Extracting {model_type} features...")

    # Assigns the appropriate vectorizer based on the model_type parameter
    if model_type == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif model_type =='bow':
        Vectorizer = CountVectorizer
    else:
        # Raises an exception if an invalid model_type parameter is passed
        raise Exception("model_type should be tfidf or bow")
    vectorizer = Vectorizer(
                        #max_df=0.95, # Maximum document frequency
                        #min_df=2, # Minimum document frequency
                        #max_features=n_features, # Maximum number of features to extract
                        vocabulary = vocabulary # vacabs
                        )
    
    # Transforms the list of texts into a numpy array of features
    array = vectorizer.fit_transform(texts).toarray()
    print(f"arr shape: {array.shape}")

    # Converts the numpy array into a Pandas DataFrame and assigns column names
    df_array = pd.DataFrame(array,columns=vectorizer.get_feature_names_out())
    
    # Returns the DataFrame of features and the fitted vectorizer object
    return df_array,vectorizer


df_tfidf,tfidf = text_extract(df['Normalized Context'],model_type = 'tfidf')
df_bow,bow = text_extract(df['Normalized Context'],model_type = 'bow')

# %%
def chat_bow(question):
    """
    Returns a response to a user's question using a chatbot model based on the Bag of words algorithm.

    Parameters:
    question (str): The user's question.

    Returns:
    str: The chatbot's response to the user's question.
    """
    # apply text normalization and stop words removing
    tidy_question = text_normalization(removeStopWords(question))  
    # create counter vector for preprocessed user's question 
    cv = bow.transform([tidy_question]).toarray()  
    # calculate cosine similarity between question in dataset and input question
    cos = 1- pairwise_distances(df_bow, cv, metric = 'cosine')  
    # find the index of the maximum cosine value
    index_value = cos.argmax()  
    # Retrieve the corresponding response from the 'Text Response' column of the original DataFrame
    return df['Text Response'].loc[index_value]  

# %%
# call the chat_bow function with the question as an argument
chat_bow('Will you help me and tell me more about yourself?')


def chat_tfidf(question):
    """
    Returns a response to a user's question using a chatbot model based on the TF-IDF algorithm.

    Parameters:
    question (str): The user's question.

    Returns:
    str: The chatbot's response to the user's question.
    """
    # apply text normalization and stop words removing
    tidy_question = text_normalization(removeStopWords(question)) 
    # create tf-idf vector for preprocessed user's question 
    tf = tfidf.transform([tidy_question]).toarray()  
    # Calculate the cosine similarities between the question vector and the preprocessed data
    cos = 1- pairwise_distances(df_tfidf, tf, metric = 'cosine') 
    # Find the index of the highest cosine similarity score
    index_value = cos.argmax()  
    # Retrieve the corresponding response from the 'Text Response' column of the original DataFrame
    return df['Text Response'].loc[index_value]  
  
# %%
# call the chat_tfidf function with the question as an argument
chat_tfidf('who is your favorite star?')

# %%
from textblob import TextBlob

def senti(text):
    """
    Analyzes the sentiment of a given text using TextBlob.

    Parameters:
    text (str): The text to analyze.

    Returns:
    float: A sentiment polarity score between -1.0 (negative) and 1.0 (positive).
    """
    # Create a TextBlob object from the given text
    blob = TextBlob(text)
    # Get the sentiment polarity score from the TextBlob object
    sentiment = (blob.polarity)
    # Return the sentiment polarity score
    return (blob.polarity)

print("polarity",senti("This is good"))
print("polarity",senti("This is not good"))

def chatbot():
    """
    uses either the bag-of-words or TF-IDF model to generate responses.

    The chatbot prompts the user to choose a model and then accepts user input in the form of questions.
    The chatbot uses the chosen model to generate a response to each question.
    The chatbot will continue accepting questions until the user enters 'q' to exit.

    Returns:
    None
    """
    # Set up chatbot variables
    exit_chatbot = False
    first_loop = True
    method = None

    # Welcome message and prompt to choose model
    print(30*"="+" Welcome to ChatBot "+30*"=" + "\n")
    # select the model
    while method not in ['tfidf','bow']:
        method = input("Choose the model: tfidf or bow(bag of words): ")
        if method == 'q':
            print("Bye!")
            return # exit
    
    # Main chatbot loop
    while exit_chatbot == False:
        if (first_loop):
            print("Welcome to the chatbot! Type q to close it, otherwise let's keep talking :)")
            first_loop = False

        # Prompt user for a question
        user_input_question = input("Your question:")
        if(senti(user_input_question) < 0) :
            print("(emotion: \U0001F44E)") # Negative emotion
        else: 
            print("(emotion: \U0001F44D)") # Positive emotion
        
        # Check if user wants to exit
        if(user_input_question.lower() == 'q'): 
            exit_chatbot = True
            print("Thank you for your time and see you around!")
        else :
            # Generate a response using the chosen model
            if (method == 'bow') : 
                print('Chatbot answer: ', chat_bow(user_input_question))
            elif (method == 'tfidf') : 
                print('Chatbot answer: ', chat_tfidf(user_input_question))
            
chatbot()

