import streamlit as st 
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import re
import numpy as np
import contractions
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


## LOAD THE CLEANED DATASET
df = pd.read_csv('clean_data.csv')
df.dropna(inplace=True)

st.title('Music Recommender System :headphones:')


## GET INPUT FROM THE USER
if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

def disable():
    st.session_state["disabled"] = True

artist_name = st.selectbox('Select the artist name',df['artist'].unique(),
                           placeholder='Choose an Option',
                           index = None,
                           disabled=st.session_state['disabled'])
if artist_name is not None:
    song_name = st.selectbox('Select the song',df[df['artist'] == artist_name]['song'].unique(),placeholder = 'Choose an Option',index = None)
absent = st.checkbox('Not present in list',on_change=disable)
if absent:
    artist_name = st.text_input('Enter the artist name')
    song_name = st.text_input('Enter the song name')
top = st.slider('Length of List',1,20,5)


## GET URL OF THE SONG
def get_url(artist,song):
    url = 'https://www.google.com/search?q='
    if len(artist.split(' '))<2:
        url = url + artist + '+'
    else:
        for name in artist.split(' '):
            url += name + '+'
    if len(song.split(' '))<2:
        url = url + song + '+'
    else:
        for name in song.split(' '):
            url += name + '+'
    url += 'lyrics'
    return(url)


## CLEAN THE LYRICS
def clean_text(txt):

    txt = txt.lower()
    txt = contractions.fix(txt)
    txt = re.sub('\d+','',txt)
    txt = re.sub('\[[^\]]*\]','',txt)
    txt = re.sub(r'\([^)]*\)', '', txt)
    txt = re.sub('[a-zA-Z]\*+','',txt)
    txt = re.sub(r'Verse \d+','',txt)
    temp = txt.split('\r\n')
    temp = [word.strip() for word in temp if len(word.strip())!=0]

    return ' '.join(temp)

def lemmatize_text(txt):
    lem = WordNetLemmatizer()
    lemma = [lem.lemmatize(word) for word in word_tokenize(txt)]
    return ' '.join(lemma)


## GET THE LYRICS FROM WEB
def get_lyrics(artist_name,song_name):
    lyrics = []

    options = webdriver.ChromeOptions()
    options.add_argument("--incognito")

    url = str(get_url(artist_name,song_name))

    browser = webdriver.Chrome(options=options)
    browser.get(url)

    block = browser.find_element(By.CLASS_NAME,'s6JM6d')
    elements = browser.find_elements(By.CLASS_NAME,'ujudUb')
    for element in elements:
        lyrics.append(element.text)

    lyrics = ' '.join(lyrics)

    lyrics = clean_text(lyrics)

    lyrics = lemmatize_text(lyrics)

    return(lyrics)


## MAKE RECOMMENDATION
def recommend(artist,song):

    idx = df[(df['artist'] == artist) & (df['song'] == song)].index[0]

    distances = cosine_similarity(matrix[idx],matrix)
    similar = np.argsort(distances)[0][::-1]
    similar = similar[1:]

    d = {'Song':[],'Artist':[],'Match(%)':[]}

    for i in range(top):
        d['Match(%)'].append((list(distances)[0][similar[i]]*100).round(3))
        d['Artist'].append(df.loc[similar[i]]['artist'])
        d['Song'].append(df.loc[similar[i]]['song'])
    
    result = pd.DataFrame(d)
    return result


## MAKE PREDICTION
if st.button('Submit'):

    placeholder = st.empty()

    placeholder.text('Getting Lyrics...')
    lyrics = get_lyrics(artist_name,song_name)

    temp = {
        'artist':[artist_name],
        'song':[song_name],
        'text':[lyrics]
            }
    
    temp = pd.DataFrame(temp)

    df = pd.concat([df,temp])
    df.reset_index(inplace = True)

    tfid = TfidfVectorizer(analyzer='word',stop_words='english')
    matrix = tfid.fit_transform(df['text'])

    result = recommend(artist_name,song_name)
    placeholder.empty()
    st.write('Recommended Songs - \n')
    st.write(result)