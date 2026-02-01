import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airline Sentiment Dashboard", layout="wide")

st.title("Sentiment Analysis Of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis Of Tweets about US Airlines")
st.markdown("This application is a Streamlit dashboard to analyze the sentiment of tweets about US Airlines ✈️.")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of tweets about US Airlines ✈️.")

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("Tweets.csv")
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

# --- Random Tweet Section ---
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# --- Chart Section ---
st.sidebar.markdown("## Number Of Tweets by Sentiment")
select = st.sidebar.selectbox('Visualization Type', ['Histogram', 'Pie chart'], key='chart_select')
sentiment_count = data['airline_sentiment'].value_counts().reset_index()
sentiment_count.columns = ['Sentiment', 'Tweets']

if not st.sidebar.checkbox("Hide Charts", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Histogram':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, names='Sentiment', values='Tweets')
        st.plotly_chart(fig)

# --- Map Section Logic ---
# We create a copy for the map to avoid breaking the charts above
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour of day", 0, 23) # Changed to slider for better UX

# 1. Filter out rows without coordinates
map_data = data.dropna(subset=["tweet_coord"]).copy()

# 2. Convert string to list and split into lat/lon
def parse_coords(c):
    try:
        return ast.literal_eval(c)
    except:
        return None

map_data["tweet_coord"] = map_data["tweet_coord"].apply(parse_coords)
map_data = map_data.dropna(subset=["tweet_coord"])

# Note: In this dataset, index 0 is Latitude, index 1 is Longitude
map_data["lat"] = map_data["tweet_coord"].apply(lambda x: x[0])
map_data["lon"] = map_data["tweet_coord"].apply(lambda x: x[1])

# 3. Filter by the selected hour
modified_data = map_data[map_data['tweet_created'].dt.hour == hour]

# 4. Display Map
if not st.sidebar.checkbox("Close Map", True, key='map_check'):
    st.markdown("### Tweets location based on the hour of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)
st.sidebar.subheader("Breakdown of Airline Tweets by Sentiment")
choice = st.sidebar.multiselect('Pick Airlines', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='0') 
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment', facet_col='airline_sentiment', labels={'airline_sentiment':'Tweets'}, height=600, width=800)
    st.plotly_chart(fig_choice)                                
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close Word Cloud", True, key='word_cloud_check'):
    st.markdown("### Word cloud for %s sentiment" % (word_sentiment))
    df_word = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df_word['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.axis('off')
    st.pyplot(plt)    