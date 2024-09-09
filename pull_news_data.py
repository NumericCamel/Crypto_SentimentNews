## 0. IMPORT ALL DEPENDENCIES AND SETTING UP ENVIORNMENT
import requests
from datetime import datetime 
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dateutil import parser
import html
import re 
from secret import secrets
import praw 
from GoogleNews import GoogleNews

## ENVIORNMENT DATE VARIABLES
current_date = datetime.now().strftime("%m_%d_%Y_%H")
current_date_sys = datetime.now()

### RSS NEWS SOURCES:
rss_urls = [
    "https://cointelegraph.com/rss",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://bitcoinist.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://cryptopotato.com/feed/",
    "https://99bitcoins.com/feed/",
    "https://cryptobriefing.com/feed/",
    "https://www.coinbackyard.com/feed/",
    "https://stratus.io/blog/feed/"
]

# INITIALIZE THE DATAFRAME
main_frame = pd.DataFrame()

'''
----- DEFINING ALL NEWS PULL FUNCTIONS ------
'''

## 1. RSS FEED FUNCTION
def rss_to_dataframe(rss_url):
    # Fetch the RSS feed
    response = requests.get(rss_url)
    xml_content = response.content

    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Extract relevant data
    data = []
    for item in root.findall(".//item"):
        title = item.find("title").text if item.find("title") is not None else None
        link = item.find("link").text if item.find("link") is not None else None
        pub_date = item.find("pubDate").text if item.find("pubDate") is not None else None
        # Convert the publication date to datetime
        pub_date = item.find("pubDate").text if item.find("pubDate") is not None else None
        if pub_date:
            pub_date = parser.parse(pub_date)  # Convert to datetime object
        
        # Clean the description by removing HTML tags
        description = item.find("description").text if item.find("description") is not None else None
        if description:
            soup = BeautifulSoup(description, "html.parser")
            description = soup.get_text()
            description = description.replace('\n', ' ')
        
        data.append({
            "MainSource": 'RSS',
            "Source": rss_url,
            "Title": title,
            "Publication Date": pub_date,
            "Description": description,
            "Pull_date" : current_date_sys
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df


## 2. GOOGLENEWS FUNCTION
def fetch_google_news_data(keywords, pages=5):
    googlenews = GoogleNews(lang='en', region='US')
    news_data = []

    for keyword in keywords:
        googlenews.search(keyword)
        
        for page in range(1, pages + 1):
            googlenews.getpage(page)
            results = googlenews.result()

            for entry in results:
                news_item = {
                    "MainSource":'GoogleNews',
                    "Source": keyword,
                    "Title": entry.get("title"),
                    "Publication Date": entry.get("date"),
                    "Description": entry.get("desc"),
                    "Pull_date" : current_date_sys
                }
                news_data.append(news_item)
        
        googlenews.clear()

    return pd.DataFrame(news_data)


## 3. REDDIT FUNCTION
def fetch_top_reddit_posts(client_id, client_secret, user_agent, subreddits, limit=100):
    # Authenticate with Reddit using OAuth
    reddit = praw.Reddit(
        client_id=secrets['client_id'],
        client_secret=secrets['client_secret'],
        user_agent='CamelQuant'
    )
    
    # List to hold the collected posts
    posts_data = {
        "MainSource":"Reddit",
        "Source": [],
        "Title": [],
        "Publication Date": [],
        "Description":[],
        "Pull_date" : current_date_sys
    }

    # Iterate over each subreddit
    for subreddit in subreddits:
        # Access the subreddit
        subreddit_instance = reddit.subreddit(subreddit)
        
        # Fetch the top posts
        top_posts = subreddit_instance.hot(limit=limit)
        
        # Process each post
        for post in top_posts:
            posts_data["Source"].append(subreddit)
            posts_data["Title"].append(post.title)
            posts_data["Publication Date"].append(datetime.fromtimestamp(post.created_utc))
            posts_data["Description"].append(post.selftext)
    
    # Convert the dictionary of lists to a pandas DataFrame
    posts_df = pd.DataFrame(posts_data)
    
    return posts_df

'''
----- RUNNING THE FUNCTIONS -----
'''

## 1. RSS DATA
for rss_url in rss_urls:
    df = rss_to_dataframe(rss_url)
    main_frame = pd.concat([main_frame, df], ignore_index=True)

print('Pulling RSS News Successful')
print(f'A total of {len(rss_urls)} URLs was pulled. A total of {len(main_frame)} different articles pulled.')


## 2. GOOGLE NEWS DATA
keywords = ["cryptocurrency", "bitcoin", "ethereum", "blockchain", "solana","Blockchain News", "Crypto","New Crypto Projects","Crypto Bullrun","Solana projects",'Ethereum Projects']
print('Starting Google News Data Pull...')
news_df = fetch_google_news_data(keywords)
print(f'Google News Data Pull Successful, a total of {len(news_df)} articles were pulled.')


## 3. REDDIT DATA
client_id=secrets['client_id'],
client_secret=secrets['client_secret'],
user_agent='CamelQuant'
subreddits = ["solana", "ethereum", "bitcoin", "news", "CryptoCurrency", "crypto", "finance","Satoshistreetbets","Ripple"]

print('Starting Reddit Pull...')
posts = fetch_top_reddit_posts(client_id, client_secret, user_agent, subreddits)
print(f'A total of {len(posts)} posts were pulled')

# 4. STACKING THE DATAFRAMES
total_raw_data = pd.concat([main_frame, news_df, posts], ignore_index=True)
print('All News Data Pull Successful.')
print(f'A total of {len(total_raw_data)} news/posts instances were pulled.')
print(total_raw_data)

### Save raw data
total_raw_data.to_csv(f'Data/Raw_Data/Newsfeed_{current_date}.csv', index=False)

'''
Starting Text Pre Processing 
'''

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

### STARTING TEXT PRE-PROCESSING
print('Starting Text Pre-Processing...')
total_raw_data['Total Text'] = total_raw_data.Title.fillna('') + '' + total_raw_data.Description.fillna('')
total_raw_data['Pre-Processed Text'] = total_raw_data['Total Text'].apply(preprocess_text)
total_raw_data = total_raw_data[['MainSource', 'Source', 'Publication Date', 'Pull_date', 'Pre-Processed Text']]


'''
DOING SENTIMENT ANALYSIS
'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis
total_raw_data['Sentiment_Vader'] = total_raw_data['Pre-Processed Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
total_raw_data['Sentiment_TextBlob'] = total_raw_data['Pre-Processed Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
total_raw_data['Average_Sentiment'] = (total_raw_data['Sentiment_Vader']+total_raw_data['Sentiment_TextBlob'])/2

print(f"Vader Sentiment Average: {total_raw_data['Sentiment_Vader'].mean()}")
print(f"Vader TextBlob Average: {total_raw_data['Sentiment_TextBlob'].mean()}")
print(f"Total Average: {total_raw_data['Average_Sentiment'].mean()}")

total_raw_data.to_csv(f'Data/Processed_Data/Newsfeed_Processed_{current_date}.csv', index=False)

print('HurayyY! News data pull is done!')
