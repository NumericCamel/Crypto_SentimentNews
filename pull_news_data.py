## 0. IMPORT ALL DEPENDENCIES AND SETTING UP ENVIORNMENT
import requests
from datetime import datetime 
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dateutil import parser
import html
import re 
import praw 
from GoogleNews import GoogleNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os 

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
user_agent = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

'''
---------------------------------------------
----- DEFINING ALL NEWS PULL FUNCTIONS ------
---------------------------------------------
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
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
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
subreddits = ["solana", "ethereum", "bitcoin", "news", "CryptoCurrency", "crypto", "finance"]
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

# Make a 
total_raw_data['Total Text'] = total_raw_data.Title.fillna('') + '' + total_raw_data.Description.fillna('')
total_raw_data['Pre-Processed Text'] = total_raw_data['Total Text'].apply(preprocess_text)
total_raw_data = total_raw_data[['MainSource', 'Source', 'Publication Date', 'Pull_date', 'Pre-Processed Text']]


'''
DOING SENTIMENT ANALYSIS
'''

# STEP 1: INITALIZE VADER VARIABLE
analyzer = SentimentIntensityAnalyzer()

# STEP 2: APPLY VADER SENTIMENT ON PRE-PROCESSED TEXT
total_raw_data['Sentiment_Vader'] = total_raw_data['Pre-Processed Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

#STEP 3: APPLY TEXTBLOB ON PRE-PROCESSED TEXT
total_raw_data['Sentiment_TextBlob'] = total_raw_data['Pre-Processed Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# STEP 4: GET THE AVERAGE SENTIMENT VARIABLE
total_raw_data['Average_Sentiment'] = (total_raw_data['Sentiment_Vader']+total_raw_data['Sentiment_TextBlob'])/2
total_raw_data.to_csv(f'Data/Processed_Data/Newsfeed_Processed_{current_date}.csv', index=False)

# STEP 5: PRINT OUT FINDINGS
print(f"Vader Sentiment Average: {total_raw_data['Sentiment_Vader'].mean()}")
print(f"Vader TextBlob Average: {total_raw_data['Sentiment_TextBlob'].mean()}")
print(f"Total Average: {total_raw_data['Average_Sentiment'].mean()}")

# STEP 6: GROUP SENTIMENT BY DAY TO GET TOTAL AVERAGE SENTIMENT FROM DATA PULL
sent = total_raw_data[['Pull_date', 'Sentiment_Vader', 'Sentiment_TextBlob', 'Average_Sentiment']]
sent = sent.groupby(['Pull_date']).mean().reset_index(drop=False)

# STEP 7: LOAD HISTORIC SENTIMENT VALUES TO ADD TO DATAFRAME
sentiment = pd.read_csv('Data/Processed_Data/Sentiment_Values/Sentiment.csv')
sentiment = pd.concat([sentiment, sent], ignore_index=True)
sentiment['Pull_date'] = pd.to_datetime(sentiment['Pull_date'])
sentiment = sentiment.sort_values(by='Pull_date')
sentiment = sentiment[['Pull_date','Sentiment_Vader','Sentiment_TextBlob','Average_Sentiment']]
sentiment.to_csv('Data/Processed_Data/Sentiment_Values/Sentiment.csv')

# STEP 8: FIND PERCENTAGE DIFFERENCE FROM LAST PULL
last_two_scores = sentiment['Average_Sentiment'].iloc[-2:]
percentage_diff = last_two_scores.pct_change().iloc[-1] * 100
print(f"Percentage difference in sentiment score between the new day and the previous day: {percentage_diff:.2f}%")


'''
STARTING WORDCLOUD AND OUTPUTS
'''

total_raw_data = pd.read_csv('Data/Processed_Data/Newsfeed_Processed_09_10_2024_21.csv')
total_raw_data['Pre-Processed Text'] = total_raw_data['Pre-Processed Text'] .fillna('')
# STEP 1: DEFINE FUNCTIONS
def tf_idf(text, ngram_range=(2, 2), exclude_words = None):
    if exclude_words is None:
        exclude_words = []
    vectorizer = TfidfVectorizer(stop_words=exclude_words, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(text)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    term_importance = tfidf_df.sum().sort_values(ascending=False)
    return term_importance

def word_cloud(text, title, type, top=65, dpi=210):
    top_n_terms = text.head(top)
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(top_n_terms)
    fig = plt.figure(figsize=(10, 5), dpi = dpi)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, fontweight='bold', backgroundcolor='White', color='black', pad=20)
    fig.savefig(f'Data/Processed_Data/Graphic_Output/{type}WC{current_date}.png', transparent=True)
    


# STEP 2: MAKE FOR FULL DATAFRAME
FullText = total_raw_data['Pre-Processed Text']
FullText = tf_idf(FullText)
word_cloud(FullText, 'All Text Wordcloud', 'AllText')

PositiveSentiment = total_raw_data[total_raw_data["Average_Sentiment"] > 0.5]['Pre-Processed Text']
PositiveSentiment = tf_idf(PositiveSentiment)
word_cloud(PositiveSentiment, 'Positive News Articles Only', 'Positive')

NegativeSentiment = total_raw_data[total_raw_data["Average_Sentiment"] < -0.5]['Pre-Processed Text']
NegativeSentiment = tf_idf(NegativeSentiment)
word_cloud(NegativeSentiment, 'Negative News Articles Only', 'Negative')

print('HurayyY! News data pull is done!')
