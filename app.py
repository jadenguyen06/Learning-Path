import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from app_store_scraper import AppStore
import re
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def parse_app_store_url(url): 
    '''Check if input URL from users is valid or not'''
    pattern = r"https://apps\.apple\.com/([a-z]{2})/app/([a-zA-Z0-9-]+)/id(\d+)" # Regular expression pattern for valid Apple App Store URLs
    match = re.match(pattern, url)
    
    if match:
        country, app_name, app_id = match.groups()
        return country, app_name, app_id
    else:
        return None  

class ReviewScraper:
    '''Scrape reviews from App Store and save to dataframe'''
    def __init__(self, country, app_name, app_id):
        self.country = country
        self.app_name = app_name
        self.app_id = app_id
        self.reviews = []

    @staticmethod
    @st.cache_data()
    def scraping(country, app_name, app_id, num):
        try:
            app = AppStore(country=country, app_name=app_name, app_id=app_id)
            app.review(how_many=num)
            return app.reviews
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []

    def saving_to_csv(self, reviews):
        if not reviews:
            st.warning("No reviews fetched.")
            return
       
        # Convert data to a DataFrame
        appdf = pd.DataFrame(np.array(reviews), columns=["review"])

        # Split the review column into separate columns
        appdf2 = appdf.join(pd.DataFrame(appdf.pop("review").tolist()))
        appdf2.head()

        # Save to CSV file
        filename = f"{self.app_name}-app-reviews.csv"
        appdf2.to_csv(filename)

class AppData:
    '''Cach the data for subsequent calls'''
    @staticmethod
    @st.cache_data()
    def load_reviews_csv(filename):
        return pd.read_csv(filename)
        
class EDA: 
    '''Conduct Exploratory Data Analysis'''
    def __init__(self, data):
        self.data = data

    def plot_rating_distribution(self):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.countplot(data=self.data, x='rating', ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    def plot_review_length_distribution(self):
        self.data['Review Length'] = self.data['review'].apply(len)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.histplot(self.data['Review Length'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Review Lengths')
        ax.set_xlabel('Length of Review')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    def plot_word_count_distribution(self):
        self.data['Word Count'] = self.data['review'].apply(lambda x: len(x.split()))
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.histplot(self.data['Word Count'], bins=50, kde=True)
        ax.set_title('Distribution of Word Counts')
        ax.set_xlabel('Number of Words in Review')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    def plot_date_distribution(self):
        # Convert 'date' column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        # Extract month and year from date column
        self.data['Year_Month'] = self.data['date'].dt.to_period('M')
        # Sort the data by date
        self.data = self.data.sort_values(by='Year_Month')
        # Plot distribution of dates
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.countplot(data=self.data, x='Year_Month', ax=ax)
        ax.set_title('Distribution of Dates')
        ax.set_xlabel('Date (Year-Month)')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

class SentimentAnalyzer:
    '''Conduct sentimemt analysis from the review'''
    def __init__(self, data):
        self.data = data

    def textblob_sentiment_analysis(self, review):
        sentiment = TextBlob(review).sentiment
        if sentiment.polarity > 0.1:
            return 'Positive'
        elif sentiment.polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_sentiments(self):
        # Applying TextBlob sentiment analysis to the reviews
        self.data['Sentiment'] = self.data['review'].apply(self.textblob_sentiment_analysis)

        # Analyzing the distribution of sentiments
        sentiment_distribution = self.data['Sentiment'].value_counts()

        # Plotting the distribution of sentiments
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values, ax=ax)
        ax.set_title('Distribution of Sentiments')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Plotting the distribution of sentiments and ratings
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=self.data, x='rating', hue='Sentiment', ax=ax)
        ax.set_title('Sentiment Distribution Across Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.legend(title='Sentiment')
        st.pyplot(fig)

class Insights:
    '''Function to generate wordcloud and topic modeling'''
    def __init__(self, data):
        self.data = data
        # self.stopwords = set()

    def generate_word_cloud(self, sentiment):
        text = ' '.join(review for review in self.data[self.data['Sentiment'] == sentiment]['review'])
        wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Word Cloud for {sentiment} Reviews')
        ax.axis('off')
        st.pyplot(fig) 

    def word_clouds_by_sentiment(self):
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            self.generate_word_cloud(sentiment)

    def topic_modeling(self, num_topics=5):

        # Initialize TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

        # Fit and transform the cleaned text data
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['review'])

        # Initialize Latent Dirichlet Allocation model
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

        # Fit the LDA model to the TF-IDF matrix
        lda_model.fit(tfidf_matrix)

        # Display the topics and associated top words
        topics = []
        for index, topic in enumerate(lda_model.components_):
            top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topics.append({"Topic": f"Topic {index + 1}", "Keywords": ', '.join(top_words)})
        
        st.write(pd.DataFrame(topics))


def main():
    st.set_page_config(page_title="Review Analysis Tool", page_icon="🦜", layout='wide')

    st.title("Apple App Store Review Analysis")
    st.markdown("This is a tool to help you get insights from the content of app reviews on the Apple App Store!")

    # Initialize session state
    if 'url_input' not in st.session_state:
        st.session_state.url_input = ""
    if 'num_reviews_input' not in st.session_state:
        st.session_state.num_reviews_input = None
    
    # User input for App Store URL
    url = st.text_input("Enter the App Store URL:", key="url_input")
    parsed_url = parse_app_store_url(url)
    if url and parsed_url is None:
        st.warning("Invalid URL") 
    else:
        num_reviews = st.number_input("Enter the number of reviews to fetch (maximum 2000):", min_value=1, max_value=2000, value=None, key="num_reviews_input", step=1)
    
    col1, col2 = st.columns([1, 1])


    with col1:

        if st.button("Submit"):
            if parsed_url is not None and num_reviews is not None and 1 <= num_reviews <= 2000:
                # Parse the URL
                country, app_name, app_id = parsed_url
        
                # Fetch reviews
                reviews = ReviewScraper.scraping(country, app_name, app_id, num_reviews)

                # Save reviews to CSV
                scraper = ReviewScraper(country, app_name, app_id)
                scraper.saving_to_csv(reviews)
                st.success(f"Reviews saved to csv file")

                # Save parsed_url and num_reviews to session state
                st.session_state.parsed_url = parsed_url
                st.session_state.num_reviews = num_reviews

            else: 
                st.warning("Input valid URL and number of reviews")
    
    with col2:
        if st.button("Refresh"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Load reviews DataFrame only if parsed_url and num_reviews are present in session state
    if 'parsed_url' in st.session_state and 'num_reviews' in st.session_state:
        parsed_url = st.session_state.parsed_url
        num_reviews = st.session_state.num_reviews
        
        # Parse the URL
        country, app_name, app_id = parsed_url

        # Display data from CSV
        filename = f"{app_name}-app-reviews.csv"
        df = AppData.load_reviews_csv(filename)

        st.sidebar.header('Functions')
        st.sidebar.caption('Click into checkboxes below')
        if st.sidebar.checkbox("Show Data"):
            st.header('Raw Data')
            st.write(df)
            st.download_button(label="Download CSV", data=df.to_csv().encode("utf-8"),
                               file_name=f"{app_name}-app-reviews.csv", mime="text/csv")
        
        if st.sidebar.checkbox("Exploratory Data Analysis"):
            st.header("Data Visualization")
            visualizer = EDA(df)
            visualizer.plot_date_distribution()
            visualizer.plot_rating_distribution()
            visualizer.plot_review_length_distribution()
            visualizer.plot_word_count_distribution()

        if st.sidebar.checkbox("Sentiment Analysis and WordCloud"):
            st.header("Sentiment Analysis Results")
            sentiment_analyzer = SentimentAnalyzer(df)
            df_with_sentiment = sentiment_analyzer.analyze_sentiments()
            
            if df_with_sentiment is not None:
                # Append sentiment to df2
                df['Sentiment'] = df_with_sentiment['Sentiment']
        
            st.header("Word Clouds Generator")
            wordcloud_generator = Insights(df)
            
            if df is not None:
                wordcloud_generator.word_clouds_by_sentiment()
                st.sidebar.header("Input")
                if st.sidebar.checkbox("Run Topic Modeling"):
                        st.header("Topic Modeling")
                        st.caption('The table below shows keywords in each topic, which helps you infer the theme or subject matter that the topic represents.')
                        num_topics = st.sidebar.slider("Select the number of topics for topic modeling:", min_value=0, max_value=10, value=5,step=1)
                        wordcloud_generator.topic_modeling(num_topics)
                               
main()

