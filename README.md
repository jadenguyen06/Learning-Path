# Apple App Store Review Analysis Tool – Data Scraping and Text Analysis
## Project Overview
> The objective of this project is to develop a robust Apple App Store review analysis tool that incorporates techniques such as exploratory data analysis, sentiment analysis, and topic modeling. This tool aims to assist app owners and product managers in gaining deeper insights into the content of app reviews from the Apple App Store. These insights will help them understand users' preferences and complaints, boost their star ratings, enhance user satisfaction, and ultimately drive higher downloads and revenue.

## Scope
The scope of this project entails providing comprehensive analysis for users based on a provided App Store URL. The program will scrape reviews from the Apple App Store and conduct various analyses to generate insights. These analyses include: 
- Exploratory Data Analysis (EDA)
- Sentiment Analysis
- Word Cloud
- Topic Modelling

![EDA](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/abad9001-5378-445f-919f-4442c02b38b4)

![EDA 2](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/2f9cf99f-b7e3-4855-9cd8-cacdf7b50f42)

![Sentiment](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/ba48f697-93b0-4d71-bfa2-81263b42065b)

![WordCloud](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/b6cc4e8a-1664-4d3a-9dfe-fc5b8807eb66)

![TopicModelling](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/93a20f0a-5565-4386-a89a-d8b011744e79)



## How to use
Start the app by using the following command:\
`streamlit run app.py`

Enter the App Store URL and number of review to fetch (max 2000 reviews/request) and click Submit button!

![Main feature](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/4da6cea8-ba15-4b23-bbff-dbd8d118e51f)

![Show Data](https://github.com/jadenguyen06/app_store_review_analysis_tool/assets/171535084/3add49c0-422a-421e-b9b5-97c16f3480e8)


The app will fetch reviews, save them to a CSV file, and perform exploratory data analysis (EDA), sentiment analysis, word cloud generation, and topic modeling. You can adjust the number of topics to better infer the themes or subject matters represented by the topics.

## Alert
A significant weakness of the program lies in the limitation imposed by Apple's lack of an API for accessing App Store data. Using Python library `app_store_scraper` to scrape reviews will face with the rate limits.

In my observations, 5,000 reviews/day/IP is an ideal number of fetching reviews within this program. When the limit point is reached, users are advised to try later or change the IP address to continue the usage of this program. Therefore, I set the maximum number of reviews that users can request is 2,000 to ensure the stability while using the program. Besides, when fetching app reviews, the library usually retrieves the most recent reviews first, it may not meet the need of customization in date or ratings.

Despite these challenges, the program remains effective for accessing the latest feedback and insights about an application, particularly for small to medium-sized datasets.

## References
- [Sentiment Analysis](https://thecleverprogrammer.com/2023/12/04/app-reviews-sentiment-analysis-using-python/)
- [Scrape App Store Reviews](https://www.freecodecamp.org/news/how-to-use-python-to-scrape-app-store-reviews/)


