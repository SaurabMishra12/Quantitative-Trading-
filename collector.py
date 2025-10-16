import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from GoogleNews import GoogleNews
import requests
from textblob import TextBlob
import warnings
import json

warnings.filterwarnings('ignore')
try:
    import snscrape.modules.twitter as sntwitter
    TWITTER_AVAILABLE = True
except Exception:
    TWITTER_AVAILABLE = False

company_name = input("Enter company name: ").strip()
date_str = input("Enter date (YYYY-MM-DD): ").strip()
target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

ticker = yf.Ticker(company_name)
stock_data = ticker.history(start=target_date - datetime.timedelta(days=5),
                            end=target_date + datetime.timedelta(days=1))

trends = pd.DataFrame()
try:
    pytrends = TrendReq(hl='en-US', tz=330)
    pytrends.build_payload([company_name], timeframe='today 1-m')
    trends = pytrends.interest_over_time()
except Exception:
    trends = pd.DataFrame()

news_df = pd.DataFrame()
try:
    googlenews = GoogleNews(lang='en', region='IN')
    googlenews.set_time_range(
        (target_date - datetime.timedelta(days=2)).strftime("%m/%d/%Y"),
        (target_date + datetime.timedelta(days=1)).strftime("%m/%d/%Y")
    )
    googlenews.search(company_name)
    news = googlenews.result(sort=True)
    news_df = pd.DataFrame(news)
except Exception:
    news_df = pd.DataFrame()

if not news_df.empty and 'title' in news_df.columns:
    news_df['sentiment'] = news_df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
else:
    news_df = pd.DataFrame(columns=['title', 'date', 'sentiment'])
    news_df['sentiment'] = []

tweets = []
if TWITTER_AVAILABLE:
    try:
        query = f"{company_name} since:{(target_date - datetime.timedelta(days=1)).date()} until:{(target_date + datetime.timedelta(days=1)).date()}"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i > 100:
                break
            tweets.append([tweet.date, tweet.content])
    except Exception:
        pass

tweet_df = pd.DataFrame(tweets, columns=['date', 'text']) if tweets else pd.DataFrame(columns=['date', 'text', 'sentiment'])
if not tweet_df.empty:
    tweet_df['sentiment'] = tweet_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
else:
    tweet_df['sentiment'] = []

weather = {}
try:
    API_KEY = "b315f055e36177870ff32fb98dbf5131"
    city = "Mumbai"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    weather = response.json()
except Exception:
    weather = {'main': {'temp': None}, 'weather': [{'description': 'N/A'}]}

all_data = []

base_info = {
    'company': company_name,
    'date': date_str,
}

if not stock_data.empty:
    last_stock = stock_data.iloc[-1]
    base_info['stock_close'] = last_stock['Close']
    base_info['stock_volume'] = last_stock['Volume']
    base_info['stock_high'] = last_stock['High']
    base_info['stock_low'] = last_stock['Low']
    base_info['stock_open'] = last_stock['Open']
else:
    base_info['stock_close'] = None
    base_info['stock_volume'] = None
    base_info['stock_high'] = None
    base_info['stock_low'] = None
    base_info['stock_open'] = None

if not trends.empty and company_name in trends.columns:
    base_info['search_trend'] = trends[company_name].iloc[-1]
else:
    base_info['search_trend'] = 0

if 'main' in weather and 'weather' in weather:
    base_info['weather_temp'] = weather['main']['temp']
    base_info['weather_description'] = weather['weather'][0]['description']
else:
    base_info['weather_temp'] = None
    base_info['weather_description'] = None

base_info['avg_news_sentiment'] = news_df['sentiment'].mean() if not news_df.empty else 0
base_info['avg_tweet_sentiment'] = tweet_df['sentiment'].mean() if not tweet_df.empty else 0
base_info['news_count'] = len(news_df)
base_info['tweet_count'] = len(tweet_df)

if not news_df.empty:
    for idx, row in news_df.iterrows():
        row_data = base_info.copy()
        row_data['data_type'] = 'news'
        row_data['news_title'] = row.get('title', '')
        row_data['news_date'] = row.get('date', '')
        row_data['news_link'] = row.get('link', '')
        row_data['news_sentiment'] = row.get('sentiment', 0)
        row_data['news_description'] = row.get('desc', '')
        row_data['tweet_text'] = ''
        row_data['tweet_date'] = ''
        row_data['tweet_sentiment'] = 0
        all_data.append(row_data)

if not tweet_df.empty:
    for idx, row in tweet_df.iterrows():
        row_data = base_info.copy()
        row_data['data_type'] = 'tweet'
        row_data['news_title'] = ''
        row_data['news_date'] = ''
        row_data['news_link'] = ''
        row_data['news_sentiment'] = 0
        row_data['news_description'] = ''
        row_data['tweet_text'] = row.get('text', '')
        row_data['tweet_date'] = str(row.get('date', ''))
        row_data['tweet_sentiment'] = row.get('sentiment', 0)
        all_data.append(row_data)

if not all_data:
    base_info['data_type'] = 'summary'
    base_info['news_title'] = ''
    base_info['news_date'] = ''
    base_info['news_link'] = ''
    base_info['news_sentiment'] = 0
    base_info['news_description'] = ''
    base_info['tweet_text'] = ''
    base_info['tweet_date'] = ''
    base_info['tweet_sentiment'] = 0
    all_data.append(base_info)

final_df = pd.DataFrame(all_data)

column_order = [
    'company', 'date', 'data_type',
    'stock_close', 'stock_open', 'stock_high', 'stock_low', 'stock_volume',
    'search_trend', 'weather_temp', 'weather_description',
    'avg_news_sentiment', 'avg_tweet_sentiment', 'news_count', 'tweet_count',
    'news_title', 'news_date', 'news_link', 'news_sentiment', 'news_description',
    'tweet_text', 'tweet_date', 'tweet_sentiment'
]

column_order = [col for col in column_order if col in final_df.columns]
final_df = final_df[column_order]

output_filename = f"{company_name}_{date_str}_complete_data.csv"
final_df.to_csv(output_filename, index=False)

def create_ml_features(stock_data, news_df, tweet_df, trends, weather, company_name, target_date):
    ml_features = {}
    
    ml_features['year'] = target_date.year
    ml_features['month'] = target_date.month
    ml_features['day'] = target_date.day
    ml_features['day_of_week'] = target_date.weekday()
    ml_features['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
    ml_features['quarter'] = (target_date.month - 1) // 3 + 1
    ml_features['day_of_year'] = target_date.timetuple().tm_yday
    
    if not stock_data.empty:
        last_day = stock_data.iloc[-1]
        ml_features['stock_close'] = float(last_day['Close'])
        ml_features['stock_open'] = float(last_day['Open'])
        ml_features['stock_high'] = float(last_day['High'])
        ml_features['stock_low'] = float(last_day['Low'])
        ml_features['stock_volume'] = float(last_day['Volume'])
        
        ml_features['price_range'] = ml_features['stock_high'] - ml_features['stock_low']
        ml_features['price_range_pct'] = (ml_features['price_range'] / ml_features['stock_low']) * 100
        ml_features['open_close_diff'] = ml_features['stock_close'] - ml_features['stock_open']
        ml_features['open_close_diff_pct'] = (ml_features['open_close_diff'] / ml_features['stock_open']) * 100
        
        if len(stock_data) > 1:
            prev_day = stock_data.iloc[-2]
            ml_features['prev_close'] = float(prev_day['Close'])
            ml_features['price_change'] = ml_features['stock_close'] - ml_features['prev_close']
            ml_features['price_change_pct'] = (ml_features['price_change'] / ml_features['prev_close']) * 100
            ml_features['volume_change'] = ml_features['stock_volume'] - float(prev_day['Volume'])
            ml_features['volume_change_pct'] = (ml_features['volume_change'] / float(prev_day['Volume'])) * 100 if float(prev_day['Volume']) > 0 else 0
            
            if len(stock_data) >= 3:
                ml_features['ma_3'] = float(stock_data['Close'].tail(3).mean())
                ml_features['price_to_ma3'] = (ml_features['stock_close'] / ml_features['ma_3'] - 1) * 100
            
            if len(stock_data) >= 5:
                ml_features['ma_5'] = float(stock_data['Close'].tail(5).mean())
                ml_features['price_to_ma5'] = (ml_features['stock_close'] / ml_features['ma_5'] - 1) * 100
                ml_features['volatility_5d'] = float(stock_data['Close'].tail(5).std())
                ml_features['volume_ma_5'] = float(stock_data['Volume'].tail(5).mean())
                ml_features['volume_to_ma5'] = (ml_features['stock_volume'] / ml_features['volume_ma_5'] - 1) * 100
        else:
            ml_features['prev_close'] = ml_features['stock_close']
            ml_features['price_change'] = 0.0
            ml_features['price_change_pct'] = 0.0
            ml_features['volume_change'] = 0.0
            ml_features['volume_change_pct'] = 0.0
    else:
        stock_features = ['stock_close', 'stock_open', 'stock_high', 'stock_low', 'stock_volume',
                         'price_range', 'price_range_pct', 'open_close_diff', 'open_close_diff_pct',
                         'prev_close', 'price_change', 'price_change_pct', 'volume_change', 'volume_change_pct']
        for feat in stock_features:
            ml_features[feat] = 0.0
    
    if not news_df.empty and 'sentiment' in news_df.columns:
        sentiments = news_df['sentiment'].dropna()
        if len(sentiments) > 0:
            ml_features['news_sentiment_mean'] = float(sentiments.mean())
            ml_features['news_sentiment_std'] = float(sentiments.std()) if len(sentiments) > 1 else 0.0
            ml_features['news_sentiment_min'] = float(sentiments.min())
            ml_features['news_sentiment_max'] = float(sentiments.max())
            ml_features['news_sentiment_range'] = ml_features['news_sentiment_max'] - ml_features['news_sentiment_min']
            ml_features['news_positive_count'] = int((sentiments > 0.1).sum())
            ml_features['news_negative_count'] = int((sentiments < -0.1).sum())
            ml_features['news_neutral_count'] = int(((sentiments >= -0.1) & (sentiments <= 0.1)).sum())
            ml_features['news_count'] = len(news_df)
            ml_features['news_positive_ratio'] = ml_features['news_positive_count'] / ml_features['news_count']
            ml_features['news_negative_ratio'] = ml_features['news_negative_count'] / ml_features['news_count']
        else:
            ml_features.update({k: 0.0 for k in ['news_sentiment_mean', 'news_sentiment_std', 'news_sentiment_min',
                                                   'news_sentiment_max', 'news_sentiment_range', 'news_count',
                                                   'news_positive_ratio', 'news_negative_ratio']})
            ml_features.update({k: 0 for k in ['news_positive_count', 'news_negative_count', 'news_neutral_count']})
    else:
        ml_features.update({k: 0.0 for k in ['news_sentiment_mean', 'news_sentiment_std', 'news_sentiment_min',
                                               'news_sentiment_max', 'news_sentiment_range', 'news_count',
                                               'news_positive_ratio', 'news_negative_ratio']})
        ml_features.update({k: 0 for k in ['news_positive_count', 'news_negative_count', 'news_neutral_count']})
    
    if not tweet_df.empty and 'sentiment' in tweet_df.columns:
        tweet_sentiments = tweet_df['sentiment'].dropna()
        if len(tweet_sentiments) > 0:
            ml_features['tweet_sentiment_mean'] = float(tweet_sentiments.mean())
            ml_features['tweet_sentiment_std'] = float(tweet_sentiments.std()) if len(tweet_sentiments) > 1 else 0.0
            ml_features['tweet_count'] = len(tweet_df)
            ml_features['tweet_positive_count'] = int((tweet_sentiments > 0.1).sum())
            ml_features['tweet_negative_count'] = int((tweet_sentiments < -0.1).sum())
        else:
            ml_features.update({k: 0.0 for k in ['tweet_sentiment_mean', 'tweet_sentiment_std', 'tweet_count']})
            ml_features.update({k: 0 for k in ['tweet_positive_count', 'tweet_negative_count']})
    else:
        ml_features.update({k: 0.0 for k in ['tweet_sentiment_mean', 'tweet_sentiment_std', 'tweet_count']})
        ml_features.update({k: 0 for k in ['tweet_positive_count', 'tweet_negative_count']})
    
    ml_features['combined_sentiment'] = (ml_features['news_sentiment_mean'] * 0.6 + 
                                         ml_features['tweet_sentiment_mean'] * 0.4)
    
    if not trends.empty and company_name in trends.columns:
        trend_values = trends[company_name].tail(7)
        ml_features['search_trend_current'] = float(trend_values.iloc[-1])
        
        if len(trend_values) > 1:
            ml_features['search_trend_prev'] = float(trend_values.iloc[-2])
            ml_features['search_trend_change'] = ml_features['search_trend_current'] - ml_features['search_trend_prev']
            ml_features['search_trend_mean_7d'] = float(trend_values.mean())
            ml_features['search_trend_std_7d'] = float(trend_values.std())
            ml_features['search_trend_max_7d'] = float(trend_values.max())
            ml_features['search_trend_min_7d'] = float(trend_values.min())
        else:
            ml_features['search_trend_prev'] = ml_features['search_trend_current']
            ml_features['search_trend_change'] = 0.0
            ml_features['search_trend_mean_7d'] = ml_features['search_trend_current']
            ml_features['search_trend_std_7d'] = 0.0
            ml_features['search_trend_max_7d'] = ml_features['search_trend_current']
            ml_features['search_trend_min_7d'] = ml_features['search_trend_current']
    else:
        trend_features = ['search_trend_current', 'search_trend_prev', 'search_trend_change',
                         'search_trend_mean_7d', 'search_trend_std_7d', 'search_trend_max_7d', 'search_trend_min_7d']
        for feat in trend_features:
            ml_features[feat] = 0.0
    
    if 'main' in weather and 'temp' in weather['main']:
        ml_features['weather_temp'] = float(weather['main']['temp'])
        ml_features['weather_feels_like'] = float(weather['main'].get('feels_like', weather['main']['temp']))
        ml_features['weather_humidity'] = float(weather['main'].get('humidity', 0))
        ml_features['weather_pressure'] = float(weather['main'].get('pressure', 0))
        
        if 'weather' in weather and len(weather['weather']) > 0:
            weather_main = weather['weather'][0].get('main', 'Clear').lower()
            ml_features['weather_is_clear'] = 1 if 'clear' in weather_main else 0
            ml_features['weather_is_cloudy'] = 1 if 'cloud' in weather_main else 0
            ml_features['weather_is_rainy'] = 1 if 'rain' in weather_main else 0
            ml_features['weather_is_stormy'] = 1 if any(x in weather_main for x in ['storm', 'thunder']) else 0
        else:
            for feat in ['weather_is_clear', 'weather_is_cloudy', 'weather_is_rainy', 'weather_is_stormy']:
                ml_features[feat] = 0
    else:
        weather_features = ['weather_temp', 'weather_feels_like', 'weather_humidity', 'weather_pressure']
        for feat in weather_features:
            ml_features[feat] = 0.0
        for feat in ['weather_is_clear', 'weather_is_cloudy', 'weather_is_rainy', 'weather_is_stormy']:
            ml_features[feat] = 0
    
    ml_features['sentiment_volume_interaction'] = ml_features['combined_sentiment'] * ml_features.get('stock_volume', 0) / 1e6
    ml_features['trend_sentiment_interaction'] = ml_features['search_trend_current'] * ml_features['combined_sentiment']
    ml_features['news_volume_score'] = ml_features['news_count'] * (1 + abs(ml_features['news_sentiment_mean']))
    
    ml_features['company'] = company_name
    ml_features['date'] = target_date.strftime('%Y-%m-%d')
    ml_features['timestamp'] = target_date.timestamp()
    
    return ml_features

ml_data = create_ml_features(stock_data, news_df, tweet_df, trends, weather, company_name, target_date)
ml_df = pd.DataFrame([ml_data])

metadata_cols = ['company', 'date', 'timestamp']
feature_cols = [col for col in ml_df.columns if col not in metadata_cols]
ml_df = ml_df[metadata_cols + sorted(feature_cols)]

ml_output_filename = f"{company_name}_{date_str}_ml_features.csv"
ml_df.to_csv(ml_output_filename, index=False)

feature_description = {
    'file': ml_output_filename,
    'total_features': len(feature_cols),
    'feature_list': feature_cols,
    'numeric_features': [col for col in feature_cols if ml_df[col].dtype in ['float64', 'int64']],
    'categorical_features': [],
    'target_suggestions': ['price_change_pct', 'stock_close', 'open_close_diff_pct'],
    'ready_for': ['sklearn', 'tensorflow', 'pytorch', 'xgboost', 'lightgbm'],
    'preprocessing_done': {
        'missing_values': 'Filled with 0',
        'encoding': 'One-hot for weather conditions',
        'scaling': 'Not applied (apply StandardScaler or MinMaxScaler as needed)',
        'feature_engineering': 'Complete'
    }
}

feature_desc_filename = f"{company_name}_{date_str}_ml_features_description.json"
with open(feature_desc_filename, 'w') as f:
    json.dump(feature_description, f, indent=2)
