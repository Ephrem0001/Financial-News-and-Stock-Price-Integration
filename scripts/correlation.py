import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

class Correlation:
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def read_csv_with_date(self, filepath, date_col, parse_dates=True):
        # Read the CSV file and optionally parse the date column(s)
        df = pd.read_csv(filepath, parse_dates=[date_col] if parse_dates else None)
        return df

    def check_nulls_and_info(self, df, df_name='DataFrame'):
        # Calculate the total number of missing values
        total_nulls = df.isnull().sum().sum()
        print(f"Total null values in {df_name}: {total_nulls}\n")
        
        # Print the info summary of the DataFrame
        print(f"{df_name} Info:")
        print(df.info())
        print("\n" + "="*50 + "\n")

    def convert_to_datetime(self, df, date_col='date', date_format='mixed', utc=True):
        # Convert the specified date column to datetime with the specified format and UTC setting
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, utc=utc)
        return df

    def format_date_column(self, df, date_col='date', date_format='%Y-%m-%d'):
        # Convert the date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Format the date column as a string in the specified format
        df[date_col] = df[date_col].dt.strftime(date_format)
        
        return df

    def analyze_sentiment(self, df, text_col='headline', sentiment_col='sentiment', sentiment_category_col='sentiment_category'):
        # Create a copy of the DataFrame to avoid modifying the original
        sentiment_data = df.copy()
        
        # Initialize the sentiment intensity analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Apply the sentiment analysis
        sentiment_data[sentiment_col] = sentiment_data[text_col].apply(lambda x: sia.polarity_scores(text=x)['compound'])
        
        # Categorize the sentiment scores
        sentiment_data[sentiment_category_col] = pd.cut(
            sentiment_data[sentiment_col],
            bins=[-1, -0.0001, 0, 0.5, 1],
            labels=['Negative', 'Neutral', 'Positive', 'Very Positive']
        )
        return sentiment_data

    def calculate_daily_returns(self, df, column_name='Close', new_column_name='Daily Returns'):
        df[new_column_name] = df[column_name].pct_change()
        return df
    
    def aggregate_daily_sentiment(self, df, sentiment_col='sentiment'):
        # Group by the index (date) and compute the average sentiment score for each day
        daily_sentiment = df.groupby(df.index)[sentiment_col].mean().reset_index()
        
        # Rename columns for clarity
        daily_sentiment.columns = ['date', 'average_sentiment']
        
        return daily_sentiment

    def merge_dataframes(self, df1, df2):
        aligned_data = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
        return aligned_data

    def analyze_correlation_and_plot(self, df, sentiment_col='average_sentiment', returns_col='Daily Returns'):
        # Calculate the correlation between sentiment and daily returns
        correlation = df[sentiment_col].corr(df[returns_col])
        print("Correlation between sentiment and daily stock returns:", correlation)
        
        # Create a correlation matrix
        correlation_matrix = df[[sentiment_col, returns_col]].corr()
        plt.show()
        # Plot the heatmap of the correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Sentiment vs {returns_col} Matrix')
        plt.show()
        
        # Scatterplot to visualize the relationship between sentiment and daily returns
        sns.scatterplot(x=sentiment_col, y=returns_col, data=df)
        plt.title(f'Sentiment vs {returns_col}')
        plt.xlabel('Sentiment')
        plt.ylabel(f'{returns_col}')
        plt.show()
