import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class FinancialAnalyzer:
    def __init__(self, file_path=None):
        """
        Initialize with an optional file path to CSV data.
        
        :param file_path: Path to a CSV file containing historical stock data.
        """
        self.file_path = file_path
        self.data = None
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        :param file_path: Path to the CSV file.
        """
        self.data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        self.data.sort_index(inplace=True)

    def retrieve_stock_data(self):
        # Download stock data from Yahoo Finance
        if self.file_path:
            return self.data
        else:
            raise ValueError("No file path provided. Please load data using load_data method.")

    def calculate_moving_average(self, data, column, window_size):
        # Calculate Simple Moving Average (SMA)
        return ta.SMA(data[column], timeperiod=window_size)

    def calculate_technical_indicators(self):
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("No data available. Please load data using load_data method.")
        
        # Calculate various technical indicators
        self.data['SMA'] = self.calculate_moving_average(self.data, 'Close', 20)
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        self.data['EMA'] = ta.EMA(self.data['Close'], timeperiod=20)
        macd, macd_signal, _ = ta.MACD(self.data['Close'])
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        return self.data

    def plot_stock_data(self):
        # Ensure data is loaded and technical indicators are calculated
        if self.data is None or 'SMA' not in self.data:
            raise ValueError("Data or SMA not available. Please ensure data is loaded and indicators are calculated.")
        
        # Plot Stock Price and Moving Average
        fig = px.line(self.data, x=self.data.index, y=['Close', 'SMA'], title='Stock Price with Moving Average')
        fig.show()

    def plot_rsi(self):
        # Ensure data is loaded and RSI is calculated
        if self.data is None or 'RSI' not in self.data:
            raise ValueError("Data or RSI not available. Please ensure data is loaded and indicators are calculated.")
        
        # Plot Relative Strength Index (RSI)
        fig = px.line(self.data, x=self.data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.show()

    def plot_ema(self):
        # Ensure data is loaded and EMA is calculated
        if self.data is None or 'EMA' not in self.data:
            raise ValueError("Data or EMA not available. Please ensure data is loaded and indicators are calculated.")
        
        # Plot Stock Price and Exponential Moving Average
        fig = px.line(self.data, x=self.data.index, y=['Close', 'EMA'], title='Stock Price with Exponential Moving Average')
        fig.show()

    def plot_macd(self):
        # Ensure data is loaded and MACD is calculated
        if self.data is None or 'MACD' not in self.data:
            raise ValueError("Data or MACD not available. Please ensure data is loaded and indicators are calculated.")
        
        # Plot MACD and MACD Signal
        fig = px.line(self.data, x=self.data.index, y=['MACD', 'MACD_Signal'], title='Moving Average Convergence Divergence (MACD)')
        fig.show()


class FinancialAnalyzer2:
    def __init__(self, file_paths):
        """
        Initialize with a list of file paths to CSV data.
        
        :param file_paths: List of paths to CSV files containing historical stock data.
        """
        self.file_paths = file_paths
        self.data = self.load_data(file_paths)

    def load_data(self, file_paths):
        """
        Load and combine data from multiple CSV files.
        
        :param file_paths: List of paths to the CSV files.
        :return: Combined DataFrame with stock closing prices.
        """
        data_frames = []
        for file_path in file_paths:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            df = df[['Close']].rename(columns={'Close': file_path.split('/')[-1].split('.')[0]})
            data_frames.append(df)

        # Combine all DataFrames on the Date index
        combined_data = pd.concat(data_frames, axis=1)
        combined_data.sort_index(inplace=True)
        return combined_data

    def calculate_portfolio_weights(self):
        """
        Calculate portfolio weights using the historical data from the loaded CSV files.
        
        :return: Dictionary of ticker symbols and their corresponding portfolio weights.
        """
        data = self.data
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        weights = dict(zip(data.columns, weights.values()))
        return weights

    def calculate_portfolio_performance(self):
        """
        Calculate the portfolio performance using the historical data from the loaded CSV files.
        
        :return: Tuple containing the portfolio return, volatility, and Sharpe ratio.
        """
        data = self.data
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
        return portfolio_return, portfolio_volatility, sharpe_ratio

