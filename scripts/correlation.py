import pandas as pd
import numpy as np


# class Correlation:
#     def __init__(self, file_path=None):
#         """
#         Initialize with an optional file path to CSV data.
        
#         :param file_path: Path to a CSV file containing historical stock data.
#         """
#         self.file_path = file_path
#         self.data = None
#         if file_path:
#             self.load_data(file_path)

#     def load_data(self, file_path):
#         """
#         Load data from a CSV file.
        
#         :param file_path: Path to the CSV file.
#         """
#         self.data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
#         self.data.sort_index(inplace=True)


class Correlation:
    def __init__(self, file_paths):
        self.data = None
        if file_paths:
            self.load_data(file_paths)

    def load_data(self, file_paths):
        data_frames = []
        for file_path in file_paths:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            data_frames.append(df)
        self.data = pd.concat(data_frames)
        self.data.sort_index(inplace=True)

    def align_by_date(df1, df2):
    # Ensure both DataFrames have Date as index
    if df1.index.name != 'Date':
        df1.set_index('Date', inplace=True)
    if df2.index.name != 'Date':
        df2.set_index('Date', inplace=True)
    
    # Align datasets by merging on the Date index
    aligned_data = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    
    return aligned_data

