import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Outliers:
    def __init__(self, limit_factor=1.5):
        self.limit_factor = limit_factor
        self.data = None
        self.numeric_columns = None
        self.lower_limit_dict = {}
        self.upper_limit_dict = {}
        self.iqr = {}
        self.outlier_counts = {}
        
    def fit(self, data):
        try:
            self.data = data 
            self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for col in self.numeric_columns:
                q1, q3 = self.data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.lower_limit_dict[col] = q1 - self.limit_factor * iqr
                self.upper_limit_dict[col] = q3 + self.limit_factor * iqr
                self.iqr[col] = iqr

            return 'Upper and lower limits identified successfully!'
        
        except Exception as e:
            return f"Error occurred in fit(): {e}"
    
    def filter_outlier(self, column_name):
        try:
            filtered_df = self.data[(self.data[column_name] < self.lower_limit_dict[column_name]) |
                                     (self.data[column_name] > self.upper_limit_dict[column_name])]
            return filtered_df
        
        except KeyError:
            return f"Column name {column_name} does not exist"
        except Exception as e:
            return f"Error occurred in filter_outlier(): {e}"
    
    def plot_outlier_count(self, column_name_list=None, figsize=(10,8)):
        try:
            if column_name_list is None:
                column_name_list = self.numeric_columns
            else:
                column_name_list = [col for col in column_name_list if col in self.numeric_columns]

            for name in column_name_list:
                filtered_df = self.filter_outlier(name)
                if isinstance(filtered_df, pd.DataFrame):
                    self.outlier_counts[name] = filtered_df.shape[0]

            if not self.outlier_counts:
                return "No outlier found for given columns"

            plt.figure(figsize=figsize)
            ax = sns.barplot(x=list(self.outlier_counts.keys()), y=list(self.outlier_counts.values()))
            plt.title('Outlier counts'.title(), fontsize=15, fontweight='bold')
            plt.xticks(rotation=90)
            plt.ylim(top=max(self.outlier_counts.values())*1.1) # set y-axis limit to fit the text annotations

            # add text annotations above each bar
            for i, v in enumerate(self.outlier_counts.values()):
                ax.text(i, v+max(self.outlier_counts.values())*0.05, str(v), color='black', ha='center', fontsize=10, fontweight='bold')

            plt.show()

        except Exception as e:
            return f"Error occurred in plot_outlier_count(): {e}"

    def get_outlier_proportion(self, column_name_list=None):
        try:
            if not self.outlier_counts:
                return "No outlier found for given columns"
                
            total_rows = self.data.shape[0]
            outlier_proportions_dict = {key: round(value * 100 / total_rows, 2) 
                                        for key, value in self.outlier_counts.items()}
            return outlier_proportions_dict
            
        except Exception as e:
            return f"Error occurred in get_outlier_proportion(): {e}"
