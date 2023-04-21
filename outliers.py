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

    def get_iqr(self, column_names=None):
        try:
            if column_names is None:
                return pd.Series(self.iqr)
            else:
                if not set(column_names).issubset(set(self.numeric_columns)):
                    return f"Error: Invalid column names provided. Please provide column names that are numerical and exist in the dataset: {self.numeric_columns.tolist()}"

                iqr_dict={}
                for key, value in self.iqr.items():
                    if key in column_names:
                        iqr_dict[key]=value
                return pd.Series(iqr_dict)

        except Exception as e:
            return f"Error occurred in get_iqr(): {e}"

    def get_limits(self, column_names=None, decimal=4):
        try:
            if column_names is None:
                column_names = self.numeric_columns
            else:
                invalid_columns = set(column_names) - set(self.numeric_columns)
                if invalid_columns:
                    return f"Invalid column name(s): {invalid_columns}. Numeric columns: {self.numeric_columns}"
            
            limits = {}
            for name in column_names:
                try:
                    limits[name] = [round(self.lower_limit_dict[name], decimal), round(self.upper_limit_dict[name], decimal)]
                except KeyError:
                    return f"Column name {name} does not exist or is not numeric"
                    
            return limits
        
        except Exception as e:
            return f"Error occurred in get_limits(): {e}"

    
    def get_outliers(self, column_name, styler=True):
        
        try:

            if column_name not in self.numeric_columns:
                return f"Column name '{column_name}' is either not numeric or does not exist in the dataset"

            try:
                filtered_df = self.data[(self.data[column_name] < self.lower_limit_dict[column_name]) |
                                        (self.data[column_name] > self.upper_limit_dict[column_name])]

                # Highlight filtered column in the resulting filtered_df
                filtered_df_style = filtered_df.style.applymap(lambda x: 'background-color: yellow', subset=pd.IndexSlice[:, [column_name]])

                if styler==True:
                    return filtered_df_style
                elif styler==False:
                    return filtered_df

            except Exception as e:
                return f"Error occurred in filter_outlier(): {e}"

        except Exception as e:
            return f"Error occurred in filter_outlier(): {e}"
            
            

    def get_outliers_count(self, column_names=None):
        try:
            if column_names is None:
                column_names = self.numeric_columns
            else:
                if not isinstance(column_names, list):
                    raise ValueError("column_names must be a list")
                for col in column_names:
                    if col not in self.numeric_columns:
                        raise ValueError(f"Column name {col} does not exist or is not numerical")

            outlier_counts = {}
            for name in column_names:
                filtered_df = self.get_outliers(name, styler=False)
                if isinstance(filtered_df, pd.DataFrame):
                    outlier_counts[name] = filtered_df.shape[0]

            if not outlier_counts:
                return "No outlier found for given columns"

            return pd.Series(outlier_counts)

        except Exception as e:
            return f"Error occurred in get_outliers_count(): {e}"
    
    def plot_outliers_count(self, column_names=None, figsize=(10, 8), threshold_percent=5):
        
        try:
            if column_names is None:
                column_names = self.numeric_columns
            
            elif not isinstance(column_names, list):
                    raise ValueError("column_names must be a list")
            for col in column_names:
                if col not in self.numeric_columns:
                    raise ValueError(f"Column name {col} does not exist or is not numerical")
            else:
                column_names = [col for col in column_names if col in self.numeric_columns]
            
            outlier_counts={}
            for name in column_names:
                filtered_df = self.get_outliers(name, styler=False)
                if isinstance(filtered_df, pd.DataFrame):
                    outlier_counts[name] = filtered_df.shape[0]

            if not outlier_counts:
                return "No outlier found for given columns"

            plt.figure(figsize=figsize)
            ax = sns.barplot(x=list(outlier_counts.keys()), y=list(outlier_counts.values()))
            plt.axhline(int(self.data.shape[0]*threshold_percent/100), linestyle='--', color='r')
            plt.title('Outlier counts'.title(), fontsize=15, fontweight='bold')
            plt.xticks(rotation=90)
            plt.ylim(top=max(outlier_counts.values())*1.1) # set y-axis limit to fit the text annotations

            # add text annotations above each bar
            for i, v in enumerate(outlier_counts.values()):
                ax.text(i, v+max(outlier_counts.values())*0.05, str(v), color='black', ha='center', fontsize=10, fontweight='bold')

            plt.show()

        except Exception as e:
            return f"Error occurred in plot_outlier_count(): {e}"
            

    def get_outlier_proportion(self, column_name_list=None, decimal=2):
        try:
            if not self.outlier_counts:
                return "No outlier found for given columns"
                
            total_rows = self.data.shape[0]
            outlier_proportions_dict = {key: round(value * 100 / total_rows, decimal) 
                                        for key, value in self.outlier_counts.items()}
            return pd.Series(outlier_proportions_dict)
            
        except Exception as e:
            return f"Error occurred in get_outlier_proportion(): {e}"
