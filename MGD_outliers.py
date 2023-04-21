import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Outliers:
    def __init__(self, limit_factor=1.5):
        self.limit_factor = limit_factor
        self.__data = None
        self.numeric_columns = None
        self.__lower_limit_dict = {}
        self.__upper_limit_dict = {}
        self.__iqr = {}
        
    def fit(self, data):
        try:
            self.__data = data 
            self.numeric_columns = self.__data.select_dtypes(include=[np.number]).columns
            
            for col in self.numeric_columns:
                q1, q3 = self.__data[col].quantile([0.25, 0.75])
                val_iqr = q3 - q1
                self.__lower_limit_dict[col] = q1 - self.limit_factor * val_iqr
                self.__upper_limit_dict[col] = q3 + self.limit_factor * val_iqr
                self.__iqr[col] = val_iqr

            return 'Upper and lower limits identified successfully!'
        
        except Exception as e:
            return f"Error occurred in fit(): {e}"

    def get_iqr(self, column_names=None):
        
        if column_names is None:
            column_names = self.numeric_columns

        elif not isinstance(column_names, list):
                raise ValueError("column_names must be a list")
        for col in column_names:
            if col not in self.numeric_columns:
                raise ValueError(f"Column name {col} does not exist or is not numerical")
        else:
            column_names = [col for col in column_names if col in self.numeric_columns]
        
        iqr_dict={}
        for key, value in self.__iqr.items():
            if key in column_names:
                iqr_dict[key]=value
        return pd.Series(iqr_dict)


    def get_limits(self, column_names=None, decimal=4):
        
        if column_names is None:
            column_names = self.numeric_columns

        elif not isinstance(column_names, list):
                raise ValueError("column_names must be a list")
        for col in column_names:
            if col not in self.numeric_columns:
                raise ValueError(f"Column name {col} does not exist or is not numerical")
        else:
            column_names = [col for col in column_names if col in self.numeric_columns]
            
        limits = {}
        for name in column_names:
            try:
                limits[name] = [round(self.__lower_limit_dict[name], decimal), round(self.__upper_limit_dict[name], decimal)]
            except KeyError:
                return f"Column name {name} does not exist or is not numeric"

        return limits

    
    def get_outliers(self, column_name, styler=True):
        
        if not isinstance(column_name, str):
                raise ValueError("column_name must be a single string")
        
        if column_name not in self.numeric_columns:
            raise ValueError(f"Column name {column_name} does not exist or is not numerical")

    
        try:
            filtered_df = self.__data[(self.__data[column_name] < self.__lower_limit_dict[column_name]) |
                                    (self.__data[column_name] > self.__upper_limit_dict[column_name])]

            # Highlight selected column in the resulting filtered_df
            filtered_df_style = filtered_df.style.applymap(lambda x: 'background-color: yellow', subset=pd.IndexSlice[:, [column_name]])

            if styler==True:
                return filtered_df_style
            elif styler==False:
                return filtered_df

        except Exception as e:
            return f"Error occurred in filter_outlier(): {e}"
            
            

    def get_outliers_count(self, column_names=None, kind='num', decimal=2):
        
        if column_names is None:
            column_names = self.numeric_columns

        elif not isinstance(column_names, list):
                raise ValueError("column_names must be a list")
        for col in column_names:
            if col not in self.numeric_columns:
                raise ValueError(f"Column name {col} does not exist or is not numerical")
        else:
            column_names = [col for col in column_names if col in self.numeric_columns]


        outlier_counts = {}
        for name in column_names:
            filtered_df = self.get_outliers(name, styler=False)
            if isinstance(filtered_df, pd.DataFrame):
                outlier_counts[name] = filtered_df.shape[0]

        if not outlier_counts:
            return "No outlier found for given columns"

        if kind=='num':
            return pd.Series(outlier_counts)
        elif kind=='perc':
            total_rows = self.__data.shape[0]
            outlier_proportions_dict = {key: round(value * 100 / total_rows, decimal) 
                                        for key, value in outlier_counts.items()}
            return pd.Series(outlier_proportions_dict)
        else:
            raise ValueError(f"Invalid input for kind. only take 'num' & 'perc'")


    
    def plot_outlier_count(self, column_names=None, figsize=(10, 8), threshold_percent=5):
        
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
        plt.axhline(int(self.__data.shape[0]*threshold_percent/100), linestyle='--', color='r')
        plt.title('Outlier counts'.title(), fontsize=15, fontweight='bold')
        plt.xticks(rotation=90)
        plt.ylim(top=max(outlier_counts.values())*1.1) # set y-axis limit to fit the text annotations

        # add text annotations above each bar
        for i, v in enumerate(outlier_counts.values()):
            ax.text(i, v+max(outlier_counts.values())*0.05, str(v), color='black', ha='center', fontsize=10, fontweight='bold')

        plt.show()


    def drop_outliers(self, column_names=None):
        if column_names is None:
            column_names=self.numeric_columns
        elif not isinstance(column_names, list):
                    raise ValueError("column_names must be a list")
        for col in column_names:
            if col not in self.numeric_columns:
                raise ValueError(f"Column name {col} does not exist or is not numerical")
        else:
            column_names = [col for col in column_names if col in self.numeric_columns]

        index=[]

        for name in column_names:
            filtered_df = self.get_outliers(name, styler=False)
            if isinstance(filtered_df, pd.DataFrame):
                index = index + list(filtered_df.index)
        
        no_outlier_data=self.__data.drop(index=np.unique(index))
        return no_outlier_data

        
