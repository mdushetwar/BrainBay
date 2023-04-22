import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jinja2


class OutlierNinja:

    """
    A class for detecting and handling outliers in a pandas DataFrame.

    Attributes:
    -----------
    data (pd.DataFrame): The input DataFrame containing the data to analyze.
    limit_factor: The threshold value to use for identifying outliers based on the IQR.
    """

    def __init__(self, limit_factor=1.5):

        """
        Initializes a new instance of the Outliers class.

        Parameters:
        -----------
        data (pd.DataFrame): The input DataFrame containing the data to analyze.
        limit_factor (float, optional): The threshold value to use for identifying outliers based on the IQR.
        """

        self.limit_factor = limit_factor
        self.__data = None
        self.numeric_columns = None
        self.__lower_limit_dict = {}
        self.__upper_limit_dict = {}
        self.__iqr = {}
        
    def fit(self, data):

        '''
        Fits the Outliers class to the provided dataset.

        Parameters:
        -----------
        data : pandas.DataFrame
            The input dataset to fit the Outliers class on.

        Returns:
        --------
        str
            A success message indicating that the upper and lower limits have been identified successfully.

        Raises:
        -------
        ValueError
            If the input dataset does not contain any numeric columns.

        Example:
        --------
        >>> from MGD_Outliers import Outliers
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
        >>> outlier = Outliers(limit_factor=1.5)
        >>> outlier.fit(df)
        'Upper and lower limits identified successfully!'
        '''

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

        """
        Returns a Pandas Series with the Interquartile Range (IQR) values for the specified columns.

        Parameter
        -----------
            column_names (list or None): A list of column names for which to calculate the IQR. If None, IQR values
                are calculated for all numeric columns in the dataset. Default is None.

        Returns
        -----------
            Pandas Series: A series with the IQR values for the specified columns.
        
        Raises
        -----------
            ValueError: 
                If any of the specified columns do not exist or are not numeric.
                If column_names is not list.

        Example
        ----------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            >>> outliers.get_iqr()
            >>> outliers.get_iqr(['age'])
        """

        
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

        """
        Compute the lower and upper limits for the specified columns.

        Parameters
        ----------
        column_names : list of str, optional
            Names of the columns to compute the limits for. If None (default),
            the limits for all numeric columns will be computed.
        decimal : int, optional
            Number of decimal places to round the limits to. Default is 4.

        Returns
        -------
        dict or str
            A dictionary of column names as keys and the corresponding lower and upper
            limits as a list [lower, upper] rounded to the specified number of decimal
            places. 

        Raises
        -----------
            ValueError: 
                If any of the specified columns do not exist or are not numeric.
                If column_names is not list.

        Example
        ----------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            >>> outliers.get_limits()
            >>> outliers.get_limits(['age'], decimal=2)

        """
        
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

    
    def detect_outliers(self, column_name, styler=True):

        """
        Returns a pandas dataframe with the outliers of a given column.

        Note: When returning a styler object, the user may need to have an environment with the required dependencies, 
        such as Jinja2, to properly view the styled dataframe. If the user does not have these dependencies, 
        they may see an unstyled version of the dataframe or an error message.

        Parameters
        ------------
            column_name (str): The name of the column to filter outliers for.
            styler (bool, optional): Whether to return a styled dataframe. Default is True.

        Returns
        ------------
            pandas.DataFrame: A dataframe containing the rows where the value in the given column is less than
                the lower limit or greater than the upper limit, as determined by the `fit()` method.
                If styler is True (default), a styled dataframe is returned with the cell containing the
                column name highlighted. If styler is False, an unstyled dataframe is returned.

        Raises
        ----------
            ValueError: If column_name is not a single string, or if the column name is not numeric.
            Exception: If an error occurs during the filtering process.

        Example
        ----------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            >>> outliers.detect_outliers('age')
            >>> outliers.detect_outliers('age', styler=False)

        """
        
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
            
            

    def get_outlier_count(self, column_names=None, kind='num', decimal=2):

        """
        Returns a Pandas Series with the count or proportion of outliers in each specified column.

        Parameters:
        -----------
        column_names : list or None, default None
            The list of column names to calculate outliers count/proportion for.
            If None, it will use all the numerical columns.
        kind : str, default 'num'
            Specifies the output kind. 'num' returns the count of outliers, while 'perc' returns
            the proportion of outliers as a percentage of the total number of rows in the dataset.
        decimal : int, default 2
            Specifies the decimal precision for the proportion output.

        Returns:
        --------
        A Pandas Series with the count or proportion of outliers in each specified column.
        If no outlier is found for the given columns, returns the string "No outlier found for given columns".

        Raises:
        -------
        ValueError: 
            If the given column_names is not a list, or any of the column_names does not exist or is not numerical.
            If the input value for kind is invalid. Only takes 'num' and 'perc'.
            If column_names is not list.

        Example:
        ----------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            >>> outliers.get_outlier_count()
            >>> outliers.get_outlier_count(['age'])

        """
        
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
            filtered_df = self.detect_outliers(name, styler=False)
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

        '''
            Plot the number of outliers for each column as a bar chart.
    
        Parameters:
        -----------
        column_names : list or None, optional
            A list of column names to plot. If None, all numeric columns are plotted.
        figsize : tuple, optional
            Figure size of the plot. Default is (10, 8).
        threshold_percent : float, optional
            The threshold value in percentage above which a column is considered to have a high number of outliers.
            Default is 5%.
        
        Raises:
        -------
        ValueError
            If column_names is not a list or if any column name in column_names is not numerical.
            If column_names is not list.
        
        Returns:
        --------
        None
        The function only plots the bar chart.
        
        Example:
        --------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            >>> outliers.plot_outlier_count(column_names=['Age', 'Income'], figsize=(12, 6), threshold_percent=10)
        '''
        
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
            filtered_df = self.detect_outliers(name, styler=False)
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

        '''
        Drop rows with outliers from the data for specified columns.

        Parameters:
        -----------
        column_names : list or None, optional (default=None)
            List of column names to drop outliers from. If None, drops outliers from all numeric columns.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame without the rows containing outliers.
        
        Raises:
        -------
        ValueError
            If column_names is not a list or any of the specified columns do not exist or are not numeric.
            If the column_names is not list.

        Example:
        --------
            >>> from MGD_Outliers import Outliers
            >>> outliers = Outliers()
            >>> outliers.fit(data)
            # Drop outliers from the 'age' column and return the updated DataFrame
            >>> outliers.drop_outliers(['age'])
            # Drop outliers from all numeric columns and return the updated DataFrame
            >>> outliers.drop_outliers()

        '''

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
            filtered_df = self.detect_outliers(name, styler=False)
            if isinstance(filtered_df, pd.DataFrame):
                index = index + list(filtered_df.index)
        
        no_outlier_data=self.__data.drop(index=np.unique(index))
        return no_outlier_data

        
