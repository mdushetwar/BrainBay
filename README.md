# MGD_Outliers
MGD_Outliers is an open-source Python package for detecting and analyzing outliers in data. The package provides a class, OutlierNinja, which can be used to detect outliers in numerical data, and provides several methods to analyze and visualize the detected outliers in quick and efficient manner.

# Current version
 version 0.1.4

# Installation

You can install MGD_Outliers using pip:

    pip install MGD_Outliers
    

# Usage

To use the OutlierNinja class in MGD_Outliers, you first need to import the package:

    from MGD_Outliers import OutlierNinja
    

Then, you have to create an instance of the OutlierNinja class:

    outliers = OutlierNinja(limit_factor=1.5)   # 1.5 is a default value, you can change it as per your project requirement
 

Then, you need to train this object using a dataframe object

    outliers.fit(data)


Once the OutlierNinja object is trained, you can sit back, relax with your favorite drink and call its methods and attributes like a true data ninja. 
For example, you can use the detect_outliers method to locate outliers in the data:

    outliers.detect_outliers(column_name='age')
   

You can also use the plot_outlier_count method to plot barplot of the columns:

    outliers.plot_outlier_count()


For a full list of methods available in the OutlierNinja class, see the documentation. I hope you find this package helpful. So, let the OutlierNinja package do the hard work, while you sip on your drink and reap the benefits of a well-optimized dataset.


# License
This package is licensed under the Apache License 2.0. See the LICENSE file for more information.


# Contributing
If you find any bugs or issues with MGD_Outliers, or if you have any suggestions for new features, please open an issue on GitHub. If you would like to contribute to the development of MGD_Outliers, please fork the repository and submit a pull request.



