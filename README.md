# MGD_Outliers
MGD_Outliers is an open-source Python package for detecting and analyzing outliers in data. The package provides a class, Outliers, which can be used to detect outliers in numerical data, and provides several methods to analyze and visualize the detected outliers.

# Current version
 version 0.1.1

# Installation

You can install MGD_Outliers using pip:

    pip install MGD_Outliers
    

# Usage

To use the Outliers class in MGD_Outliers, you first need to import the package:

    from MGD_Outliers import Outliers
    

Then, you can create an instance of the Outliers class and pass your data to it:

    outliers = Outliers(data)
 

Once you have an instance of the Outliers class, you can use its methods to detect and analyze outliers. For example, you can use the detect_outliers method to detect outliers in the data:

    outliers.get_outliers()
   

You can also use the plot_outlier_count method to plot barplot of the columns with the detected outliers highlighted:

    outliers.plot_outlier_count()


For a full list of methods available in the Outliers class, see the documentation.

# License
This package is licensed under the Apache License 2.0. See the LICENSE file for more information.


Contributing
If you find any bugs or issues with MGD_Outliers, or if you have any suggestions for new features, please open an issue on GitHub. If you would like to contribute to the development of MGD_Outliers, please fork the repository and submit a pull request.



