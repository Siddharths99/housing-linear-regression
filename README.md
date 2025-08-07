# housing-linear-regression
This project demonstrates how to predict housing prices using a linear regression model built with Python. The dataset contains various property-related features such as area, number of bedrooms, bathrooms, stories, and whether facilities like a main road or guestroom are available.
The goal is to train a machine learning model that can estimate house prices based on these features.

The process includes:
1.Loading and exploring the dataset (Housing.csv)
2.Preprocessing data: handling missing values, converting categorical columns (like mainroad and guestroom) into numerical format
3.Selecting relevant features for model input
4.Splitting the dataset into training and testing sets(80% and 20%)
5.Training a linear regression model using Scikit-learn
6.Making predictions on test data
7.Evaluating model performance using:
a)Mean Absolute Error (MAE)
b)Mean Squared Error (MSE)
c)R² Score

The project also includes a simple linear regression section using only one input (e.g., area) to visualize the regression line on a scatter plot of house prices.

This project helps in understanding how regression models work and how to apply them to real-world datasets. It also introduces data cleaning, feature selection, and performance evaluation — essential steps in any machine learning workflow.

Libraries Used:-
Pandas
Scikit-learn
Matplotlib

📁Dataset
The dataset Housing.csv must be placed in the same directory for the notebook or script to work.

