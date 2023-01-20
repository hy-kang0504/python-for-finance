# python-for-finance

In this repository, I will share data analysis projects using Python. The projects will revolve around finance-related topics like macroeconomic analysis, trading strategies and portfolio allocation strategies among many others.

## FRED

In this project, I use Federal Reserve Economic Data(FRED) to see how these economic indicators can predict future economic regimes. FRED provides growth, employment, inflation, labor, manufacturing and other US economic statistics from the research department of the Federal Reserve Bank of St. Louis.

I use the FRED API to pull down necessary data and scikit-learn to perform predictive analysis.

My FRED project comes in 3 main parts:
1) Data preparation
2) Train and test classification models 
3) Feature selection

#### Data Preparation
In part 1, I download economic indicators through the API and clean / transform the dataset so it can be used in further research.

#### Train and Test Classification Models
In part 2, I use different classification models provided by scikit-learn to find out which model and parameters work best for forecasting future economic regimes.

#### Feature Selection
In part 3, I select k best features (or indicators) used to forecast future economic regimes. I explore whether the k most important features change over time. Finally, I employ the latest k best features to predict recession probability in the next several months.

## PORTFOLIO ALLOCATION

In this project, I perform analysis that helps make portfolio allocation decisions. I look at correlation matrices over time across various asset classes, experiment with different weighting schemes and more. 
