# Electricity-Price-Forecasting-ML
This project is part of the course _Scalable Machine Learning and Deep Learning_ at KTH. 

## Problem Description 
The prediction problem targeted in this project is electricity price forecasting. The electricity price prediction is usually based on data such as historical electricity prices, power generation data and weather data. The goal of the project is to build a serverless real-time machine learning system that is able to predict the daily average electricity price in the Stockholm region.

## Data Description
There are four main data sources used in this project: 
- **Elpris**, which provides historical and real-time hourly electricity price data in Sweden. The data is divided into four geographical regions in Sweden: SE1, SE2, SE3 and SE4, which is the standard division used by all entities working with electricity in Sweden. As Stockholm is located in SE3, the data used in this project is only from this region. The prices are provided in both Euro and SEK but only prices in SEK are used here. As the hourly electricity prices vary too much between day and night, making it too complex for a machine learning model to learn, the data is combined into a daily average instead to make the prediction problem simpler.
- **Open-meteo**, which provides historical and real-time weather data. The data used includes only relevant features for this prediction problem such as temperature, wind speed, precipitation, wind direction and sunshine duration.
- **ENTSO-e**, is a transparacy platform that has a central collection of electricity generation, transportation and consumption data in the pan-European region. The data used in this project was the electricity generation for Sweden, in which the generated electricity came from hydro water reservoir, nuclear,	wind onshore and a feature "other" which includes many smaller generation types. Due to Elpris being a daily average, the data for electricity generation also averaged.
## Solution Description 
The system architecture used in this project consists of four main pipelines that can be found in the notebooks folder:
- **Backfill pipeline**: in this pipeline, historical data is collected from the previously mentioned data sources. The collected data is then preprocessed and cleaned to make it ready to be saved. The final data is then saved into a feature store provided by Hopsworks. Data from each source is saved into a separate feature group resulting in three different groups(electricity prices, power generation and weather data).
- **Daily feature pipeline**: in this pipeline, real-time daily data is fetched from the different data sources and inserted into the different feature groups. This data will be later used in the inference pipeline to predict the daily electricity price based on it. This pipeline is scheduled to run daily using GitHub actions.
- **Training pipeline**: in this pipeline, the historical data stored in different feature groups on Hopsworks is fetched. Two different machine learning models are trained in this pipeline. The first model is an LSTM-based deep neural network as we are working with time series data. The second model is a gradient-boosted regrssion decision tree provided by the XGBoost library(XGBRegressor). For LSTM, the data is transformed using the MinMaxScaler from the Scikit-learn library. For XGBoost, the data is used directly as such transformation is not needed for it, however, lagged prices for three past days was added as features for the XGBoost model. The data is then split into training, validation and test data. For LSTM, the data is then transformed into sequences of length 3 to make it compatible with the model. The training data is used to train the model while the validation data is used for hyperparamter tuning. The test data is used for final evaluation. As this is a regression problem, the objective function used is the Mean Squared Error (MSE) is used for LSTM. The models are evalutated using both the MSE loss and the R^2 metric which measures how well the model fits the data. The best model is then chosen based on these two metrics.
- **Inference pipeline**: in this pipeline, the real-time daily data from the daily feature pipeline is fetched and the model from the training pipeline is also fetched from Hopsworks. This data consisted of one day weather data, one day electricity generation (average) and the three earlier electricity prices as lagged features and together they were inserted into the fetched model. The model predicts tomorrows electricity price and uploads it into a seperate feature group to be used in the dashboard.
- **Dashboard**: This dashboards purpose was to show the predicted values and true values over time to users. It is built as an Huggingface Space at this link [dashboard](https://huggingface.co/spaces/SWAH-KTH/el_price_predictions). Functionality wise it fetches the predicted values and true electricity price from Hopsworks that are created in the daily feature and inference pipelines. It then creates a graph and table to make it viewable to interested users.

## Results 
Here are the final training results of the LSTM and XGBoost models after hyperparameter fine-tuning
| Models | MSE  | R^2 |
| ------- | --- | --- |
| LSTM | 0.067 | -1.675 |
| XGBoost | 0.074 | 0.572 |

Based on these results the XGBoost model was chosen to perform the prediction in the application. 

Here is a snapshot from the GUI where the predictions are being displayed and monitored:
![image](https://github.com/user-attachments/assets/3f3b64bd-4389-4357-a1e4-01fca58da41f)
 
## How to run
Prerequisites: <br>
- Setup a Hopsworks account and create a new project <br>
- Generate an Hopsworks API key <br>
- For ENTSO-e you need to create an account and send an email request for an API key (might take a couple of days before they respond). <br>

Then: <br>
- git clone https://github.com/hishamad/Electricity-Price-Forecasting-ML.git <br>
- cd your-location <br>
- pip install -r requirements.txt <br>
- Create a .env file for API keys and add this code: <br>
  - HOPSWORK_API_KEY="hopsworks_key_here" <br>
  - ELECTRICTY_MAP_API_KEY="entso_key_here" <br>

