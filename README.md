# Time-Series-Forecasting-Model

The main goal of this project is to create a forecasting model and evaluate its performance comparing with other baseline models. The dataset I used for this project is Japan real GDP from 1994-2020 in quarterly format and seasonally adjusted.\

The very first stage of this project was understanding the dataset thoroughly and finding out which model is appropriate to use. After I imported the data and turned data into time series format, I plotted a simple time plot and a log difference plot to gain a general understanding of the dataset. I found that Japan GDP is following an upward slop trend with a gap in 2009, but there’s no clear trend in GDP growth. Then, I used ACF and PACF plots to help with determining suitable model. In this case, because ACF showed autocorrelation dies out slowly as lag increases, while PACF dropped to 0 suddenly after lag 1, it could be appropriate to use AR (1) model here. I also conducted Augmented Dickey-Fuller test and was not able to reject the null hypothesis. This is a Random walk plus drift, which make sense given that there is a growth rate. Thus, I decide to use Arima model since Arma can only be used when the time series data is stationary. With non-stationary data, Arima can be used to take the first difference and forecast based on the difference. Combining this rule with the information that ACF and PACF provided, I decided to use Arima (1,1,0) model with drift. \

I used about 70% (1994-2012) of my data as training set and 30% (2012-2020) as validation set. As comparison with my model, I also created an Arima (0,1,0) random walk model and an ETS(AAN)/ Holt model with trend using the same training set. I plotted a graph for all three models and their forecasting results. It’s clear that Arima (1,1,0) with drift and Holt model performed better than Arima (0,1,0). In order to evaluate the forecasting performance, I printed out the accuracy of all three models and conducted two DM tests to see if my model is significantly better than the other two models. Even though the DM tests results showed that my model was not significantly better in 5% level, the accuracy of my model performance is better than the Holt model and way better than Arima(0,1,0) model. The RMSE of my Arima(1,1,0) with drift model is 0.0185, while the RMSE of Arima(0,1,0) and Holt model are 0.0517 and 0.0216, respectively. The p-value of DM tests also comply with the forecasting result plots and accuracy information: it’s more likely that my Arima(1,1,0) with drift model is better than Arima(0,1,0) model comparing with the chance that my Arima(1,1,0) with drift model is better than the Holt model.
