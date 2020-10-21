library(forecast)
library(ggplot2)
library(urca)

#Import Data - Japan Real GDP, Quarterly Data, seasonaly adjusted
gdp <- read.csv('JPNRGDPEXP.csv')
#Turn the data into time series format
gdp.ts <- ts(gdp$JPNRGDPEXP,start=c(1994,1),freq=4)

#Simple Data Plot
plot(gdp.ts,  ylab = "Real GDP", xlab = "Year", bty = "l",xaxt="n")
axis(1, at = seq(1994, 2020, 1)) 
grid()

#Log Difference Plot
gdp.log <- log(gdp.ts)
gdp.ts.diff <- diff(gdp.log, lag=1)
plot(gdp.ts.diff,  ylab = "GDP growth rate", xlab = "Year", bty = "l",xaxt="n")
axis(1, at = seq(1994, 2020, 1)) 
grid()

#Look into ACF and PACF plots in order to gain better understannding of the data
acf(gdp.ts)
pacf(gdp.ts)

# DF Test for stationarity
# Use trend ADF test because there's a clear trend in the GDP data, "BIC" since the dataset is large
df.test <- ur.df(gdp.log,type="trend",selectlags="BIC")
print(summary(df.test))
sumStats <- summary(df.test)
#Third statistic since type = 'trend'
teststat <- sumStats@teststat[3]
# critical value at 5 percent
critical <- sumStats@cval[2]
teststat > critical

#Use data until 2012 as training set and the rest as validation set
gdp.train.ts <- window( gdp.log, start=c(1994,1),end=c(2012,4))
gdp.valid.ts <- window( gdp.log, start=c(2013,1))
# length of full data set
T <- length(gdp.log)
# length of training data set
T2 <- length(gdp.train.ts)

# Arima model with trend 
mod1 <- Arima(gdp.train.ts,order=c(1,1,0),include.constant = TRUE)
mod1.fcast <- forecast(mod1,h=T-T2,level=95)

#plot forecase
plot(mod1.fcast, xlab="Year",ylab="Log(GDP)")
lines(gdp.log)
lines(gdp.log,lwd=1)
legend(x="topleft",c("Data","Forecast"),col=c("black","blue"),lwd=2)
grid()

#residual diagnosis: uncorrelated
plot(mod1$residuals)
ggAcf(mod1$residuals)

# ARIMA(d=1) model, because 1994-2020 is a lonng time horizon, use bic here
mod2 <- auto.arima(gdp.train.ts,d=1,ic="bic",seasonal=FALSE)
mod2.fcast <- forecast(mod2,h=T-T2,level=95)
plot(mod2.fcast,xlab="Year",ylab="Log(GDP)")
lines(gdp.log)
legend(x="topleft",c("Data","Forecast"),col=c("black","blue"),lwd=2)
grid()

# Following the suggestion in peer review, add another double exponential filtering/ Holt model wtih trend as comparison with my arima model as well
ses <- ets(gdp.train.ts, model="AAN")
ses.pred <- forecast(ses, h=T-T2, level=95)
plot(ses.pred,xlab="Year",ylab="Log(GDP)")
lines(gdp.log)
legend(x="topleft",c("Data","Forecast"),col=c("black","blue"),lwd=2)
grid()

# Compare the accuracy of three models
accuracy(mod1.fcast,gdp.log)
accuracy(mod2.fcast,gdp.log)
accuracy(ses.pred,gdp.log)
# Conduct DM test to see if my model is a better model comparing with baseline Arima(0,1,0) & the Holt model
dm.test(residuals(mod1.fcast),residuals(ses.pred),alternative="less")
dm.test(residuals(mod1.fcast),residuals(mod2.fcast),alternative="less")

