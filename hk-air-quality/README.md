# HK Air Quality Prediction

This is an example of preparing a dataset and training a simple neural network to predict the NOX (nitrogen oxides) pollution quantity in the air in Hong Kong given the month, day, time (to the nearest hour), and the day's weather.

The set of model inputs are as follows:
- month: converted to one-hot encoding (1-12)
- day: converted to one-hot encoding (1-31)
- hour: converted to one-hot encoding (0-23)
- maximum temperature
- minimum temperature
- minimum grass temperature
- maximum relative humidity
- minimum relative humidity

The only output of the model:
- NOX

## Preparing the dataset

The dataset is downloaded from public sources provided by the HK Gov.

Air quality data is obtained from the [Environmental Protection Department](https://cd.epic.epd.gov.hk/EPICDI/air/station/?lang=en)
for which NOX data from 1/1/2020 to 28/2/2023 has been obtained.
This data has granularity up to the hour mark.

Weather data is obtained from the [HK Observatory](https://data.gov.hk/en-data/dataset/hk-hko-rss-weather-and-radiation-level-report).
This data has granularity only up to the daily mark.

Because the 2 sets of data have different time granularities, they need to be combined by copying the daily weather data points for each hour during the same day,
i.e. 1/1/2020 in the air quality dataset has a total of 24 rows of data for which each such row receives the same 1/1/2020 weather data.
See [data.ipynb](data.ipynb) for details.

## Training a DNN model

A simple deep neural network is designed.
Because temporal input (date and time) seems to have a large effect on the resulting NOX output,
this set of input goes through a few network layers first before being combined with the weather inputs.
In fact, the set of weather input also goes through a few network layers first.
The combined neurons then go through another intermediate layer before finally emitting a singular output that is desired.
See [model.ipynb](model.ipynb] for details.

