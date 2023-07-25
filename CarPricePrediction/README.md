
## Car Selling Price Prediction App

This is a Streamlit app that predicts the selling price of a car based on its features. The app uses a random forest regression model that was trained on a dataset of car sales records. 

## Dependencies

The following Python libraries are required to run this script:

- scikit-learn==1.3.0
- pickleshare==0.7.5
- numpy==1.25.1
- streamlit==1.25.0
- pandas==2.0.3


### Files Required

The python script requires a pickle files to be saved in the same directory:model.pkl and a requirements.txt file that includes the name and versions of all the packages used. The pickle files contains the Random Forest Regressor model, which is loaded into the script using the pickle library. 

## Usage

To run the app, simply run the `app.py` script using the following command:

```shell
 streamlit run app.py --server.address="0.0.0.0"
```

