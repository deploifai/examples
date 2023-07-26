
## Car Selling Price Prediction App

This is a Streamlit app that predicts the selling price of a car based on its features. The app uses a random forest regression model that was trained on a dataset of car sales records. The data is taken from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) and is based on the indian car selling market. 

## Dependencies

The following Python libraries are required to run this script:

- `scikit-learn==1.3.0`
- `pickleshare==0.7.5`
- `numpy==1.25.1`
- `streamlit==1.25.0`
- `pandas==2.0.3`


### Files Required

The python script requires a pickle file to be saved in the same directory: `model.pkl` and a `requirements.txt` file that includes the name and versions of all the packages used. The pickle files contains the Random Forest Regressor model, which is loaded into the script using the pickle library. 

<!-- ## Usage

To run the app, simply run the `app.py` script using the following command:

```shell
 streamlit run app.py --server.address="0.0.0.0"
```
 -->

### Deploying the model 

For the deployment of this script, a Docker is created that includes all the necessary dependencies and files, with the Dockerfile specifying the base image and installation of the required Python libraries through pip, as well as the copying of the script and pickle files into the Docker image. 

On your terminal, you can pull the Docker image using the command:
```shell
docker pull deploifai/car-price-prediction
```

and to run the Docker image, use the command:
 ```shell
 docker run -it --rm -p 8501:8501 deploifai/car-price-prediction
 ```
 
After running the Docker image, the Car Price Prediction script can be accessed on the local host with the port `8501`.

 ```shell
 http://localhost:8501/
 ```