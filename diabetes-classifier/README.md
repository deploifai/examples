# Diabetes Classifier using KNN

This is a simple diabetes classifier built using the K-Nearest Neighbors (KNN) algorithm. The classifier is implemented as a Streamlit app, which allows users to input their medical information and find out whether they have diabetes or not. The data is taken from [Kaggle↗](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The model creation process can be found in `main.ipynb` file.


## Dependencies

The following Python libraries are required to run this script:


- `pickleshare==0.7.5`
- `numpy==1.25.1`
- `streamlit==1.25.0`
- `pandas==2.0.3`


## App Details

The app takes in the following medical information as input:

* Number of Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

Using this information, the app makes a prediction on whether the user has diabetes or not. The prediction is made using a KNN model that has been trained on the Pima Indian Diabetes Dataset.


### Deploying the model 

For the deployment of this script, a Docker image is created that includes all the necessary dependencies and files, with the Dockerfile specifying the base image and installation of the required Python libraries through pip, as well as the copying of the script and the pickle file into the Docker image. 

On your terminal, you can pull the Docker image using the command:
```shell
docker pull deploifai/diabetes-classifier
```

and to run the Docker image, use the command:
 ```shell
 docker run -it --rm -p 8501:8501 deploifai/diabetes-classifier
 ```
 
After running the Docker image, the Diabetes classifier web application can be accessed on the local host with the port `8501`.

 ```shell
 http://localhost:8501/
 ```
