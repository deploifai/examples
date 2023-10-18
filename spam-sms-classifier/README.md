# Spam SMS Classifier

This is a spam SMS classifier that uses a machine learning model to detect whether a user input string is a spam SMS or not. The model is built using the scikit-learn machine learning library, specifically using the Naive Bayes algorithm and is trained on a dataset of 5572 text messages collected by the [UCI Machine Learning Repository
(Almeida & Hidalgo, 2012)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

To provide an easy-to-use interface for users, the model is deployed using Gradio, a Python library that allows for the creation of customizable web interfaces.

## Dependencies

Make sure you have the following python libraries installed before running the code:
- `scikit-learn==1.3.0`
- `pickleshare==0.7.5`
- `gradio==3.37.0`
- `numpy==1.25.1`
- `pandas==2.0.3`


## Model

The `model.pkl` file contains the machine learning model, which is loaded into memory using `pickle` in the script. The `main.ipynb` file contains the complete process of creating the model.

## Usage

To use the spam SMS classifier, run the script and launch the `gradio` interface. by running the following command:
```shell
python app.py
```

The interface will prompt you to type your SMS text in the input textbox. Once you submit your input, the model will predict whether the text is a spam SMS or not and return a corresponding message in the output textbox.

## Function

The `results` function takes a user input string and returns a message indicating whether the text is a spam SMS or not. The function first converts the input string to a pandas Series object and then uses the pre-trained machine learning model to make the prediction. If the prediction is 1, the function returns "Spam SMS detected". Otherwise, it returns "The SMS is NOT spam".

## Gradio Web Interface

The `app.py` script includes a Gradio web interface for easy testing of the classification model. The demo function creates a simple web interface that allows the user to enter an SMS and get an instant prediction whether it is a spam message or not. The web interface can be launched by calling the launch method of the demo object.

### Deploying the model 

For the deployment of this script, a Docker image is created that includes all the necessary dependencies and files, with the Dockerfile specifying the base image and installation of the required Python libraries through pip, as well as the copying of the script and pickle files into the Docker image. 

On your terminal, you can pull the Docker image using the command:
```shell
docker pull deploifai/spam-sms-classifier
```

and to run the Docker image, use the command:
 ```shell
 docker run -it --rm -p 7860:7860 deploifai/spam-sms-classifier
 ```
 
After running the Docker image, the Spam SMS Classifier script can be accessed on the local host with the port `7860`.

 ```shell
 http://localhost:7860/
 ```
