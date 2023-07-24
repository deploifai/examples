## Safe Tweet - Deploying an ML Model to Classify Cyberbullying Tweets

The Python script `app.py` provides a preprocessing and classification model for cyberbullying tweets. It uses various NLP techniques to preprocess the input text and classify it into different types of cyberbullying. The model is built using the scikit-learn machine learning library, specifically using the Support Vector Machine (SVM) algorithm and is trained on a dataset of cyberbullying tweets. To provide an easy-to-use interface for users, the model is deployed using Gradio, a Python library that allows for the creation of customizable web interfaces.


### Dependencies

The following Python libraries are required to run this script:

- scikit-learn==1.3.0
- pickleshare==0.7.5
- numpy==1.25.1
- gradio==3.37.0
- nltk==3.8.1
- regex==2023.6.3
- emoji==2.6.0


### Usage

To use this script, you can import it into your Python code and call the `conversion` function, passing in a string containing the text of the tweet you want to classify. The function will preprocess the text using NLP techniques and then classify it into one of the following six categories:

1. Religion
2. Age
3. Ethnicity
4. Gender
5. Others
6. Not Cyberbullying


### Files Required

The script requires two pickle files to be saved in the same directory: `vectorizer.pkl` and `model.pkl` and a `requirements.txt` file that includes all the packages. The pickle files contain a pre-trained vectorizer and classification model, respectively, which are loaded into the script using the `pickle` library. 


### Gradio Web Interface

The script includes a Gradio web interface for easy testing of the classification model. The `demo` function creates a simple web interface that allows the user to enter a tweet and get an instant prediction of the type of cyberbullying it contains. The web interface can be launched by calling the `launch` method of the `demo` object.


### Deploying the model 

For the deployment of this script, a Docker is created that includes all the necessary dependencies and files, with the Dockerfile specifying the base image and installation of the required Python libraries through pip, as well as the copying of the script and pickle files into the Docker image. 

On your terminal, you can pull the Docker image using the command:
```shell
docker pull deploifai/safetweet
```

 and to run the Docker image, use the command:
 ```shell
 docker run -it --rm -p 7860:7860 deploifai/safetweet
 ```
 
 After running the Docker image, the Safe Tweet script can be accessed on the local host with the port 7860.

 ```shell
 http://localhost:7860/
 ```