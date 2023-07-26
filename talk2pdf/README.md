## PDF Chatbot - Extracting Information from Multiple PDFs

The Python script `app.py` is a Streamlit app that allows users to upload PDF files and use the app as a chatbot to gain information about the documents. The app uses several libraries including `langchain` for language processing, `PyPDF2` for reading PDF files, and `streamlit` for creating the app interface.

### Dependencies

The following Python libraries are required to run this script:

- langchain, PyPDF2, streamlit, tiktoken


### Usage

To use this script, simply execute the `main()` function in the script. The app interface allows the user to ask questions about the uploaded documents using the text input field. The user's question is handled using the `user_input_handler` function, which generates a response from the language model and displays it in the app interface. The app also keeps track of the chat history between the user and the language model using the `ConversationBufferMemory` from the `langchain` library.


### Files Required

The script requires a `htmlTemplates.py` file for formatting and a `requirements.txt` file that includes all the packages. 

### Deployment

For the deployment of this script, a Docker image was created that includes all the necessary dependencies and files, with the Dockerfile specifying the base image and installation of the required Python libraries through pip, as well as the copying of the script and other files into the Docker image. 

On your terminal after arriving in this directory, you can build your Docker image using the command:

```shell
docker build -t app .
```

 and to run the Docker image, use the command:
 ```shell
 docker run -it --rm -p 8501:8501 app
 ```
 
 After running the Docker image, the python script can be accessed on the local host with the port 8501.

 ```shell
 http://localhost:8051/
 ```
docker pull deploifai/talk-2-pdf

### Note

The app uses the OpenAI API to generate responses to user questions. Users will need to sign up for an OpenAI API key to use this feature and store it in an `env` file in the same directory.





