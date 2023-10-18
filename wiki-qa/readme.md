# Wiki-QA

Based on [Wiki Question and Answer Retrieval](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/semantic-search#question--answer-retrieval) example from [sentence-transformers](https://github.com/UKPLab/sentence-transformers), this example uses a model that was trained on the [Natural Questions dataset](https://ai.google.com/research/NaturalQuestions/). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an asymmetric search task. As corpus, the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) was used so that it fits easily into memory.

This example is a deployment of this model.

### Docker Image

Build this image:

```shell
docker build -t wiki-qa .
```

A publicly available image is also available on [Docker Hub](https://hub.docker.com/r/deploifai/wiki-qa).


### Run

This image requires an Nvidia GPU with drivers that can support cuda 11.6.

```shell
docker run -p 8000:8000 wiki-qa
```

### API

The API is a simple REST API that takes a question as input and returns a list of answers.

```shell
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"question\":\"What is the capital of Germany?\"}"
```
