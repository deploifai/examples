# MNIST

Train a simple deep CNN on the MNIST small images dataset using Tensorflow Keras.
This example is designed to illustrate how an 
[experiment](https://docs.deploif.ai/cloud-services/experiments/overview)
can be setup to run on a managed runner.

## Experiment

An experiment needs to be 
[configured](https://docs.deploif.ai/cloud-services/experiments/setup)
before it can be run.

### Python requirements

The python requirements are specified in the `requirements.txt` file.

### Entrypoint shell script

The entrypoint shell script is the `experiment.sh` file
which simply runs the `train.py` python script.

### Artifacts directory

The artifacts directory is the `artifacts` directory.
The test loss and accuracy of the model is saved in the `artifacts/metrics.json` file,
whereas the model is saved in the `artifacts/model` directory.

## Create an experiment

If you want to run this example in your own experiment.

Follow the following steps:

1. Fork this repository.
 
2. Create an experiment to run on a managed runner.
    Designate the following configuration to setup your experiment.
    - pip requirements file: `mnist/requirements.txt`
    - entrypoint shell script: `mnist/experiment.sh`
    - artifacts directory: `mnist/artifacts`

3. [Connect](https://docs.deploif.ai/projects/overview#how-to-connect-to-a-github-repository) your forked repository to the Deploifai [project](https://docs.deploif.ai/projects/overview) where you created the experiment.

4. Run the experiment by clicking on the "Start run" button in the experiment page.
