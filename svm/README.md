# Email classification using SVM

Train a support vector machine (SVM) to classify emails as spam or not spam.

This is an example that uses Deploifai
[datasets](https://docs.deploif.ai/cloud-services/datasets/overview),
[training servers](https://docs.deploif.ai/cloud-services/training-servers/overview),
and [experiments](https://docs.deploif.ai/cloud-services/experiments/overview).

The following sections describe how to run this example yourself.

## Prerequisites

Make sure to do the following before running this example:

- [Install the Deploifai CLI](https://docs.deploif.ai/cli/install)
- [Create a cloud profile](https://docs.deploif.ai/cloud-services/connect-your-account) for a cloud provider of your choice

## Fork this repository

Fork this repository to your own GitHub account.
And clone it somewhere in your computer.

## Create a project

[Create a project](https://docs.deploif.ai/projects/quick-start#creating-a-project-in-the-dashboard)
in the Deploifai dashboard.

[Connect](https://docs.deploif.ai/projects/overview#how-to-connect-to-a-github-repository) the forked repository in your GitHub account to the project.

[Initialize](https://docs.deploif.ai/cli/commands/project#initialize-a-project)
the `examples` repository you just cloned as the project.

```shell
cd examples
deploifai project init
```

## Create a dataset

[Create a dataset](https://docs.deploif.ai/cloud-services/datasets/quick-start#creating-a-dataset)
named `dataset` in the new project you created.

Download the `emails.csv` file from this 
[Kaggle dataset](https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/)
into the `svm/dataset` directory.

[Initialize](https://docs.deploif.ai/cli/commands/dataset#initialize-a-dataset)
the dataset directory to mirror the dataset you created.

```shell
cd svm/dataset
deploifai dataset init -n dataset
```

[Push](https://docs.deploif.ai/cli/commands/dataset#upload-a-dataset)
the local `dataset` directory to the `dataset` on the cloud.

```shell
deploifai dataset push
```

## Create an experiment with a managed runner

[Create an experiment](https://docs.deploif.ai/cloud-services/experiments/quick-start/managed-runner#creating-an-experiment)
that runs on a managed runner.

1. Add the `dataset` that has been created
2. Setup the training server with a `MEDIUM CPU` instance size
3. Setup the experiment configuration according to the following:

   - pip requirements file: `svm/requirements.txt`
   - entrypoint shell script: `svm/experiment.sh`
   - artifacts directory: `svm/artifacts`

## Run the jupyter notebook

[Connect](https://docs.deploif.ai/cloud-services/training-servers/quick-start#connecting-to-the-training-server)
to the training server that has been created for the experiment.

Clone the forked repository to the training server.

Open the `training.ipynb` notebook and run the cells.

The notebook expects the dataset to be at `~/data/dataset` for which the `emails.csv` file is located at `~/data/dataset/emails.csv`.
And the artifacts directory to be located at `svm/artifacts` relative to the cloned repository.

## Run the experiment

The experiment runs the `experiment.sh` script which runs the `training.ipynb` notebook.

It will generate a `metrics.json` file and the model is saved as `model.pkl` in the artifacts directory.
These can be downloaded after the experiment run has finished.

## Clean up

Delete the following resources used in this example from the Deploifai dashboard.

- dataset
- training server
- experiment

## References

- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/
