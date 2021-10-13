# Email classification using SVM

In this example, we create a machine learning project to classify emails as spam or not spam.

The code also uses Deploifai to manage the training data and training servers.

The dataset has been taken from [Kaggle](https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/)

## Initialise deploifai project

In the project directory, create a `data` directory that contains the dataset.

```
mkdir data
```

Now you can initialise a Deploifai dataset using the CLI

```
deploifai data init
```

Follow the commands and connect at least one container of the data storage to the `data` directory.



https://user-images.githubusercontent.com/19630580/137169299-40f85376-8924-486a-841e-2b6cd7f5ed74.mp4



## Upload dataset to the data storage

Once the set up has been completed, upload the data using:

```
deploifai data push
```

## Run training on the repo

Create a training server on Deploifai using the [dashboard](https://deploif.ai).

Login to the server using the given instructions and then clone this repo on the server.

```
git clone https://github.com/deploifai/examples
```

Make the change to the `data` variable in the notebook to the right path. You can find the path at the bottom in the dashboard page for the training server.

You can run the notebook! Watch the video below to see all the steps in action.


https://user-images.githubusercontent.com/19630580/137170053-bbc37b56-8201-47e9-82d6-7a831f560cbf.mp4


## References

- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/
