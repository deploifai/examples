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

## Upload dataset to the data storage

Once the set up has been completed, upload the data using:

```
deploifai data push
```

## References

- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv/
