import tensorflow as tf
import os, json

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

evaluation = model.evaluate(x_test, y_test, verbose=2)

metrics = {
    "loss": evaluation[0],
    "accuracy": evaluation[1],
}

print(metrics)

artifacts_directory = "artifacts"
with open(os.path.join(artifacts_directory, "metrics.json"), 'w') as metrics_file:
    json.dump(metrics, metrics_file, indent=2)

model.save(os.path.join(artifacts_directory, "model"))
