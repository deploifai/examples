import run_classifier
import os
import tensorflow as tf
import json
import pandas as pd

tokenization = run_classifier.tokenization
model_base_path = './model' #modify accordingly
init_checkpoint = os.path.join(model_base_path, 'model.ckpt')
bert_config_file = os.path.join(model_base_path, 'bert_config.json')
vocab_file = os.path.join(model_base_path, 'vocab.txt')
processor = run_classifier.ColaProcessor()
label_list = processor.get_labels()
emotions_file = 'emotions.txt'

#since the original bert source code combines train, eval and predict in one single configuration,
#we need to feed such data during initialization, can be anything as it is needed for run configuration
BATCH_SIZE = 8
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 500
OUTPUT_DIR = "./output"

#variables that needed to be modified
labels = [str(i) for i in range(28)] #modify based on the labels that you have
MAX_SEQ_LENGTH = 50 #modify based on the seq length
is_lower_case = True #modify based on uncased or cased

#variables for configuration
tokenization.validate_case_matches_checkpoint(is_lower_case, init_checkpoint)
bert_config = run_classifier.modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=is_lower_case)
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
    model_dir=OUTPUT_DIR,
    cluster=None,
    master=None,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=1000,
        num_shards=8,
        per_host_input_for_training=is_per_host))

#model
model_fn = run_classifier.model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
    init_checkpoint=init_checkpoint,
    learning_rate=5e-5,
    num_train_steps=None,
    num_warmup_steps=None,
    use_tpu=False,
    use_one_hot_embeddings=False)

#estimator
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    predict_batch_size=BATCH_SIZE)

# emotions
emotions = pd.read_csv(emotions_file, header=None)

def predict(sentence):
    input_example = run_classifier.InputExample(guid="", text_a = sentence, text_b = None, label = "0") # here, "" is just a dummy label
    input_examples = [input_example]

    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

    predictions = estimator.predict(input_fn=predict_input_fn)

    df = pd.DataFrame(predictions)
    label_index = df.iat[0, 0].argmax()
    confidence = df.iat[0,0].max()
    predicted_label = emotions.iat[int(label_index), 0]

    return {'predicted_label': predicted_label, 'confidence': confidence}

def main():
    sentence = "Today is an terrible day."

    print(predict(sentence))

if __name__ == '__main__':
    main()
