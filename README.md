# Prefix-Tuning-Bert
## Installation

This assignment is implemented in python 3.6 and torch 1.9.0. Follow these steps to setup your environment:

1. [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-project python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-project
```
4. Install the requirements:
```
pip3 install -r requirements.txt
cd transformers; pip install -e .;
```

5. Download Sarcasm Data:
```
Need to download sarcasm data from kaggle: https://www.kaggle.com/danofer/sarcasm/
The file can be passed as an input argument during runtime
```

## Modeling Files

The main code to build models is contained in only one file:

- `train.py`

There are several tuning_modes that can be chosen from to train the models 
* baseline_finetune
* baseline_lightweight_finetune
* prefix_bottom_two_layers
* prefix_top_two_layers
* prefix_bert_embedding_layer
* prefix_custom_initializaition
* prefix_random_initializaition
* noprefix_top_two_layers
* noprefix_bottom_two_layers
* noprefix_embedding_layer_update

### Training command:

Below are some of the examples of training command:

```
# By default train.py trains prefix tuning with random embedding initializaition of 5 tokens:
# However all the parameters are configurable with the arguments described in hyper-parameters section.

# Default training:
python train.py

# Training with different tuning mode
python train.py --tuning_mode noprefix_top_two_layers
```
### Training hyper-parameters
```
optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        training dataset file that have to be used
  --prepare_data        if passed, will prepare data.
  --save_processed_data
                        if passed, save the processed data.
  --batch_size BATCH_SIZE
                        batch_size
  --custom              if passed, use no custom.
  --epochs EPOCHS       epochs
  --learning_rate LEARNING_RATE
                        learning_rate
  --save_model          if passed, save model.
  --prefix_length PREFIX_LENGTH
                        number of prefix tokens
  --model_save_directory MODEL_SAVE_DIRECTORY
                        save the model to
  --tuning_mode {baseline_finetune,baseline_lightweight_finetune,prefix_bottom_two_layers,
                  prefix_top_two_layers,prefix_bert_embedding_layer,
                  prefix_custom_initializaition,prefix_random_initializaition,noprefix_top_two_layers,
                  noprefix_bottom_two_layers,noprefix_embedding_layer_update}
                        Name of the tuning_mode
  --use_multi_gpu USE_MULTI_GPU
                        Use Multiple GPUs
  --phrase_for_init PHRASE_FOR_INIT
                        If using custom initialization this will be used to
                        initialize the prefix tokens
  --checkpoint CHECKPOINT
                        to checkpoint the model at each epoch
  --analyze_tokens ANALYZE_TOKENS
                        Closest words in bert vocab in each epoch are
                        extracted
  --test_file TEST_FILE
                        test file that have to be used
  
  --evaluate            To run the script in Evaluation mode
  
  --saved_model_location SAVED_MODEL_LOCATION
                        Loaction of the stored model, must be used when only
                        evaluation is called
```

## Disclaimer

We are trying to do distributed training using `distributed_training.py`, which is a work in progress. We just wanted to include this in the submission to show what we are currently working on.
