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
```

5. Download Sarcasm Data:
```
Need to download sarcasm data from kaggle: https://www.kaggle.com/danofer/sarcasm/
The data should be kept in main folder
```

## Modeling Files

The main code to build models is contained in only one file:

- `train.py`

There are two kinds of models in this code: `light_weight` and  `fine_tune`

- The light_weight model freezes the LM and only trains the last added  layer
- The fine_tune model trains full LM layers

### Training command:

The following command trains the `light_weight` and  `fine_tune`:

```
#By default it trains light weight with below code:
1. python train.py

#Second option is with fine_tuning
2. python main.py --tuning_mode 'fine_tune'

```
### Training hyper-parameters
```
optional arguments:
  -h, --help            show this help message and exit
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
  --tuning_mode {light_weight,fine_tune}
                        tuning_mode
```

## Disclaimer

We are trying to do distributed training using `distributed_training.py`, which is a work in progress. We just wanted to include this in the submission to show what we are currently working on.
