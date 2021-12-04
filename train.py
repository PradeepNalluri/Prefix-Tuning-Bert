from urllib import parse
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import torch
import torch.nn as nn
import multiprocessing as mp
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from SARCBertClassifier import SARCBertClassifier
from keras.preprocessing.sequence import pad_sequences
from string import ascii_uppercase
from tqdm import tqdm
import seaborn as sn
import time
import datetime
import pickle
import os
import random
import itertools
import json
import argparse

def get_bert_embedding(word):
    """
    Create embedding for phrases this can be used to initialize the prefix embeddings based on phrases
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_res = tokenizer.encode_plus(
                            word,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 5,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
    embed_model = BertModel.from_pretrained('bert-base-uncased')
    embed_model.eval()
    embedding = embed_model(**tokenized_res)
    embedding = embedding.last_hidden_state.detach()#[:,tokenized_res['attention_mask']]    
    required_embedding = embedding[0][tokenized_res['attention_mask'][0]==1,:]
    required_embedding = required_embedding.mean(dim=0)
    del embed_model
    del embedding 
    return required_embedding

def parallelize(function_pointer,list_to_parallelize,NUM_CORE=2*mp.cpu_count()):
    '''
    Prallel apply the given function to the list the numeber of process will 
    be twice the number of cpu cores by default 
    '''
    start=time.time()
    component_list=np.array_split(list_to_parallelize,NUM_CORE*10)
    pool = mp.Pool(NUM_CORE)
    results = pool.map(function_pointer,component_list)
    pool.close()
    pool.join()
    end=time.time()
    print("Executed in:",end-start)
    return results

def find_max_length(sentences,tokenizer):
    """
    Find the max length of the senteces
    """
    max_len = 0
    for _,row in sentences.iterrows():
        sent=row["comment"]
        try:
            train_inputs_ids = tokenizer.encode(sent, add_special_tokens=True)
        except:
            train_inputs_ids = tokenizer.encode("", add_special_tokens=True)
        max_len = max(max_len, len(train_inputs_ids))
    return max_len

def compute_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))

def main(args):
    prepare_data = args.prepare_data
    save_processed_data = args.save_processed_data
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    save_model = args.save_model
    tuning_mode = args.tuning_mode
    model_save_directory = args.tuning_mode
    prefix_tuning = True if "prefix" in tuning_mode else False
    experiment = args.experiment_type
    use_multi_gpu = args.use_multi_gpu
    phrase_for_init = args.phrase_for_init
    checkpoint = args.checkpoint
    analyze_tokens = args.analyze_tokens
    test_file = args.test_file
    train_model = args.train_model

    if(prefix_tuning):
        prefix_length = args.prefix_length

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using:",device)
    
    if(train_model):
        if(prepare_data):
            data = pd.read_csv("train-balanced-sarcasm.csv")
            training_set,test_set = train_test_split(data,stratify=data[["label"]], test_size=0.1)
            del data

            #Storing for future use across experiments
            test_set.to_csv("test_set.csv",index=False)

            training_set.dropna(subset=["comment"],inplace=True)
            training_set.reset_index(drop=False,inplace=True)
            training_set.rename(columns={"index":"id"},inplace=True)
            sentences = training_set[["id","comment"]]
            labels = training_set[["id","label"]]

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            max_len = max(parallelize(tokenizer,sentences))

            sentences=training_set.comment.values
            labels = training_set.label.values

            train_inputs_ids = []
            training_attention_masks = []

            #Tokenizing the sentences
            for sent in tqdm(sentences):
                encoded_sentences = tokenizer.encode_plus(sent,add_special_tokens = True,max_length = 64,pad_to_max_length = True,
                return_attention_mask = True,return_tensors = 'pt',)
                
                train_inputs_ids.append(encoded_sentences['train_inputs_ids'])
                
                training_attention_masks.append(encoded_sentences['attention_mask'])

            train_inputs_ids = torch.cat(train_inputs_ids, dim=0)
            training_attention_masks = torch.cat(training_attention_masks, dim=0)
            labels = torch.tensor(labels)
            #save data for future use
            if(save_processed_data):
                f = open('processed_data.pckl', 'wb')
                pickle.dump([train_inputs_ids,training_attention_masks,labels], f)
                f.close()
        
        else:
            #lOAD THE DATA
            f = open('processed_data.pckl', 'rb')
            input_processed_data = pickle.load(f)
            f.close()
            train_inputs_ids = input_processed_data[0]
            training_attention_masks = input_processed_data[1]
            labels = input_processed_data[2]
        
        print("Data Preperation Done")

        main_dataset = TensorDataset(train_inputs_ids, training_attention_masks, labels)

        train_size = int(0.9 * len(main_dataset))
        val_size = len(main_dataset) - train_size

        train_dataset, validation_data = random_split(main_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size,)

        validation_dataloader = DataLoader(validation_data,sampler = SequentialSampler(validation_data), batch_size = batch_size,)
    
        config = BertConfig.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            )
        if(prefix_tuning):
            config.prefix_length = prefix_length
            phrase_for_init = phrase_for_init
        
        if prefix_tuning:
            model = SARCBertClassifier(config)
            model.update_network_sarc(2,device,freeze_bert_layers=True)
            
            if(experiment == "prefix_bottom_two_layers"):
                for n,p in model.named_parameters():
                    if(n=="prefix_embeddings.weight" or "bert.encoder.layer.0." in n or "bert.encoder.layer.1." in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
            
            elif(experiment == "prefix_top_two_layers"):
                for n,p in model.named_parameters():
                    if(n=="prefix_embeddings.weight" or "bert.encoder.layer.10." in n or "bert.encoder.layer.11." in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
            
            elif(experiment == "prefix_bert_embedding_layer"):
                for n,p in model.named_parameters():
                    if(n=="prefix_embeddings.weight" or "bert.embeddings.word_embeddings.weight" in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
            
            elif(experiment == "prefix_custom_initializaition"):
                del model
                custom_embedding =  get_bert_embedding(phrase_for_init)
                model = SARCBertClassifier(config)
                model.update_network_sarc(2,device,freeze_bert_layers=True,custom_embedding=True,custom_embedding_vector=custom_embedding)

                for n,p in model.named_parameters():
                    if(n=="prefix_embeddings.weight" or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
            
            elif(experiment == "prefix_random_initializaition"):
                for n,p in model.named_parameters():
                    if(n=="prefix_embeddings.weight" in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)

            else:
                raise Exception("Exception: Unknow Experiment")
        else:
            model = BertForSequenceClassification(config)
            
            if(experiment == "noprefix_top_two_layers"):
                for n,p in model.named_parameters():
                    if("bert.encoder.layer.10." in n or "bert.encoder.layer.11." in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
            
            elif(experiment == "noprefix_bottom_two_layers"):
                for n,p in model.named_parameters():    
                    if("bert.encoder.layer.0." in n or "bert.encoder.layer.1." in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
        
            elif(experiment == "noprefix_embedding_layer_update"):
                for n,p in model.named_parameters():    
                    if("bert.embeddings.word_embeddings.weight" in n or n=="classifier.weight"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    if p.requires_grad:
                        print("Tuning:",n)
        
            else:
                raise Exception("Exception: Unknow Experiment")
        
        if(use_multi_gpu and torch.cuda.device_count()>1):
            print("Parallelizing Model")
            model = nn.DataParallel(model)
            model.to(device)
            model = model.cuda()
        
        print("Model Initialization Done")

        optimizer = AdamW(model.parameters(),lr = learning_rate,eps = 1e-8)

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)

        print("Optimizer setup done")

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        training_stats = []

        total_t0 = time.time()
        
        for epoch_i in range(0, epochs):
            t0 = time.time()

            batch_train_loss = 0

            model.train()
            
            total_training_loss = 0
            correct_preds, total_predictions = 0, 0
            generator_tqdm = tqdm(train_dataloader)
            if(checkpoint):
                output_dir = model_save_directory+"_checkpoint/"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print("Saving model to %s" % output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir) 
                if(analyze_tokens):
                    with open(output_dir+"prefix_embed_matching_words_epoch_"+str(epoch_i)+".json", 'w') as fp:
                        json.dump(model.closest_matching_bert_model(), fp)           
            for step, batch in enumerate(generator_tqdm):
                if(prefix_length):
                    b_input_ids = torch.cat((torch.arange(0, config.prefix_length).expand(batch[0].shape[0], config.prefix_length),batch[0]),1).to(device)
                    b_input_mask = torch.cat((torch.ones(config.prefix_length).expand(batch[1].shape[0], config.prefix_length),batch[1]),1).to(device)
                else:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                model.zero_grad()        

                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            return_dict=True)

                loss = result.loss
                logits = result.logits

                batch_train_loss += loss.mean()

                loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad() 

                scheduler.step()
                
                batch_predictions = np.argmax(nn.Softmax(dim=1)(logits).detach().cpu().numpy(), axis=-1)
                correct_preds += (batch_predictions == b_labels.detach().cpu().numpy()).sum()
                total_predictions += b_labels.shape[0]
                    
                description = ("Average training loss: %.2f Accuracy: %.2f  Lable sum: %2f"
                                % (batch_train_loss/(step+1), correct_preds/total_predictions,batch_predictions.sum()))
                generator_tqdm.set_description(description, refresh=False)
                
            train_loss = batch_train_loss / len(train_dataloader)            
            
            training_time = format_time(time.time() - t0)
            
            t0 = time.time()

            model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            
            for batch in tqdm(validation_dataloader):
                if(prefix_tuning):
                    b_input_ids = torch.cat((torch.arange(0, config.prefix_length).expand(batch[0].shape[0], config.prefix_length),batch[0]),1).to(device)

                    b_input_mask = torch.cat((torch.ones(config.prefix_length).expand(batch[1].shape[0], config.prefix_length),batch[1]),1).to(device)
                else:
                    b_input_ids=batch[0].to(device)
                    b_input_mask=batch[1].to(device)
                b_labels = batch[2].to(device)
                with torch.no_grad():
                    result = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                return_dict=True)

                loss = result.loss
                logits = result.logits
                    
                total_eval_loss += loss.mean()
                
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += compute_accuracy(logits, label_ids)
                
                

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            validation_time = format_time(time.time() - t0)

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            
        print("Training Time:".format(format_time(time.time()-total_t0)))

        if(not prepare_data):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        if(save_model):
            output_dir = model_save_directory

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("Saving model to %s" % output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            pd.set_option('precision', 2)
            df_stats = pd.DataFrame(data=training_stats)
            df_stats = df_stats.set_index('epoch')
            df_stats.to_csv(output_dir+"/perfomance_stats.csv",index=False)
            print("Models Saved")
        
    else:
        if(prefix_tuning):
            output_dir = model_save_directory
            config = BertConfig.from_pretrained(output_dir,)
            model = SARCBertClassifier.from_pretrained(output_dir)

    if(not prepare_data):
        test_set = pd.read_csv(test_file)
    
    print("Started Testing")
    
    sentences = test_set.dropna(subset=["comment"]).comment.values
    labels = test_set.dropna(subset=["comment"]).label.values

    test_inputs_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True,)
        test_inputs_ids.append(encoded_sent)

    test_inputs_ids = pad_sequences(test_inputs_ids, maxlen=64, 
                            dtype="long", truncating="post", padding="post")

    test_attention_masks = []

    for seq in tqdm(test_inputs_ids):
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask) 

    prediction_inputs = torch.tensor(test_inputs_ids)
    prediction_masks = torch.tensor(test_attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 32  

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()

    predictions , true_labels = [], []

    for batch in tqdm(prediction_dataloader):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch
        if(prefix_tuning):
            b_input_ids = torch.cat((torch.arange(0, 5).expand(b_input_ids.shape[0], 5).to(device),b_input_ids),1).to(device)
            b_input_mask = torch.cat((torch.ones(5).expand(b_input_mask.shape[0], 5).to(device),b_input_mask),1).to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels,return_dict=True)

        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        
    
    preds=[]
    actuals=[]
    for i in range(len(true_labels)):
        preds.append(list(np.argmax(predictions[i], axis=1).flatten()))
        actuals.append(list(true_labels[i]))

    preds = list(itertools.chain(*preds))    
    actuals = list(itertools.chain(*actuals))

    test_metrics = {}
    test_metrics['accuracy_score'] = accuracy_score(actuals,preds)*100
    test_metrics['f1_score'] = f1_score(actuals, preds, average='macro')

    confm = confusion_matrix(actuals, preds,normalize='true')
    columns = ["Sarcastic","Normal"]
    df_cm = DataFrame(confm, index=columns, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)

    test_metrics["confusion_matrix"] = df_cm.to_dict('list')
    
    # if(save_processed_data):
    with open(output_dir+'/test_metrics.json', 'w') as fp:
        json.dump(test_metrics, fp)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Dependency Parsing Model')
    # General training arguments

    parser.add_argument('--prepare_data', action="store_true", default=False,
                        help='if passed, will prepare data.')
    
    parser.add_argument('--save_processed_data', action="store_true", default=False,
                        help='if passed, save the processed data.')
    parser.add_argument('--batch_size', type=int, help='batch_size ', default=128)
    
    parser.add_argument('--custom', action="store_true", default=True,
                        help='if passed, use no custom.')
    
    parser.add_argument('--epochs', type=int, help='epochs ', default=4)
    
    parser.add_argument('--learning_rate', type=float, help='learning_rate ', default=0.005)
    
    parser.add_argument('--save_model', action="store_true", default=True,
                        help='if passed, save model.')
    
    parser.add_argument('--tuning_mode', type=str, choices=("light_weight", "fine_tune"),
                        help='tuning_mode', default="light_weight")

    parser.add_argument('--model_save_directory', type=str,
                        help='tuning_mode', default="temper")

    parser.add_argument("--experiment_type", type=str,
                        help='Name of the experiment', default="prefix_random_initializaition")

    parser.add_argument("--use_multi_gpu",type=bool,help="Use Multiple GPUs",default=False)
    
    parser.add_argument("--phrase_for_init",type=str,help="If using custom initialization this will be used to initialize the prefix tokens",default=False)
    
    parser.add_argument("--checkpoint",type=str,help="to checkpoint the model at each epoch",default=True)
    
    parser.add_argument("--analyze_tokens",type=bool,help="Closest words in bert vocab in each epoch are extracted",default=False)
    
    parser.add_argument("--test_file",type=str,help="test file that have to be used",default="test.csv")
    
    parser.add_argument("--train_model",type=bool,help="True to train the model",default=True)

    args = parser.parse_args()

    main(args)
