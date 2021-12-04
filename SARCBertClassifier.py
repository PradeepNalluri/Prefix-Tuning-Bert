import transformers
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from transformers import BertTokenizer,BertForSequenceClassification,BertModel

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class SARCBertClassifier(BertForSequenceClassification):
    """
    Classifier to handle classification task on SARC dataset
    """
    def __init__(self,config):
        super(SARCBertClassifier, self).__init__(config)
#         self.mlp_layer=None
#         self.prefix_embeddings =None
        self.run_device = None
    def update_network_sarc(self,num_layers,device,freeze_bert_layers=False,custom_embedding=False,custom_embedding_vector=None,add_user_information=False):
        """
        Update the network architecture all the variable are class variables from source code of BerforSequenceClassification
        transformer module
        """
        config=self.config
        if(freeze_bert_layers):
            for name,param in self.bert.named_parameters():
                if(name!="embeddings.prefix_embeddings.weight"):
                    param.requires_grad = False
        self.prefix_embeddings = nn.Embedding(config.prefix_length, config.hidden_size)
        self.prefix_length = config.prefix_length
        self.mlp_layer = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size,config.hidden_size))
        
        if(add_user_information):
            self.classifier = nn.Linear(config.hidden_size+2, config.num_labels)
        
        if(custom_embedding):
            self.prefix_length = config.prefix_length
            self.mlp_layer = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Linear(config.hidden_size,config.hidden_size))

            self.init_weights()
            
            custom_embedding_vector = custom_embedding_vector.expand(config.prefix_length,custom_embedding_vector.shape[0])
            self.prefix_embeddings=nn.Embedding.from_pretrained(custom_embedding_vector)
        else:
            self.prefix_embeddings = nn.Embedding(config.prefix_length, config.hidden_size)
            self.prefix_length = config.prefix_length
            self.mlp_layer = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Linear(config.hidden_size,config.hidden_size))
            self.init_weights()
        self.run_device = device
    
    def check_closest_matching_bert_model(self):
        prefix_tokens = self.prefix_embeddings(torch.LongTensor(torch.arange(1)).to(self.run_device)).detach()
        bert_base = self.bert.embeddings.word_embeddings(torch.LongTensor(torch.arange(30522)).to(self.run_device)).detach()
        closest_words_ids = [] 
        for embd in prefix_tokens:
            closest_words_ids.append(torch.norm(bert_base - embd.unsqueeze(0), dim=1).topk(5).indices)
        tokenizer = BertTokenizer.from_pretrained("./prefix_tuning_model_random_initializations_prefix_tuninglr_2e-5/")
        closest_words_ids=torch.stack(closest_words_ids)
        closest = {}
        for idx,t in enumerate(closest_words_ids):
            word_l = []
            for tok in t:
                word_l.append(tokenizer._convert_id_to_token(int(tok)))
            closest[idx]=word_l
        return closest
    
    def closest_matching_bert_model(self):
        prefix_tokens = self.prefix_embeddings(torch.LongTensor(torch.arange(self.prefix_embeddings.weight.shape[0])).to(self.run_device)).detach()
        bert_base = self.bert.embeddings.word_embeddings(torch.LongTensor(torch.arange(30522)).to(self.run_device)).detach()
        closest_words_ids = [] 
        for embd in prefix_tokens:
            closest_words_ids.append(torch.norm(bert_base - embd.unsqueeze(0), dim=1).topk(5).indices)
        tokenizer = BertTokenizer.from_pretrained("./prefix_tuning_model_random_initializations_prefix_tuninglr_2e-5/")
        closest_words_ids=torch.stack(closest_words_ids)
        closest = {}
        for idx,t in enumerate(closest_words_ids):
            word_l = []
            for tok in t:
                word_l.append(tokenizer._convert_id_to_token(int(tok)))
            closest[idx]=word_l
        return closest
    
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,
        head_mask=None,inputs_embeds=None,labels=None,output_attentions=None,output_hidden_states=None,return_dict=None,
                user_information=False):
        r"""

        FROM CORE HUGGINGFACE MODULE 
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        prefix_embds = self.prefix_embeddings(torch.arange(0, self.prefix_length).to(self.run_device))
        prefix_embds = self.mlp_layer(prefix_embds)
        prefix_embds = prefix_embds.expand(len(input_ids),prefix_embds.shape[0],prefix_embds.shape[1])
        attention_mask = torch.cat((torch.ones(self.prefix_length).to(self.run_device).expand(attention_mask.shape[0],self.prefix_length),attention_mask),1)
        
#         if(user_information):
#             attention_mask = attention_mask[:,2:] 
#             user_ids =  input_ids[:,:2]
#             input_ids = input_ids[:,2:].to(self.run_device).long()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefix_embeddings=prefix_embds,
        )

        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        
#         if(user_information):
#             pooled_output = torch.cat((user_ids,pooled_output),dim=1)
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )