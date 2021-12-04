from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import  CrossEntropyLoss
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW, BertConfig

class SARCBertClassifier(BertForSequenceClassification):
    """
    Classifier to handle classification task on SARC dataset
    """
    def __init__(self,config):
        super(SARCBertClassifier, self).__init__(config)
        
    def update_network_sarc(self,num_layers,device,freeze_bert_layers=False):
        """
        Update the network architecture all the variable are class variables from source code of BerforSequenceClassification
        transformer module
        """
        config=self.config
        if(freeze_bert_layers):
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential()
        for layer in range(num_layers-1):
            self.classifier.add_module("classification_layer_"+str(layer+1),nn.Linear(config.hidden_size, config.hidden_size))
            self.classifier.add_module("activation_layer_"+str(layer+1),nn.ReLU())
            
        self.classifier.add_module("output_layer",nn.Linear(config.hidden_size, config.num_labels))
        self.classifier.to(device)
        self.init_weights()
        
        def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,
            head_mask=None,inputs_embeds=None,labels=None,output_attentions=None,output_hidden_states=None,return_dict=None,):
            r"""

            FROM CORE HUGGINGFACE MODULE 
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
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