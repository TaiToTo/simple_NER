import os
import pandas as pd
import itertools
import numpy as np
import torch

label_idx_dict = {'O': 0, 
    'SUPP_N': 1, # Supplier name
    'SUPP_G': 2, # Supplier GST
    'BUY_N': 3, # Buyers name
    'BUY_G': 4, # Buyers GST
    'GSTL': 5, # GST label
    'INV_NO': 6, # Invoice nubmer
    'INV_L': 7, # Invoice nubmer label
    'INV_DL': 8, # Invoice data label
    'INV_DT': 9, # Invoice data
    'GT_AMTL': 10, # Grand total amount label
    'GT_AMT': 11,
    'P': 12} # Grand total amount


def split_tokenize_label_dataset(split_path, tokenizer, overlap = 20):

    def conver_tag_to_id(tag):
        if tag=='O':
            return 0

        try:
            return label_idx_dict[str(tag).split('-')[1]]

        except:
            return 0




    with open(split_path, "r", encoding="utf-8") as f:
        file_path_list = f.read().splitlines()

    final_split_and_tokenized_file_list = []
    final_split_and_tokenized_labels = []
    final_split_word_id_list= []
    final_split_token_ids_list = []
    for cnt, file_name in enumerate(file_path_list):
        file_path = os.path.join('data', file_name)
        try:
            temp_df = pd.read_csv(file_path, encoding='unicode_escape')
        except UnicodeDecodeError:
            temp_df = pd.read_csv(file_path)
        
        labeled_token_list = [str(token) for token in temp_df[temp_df.columns[0]]] # Some data are float in the original file

        tag_list = [tag for tag in list(temp_df[temp_df.columns[1]])]
        label_list = [conver_tag_to_id(elem) for elem in tag_list]

        split_and_tokenized_file_list, split_and_tokenized_labels, split_word_id_list, split_token_ids_list = split_tokenize_label_file(labeled_token_list, label_list, tokenizer, overlap)
        final_split_and_tokenized_file_list.append(split_and_tokenized_file_list)
        final_split_and_tokenized_labels.append(split_and_tokenized_labels)
        final_split_word_id_list.append(split_word_id_list)
        final_split_token_ids_list.append(split_token_ids_list)
        del temp_df

     
    final_split_and_tokenized_file_list = list(itertools.chain.from_iterable(final_split_and_tokenized_file_list))
    final_split_and_tokenized_labels = list(itertools.chain.from_iterable(final_split_and_tokenized_labels))
    final_split_word_id_list = list(itertools.chain.from_iterable(final_split_word_id_list))
    final_split_token_ids_list = list(itertools.chain.from_iterable(final_split_token_ids_list))
    
    
    return final_split_and_tokenized_file_list, final_split_and_tokenized_labels, final_split_word_id_list, final_split_token_ids_list



def split_tokenize_label_file(labeled_token_list, label_list, tokenizer, overlap, max_bert_input_length=128):
    
    word_id_list = []
    bert_token_list = []
    bert_token_ids_list = []
    for word_id_cnt, labeled_token in enumerate(labeled_token_list):
        bert_tokens = tokenizer.tokenize(labeled_token.lower())
        bert_token_ids = tokenizer(labeled_token.lower())['input_ids'][1:-1]
        bert_token_ids_list.append(bert_token_ids)
        word_id_list.append([word_id_cnt]*len(bert_tokens))
        bert_token_list.append(bert_tokens)
        
    
    word_id = np.concatenate(word_id_list)
    bert_tokens = np.concatenate(bert_token_list)
    bert_token_ids = np.concatenate(bert_token_ids_list)
    
    start = 0
    end = len(bert_tokens)
    
    split_and_tokenized_file_list = []
    split_and_tokenized_labels = []
    split_word_id_list = []
    split_token_ids_list = []

    for i in range(start, end, max_bert_input_length-overlap):
        split_and_tokenized_file_list.append(bert_tokens[i: (i + max_bert_input_length)])
        temp = word_id[i: (i + max_bert_input_length)]
        split_word_id_list.append(temp)
        split_token_ids_list.append(bert_token_ids[i: (i + max_bert_input_length)])
        split_and_tokenized_labels.append(np.array(label_list)[temp])
        
    return split_and_tokenized_file_list, split_and_tokenized_labels, split_word_id_list, split_token_ids_list

class form_input():
    def __init__(self, bert_tokenized_sentence_list, bert_tokenized_label_list, 
                 bert_tokenized_id_list, bert_tokenized_word_id_list, config, data_type='test'):
        assert len(bert_tokenized_sentence_list)==len(bert_tokenized_label_list), \
                    print("The numbers of split texts and corresponding labels are supposed to be the same")
        
        self.bert_tokenized_sentence_list = bert_tokenized_sentence_list
        self.bert_tokenized_label_list = bert_tokenized_label_list
        self.bert_tokenized_id_list = bert_tokenized_id_list
        self.bert_tokenized_word_id_list = bert_tokenized_word_id_list
        
        self.max_length = config['MAX_LEN']
        #self.tokenizer = config['tokenizer']
        self.data_type = data_type
    
    def __len__(self):
        return len(self.bert_tokenized_sentence_list)
    
    def __getitem__(self, item):
        bert_tokens = self.bert_tokenized_sentence_list[item]
        bert_labels = self.bert_tokenized_label_list[item]
        bert_token_ids = self.bert_tokenized_id_list[item]
        bert_tokenized_word_ids = self.bert_tokenized_word_id_list[item]
        
        ########################################
        # Forming the inputs
        #bert_token_ids = config['tokenizer'].convert_tokens_to_ids(toks)
        tok_type_id = np.zeros( len(bert_tokenized_word_ids))
        att_mask = np.ones(len(bert_tokenized_word_ids))
        
        # Padding
        pad_len = self.max_length - len(bert_tokenized_word_ids) 
        if(pad_len!=0):
            bert_token_ids = np.concatenate([bert_token_ids, 12*np.ones(pad_len)], 0)
            tok_type_id = np.concatenate([tok_type_id, np.zeros(pad_len)], 0) 
            att_mask = np.concatenate([att_mask, np.zeros(pad_len)], 0)  
            bert_labels = np.concatenate([bert_labels, 12*np.ones(pad_len)], 0)


            
        
        if self.data_type=='test':
            return {'ids': torch.tensor(bert_token_ids, dtype = torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'target': torch.tensor(bert_labels, dtype = torch.long), 
                'bert_tokens': bert_tokens, 
                'bert_token_word_ids': bert_tokenized_word_ids }
        else:

            pass
            return {'ids': torch.tensor(bert_token_ids, dtype = torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'target': torch.tensor(bert_labels, dtype = torch.long)
               }