import os
import pandas as pd
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
import torch.nn as nn
from tqdm import tqdm


from prepare_invoice_ner_dataset import label_idx_dict
from prepare_invoice_ner_dataset import split_tokenize_label_dataset
from prepare_invoice_ner_dataset import split_tokenize_label_file
from prepare_invoice_ner_dataset import form_input
from print_ner_tag import print_ner_labels
from print_ner_tag import print_ner_labels_detokenized



def print_NER_on_invoice(model, prod_input):
    invoice_number = 1

    if_estimation = True
    estimation_list = []
    gt_list = []
    print("Sample Invoice {}\n".format(invoice_number))
    for sample_idx in range(len(prod_input)):

        dataset = prod_input[sample_idx]

        batch_input_ids = torch.unsqueeze(dataset['ids'], 0).to(config['device'], dtype=torch.long)
        batch_att_mask = torch.unsqueeze(dataset['att_mask'], 0).to(config['device'], dtype=torch.long)
        batch_tok_type_id = torch.unsqueeze(dataset['tok_type_id'], 0).to(config['device'], dtype=torch.long)
        batch_target = torch.unsqueeze(dataset['target'], 0).to(config['device'], dtype=torch.long)

        bert_tokens = dataset['bert_tokens']
        bert_token_word_ids = dataset['bert_token_word_ids']

        output = model(batch_input_ids,
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       # labels=batch_target
                       )
        sample_ner_result = output['logits'].detach().numpy()[0]
        sample_estimated_tags = np.argmax(sample_ner_result, -1)
        sample_tag_labels = prod_input[sample_idx]['target'].numpy()

        sample_estimated_tag_labels = np.array(
            [list(label_idx_dict.keys())[label_idx] for label_idx in list(sample_estimated_tags)])
        sample_ground_truth_tag_labels = np.array(
            [list(label_idx_dict.keys())[label_idx] for label_idx in list(sample_tag_labels)])

        estimation_list.append((bert_tokens, sample_estimated_tag_labels[:len(bert_tokens)], bert_token_word_ids))
        gt_list.append((bert_tokens, sample_ground_truth_tag_labels[:len(bert_tokens)], bert_token_word_ids))

        if (len(bert_tokens) != 128):
            print("Estimation")
            for estimation_tuple in estimation_list:
                print_ner_labels_detokenized(estimation_tuple[0], estimation_tuple[1], estimation_tuple[2])
            print()
            print()
            print("Ground truth")
            for gt_tuple in gt_list:
                print_ner_labels_detokenized(gt_tuple[0], gt_tuple[1], gt_tuple[2])
            print()
            print()
            print()

            gt_list = []
            estimation_list = []
            invoice_number += 1

            # if (invoice_number == 6):
            #     break

            print("Sample Invoice {}".format(invoice_number))



if __name__ == "__main__":
    test_split_path = 'split/test.txt'

    # bert_path = '../kaggle_ner/huggingface-bert/bert-base-uncased/'
    model_checkpoint = "bert-base-cased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)

    config = {'MAX_LEN': 128,
              'tokenizer': tokenizer,
              'batch_size': 32,
              'Epoch': 1,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'model_name': 'model1_bert_base_uncased_3_epochs.bin'
              }

    final_test_split_and_tokenized_file_list, final_test_split_and_tokenized_labels, final_test_split_word_id_list, final_test_split_token_ids_list = split_tokenize_label_dataset(
        test_split_path, tokenizer, overlap=0)

    test_prod_input = form_input(final_test_split_and_tokenized_file_list, final_test_split_and_tokenized_labels,
                                 final_test_split_token_ids_list, final_test_split_word_id_list, config,
                                 data_type='test')

    model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_idx_dict))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config['model_name']))
    model.eval()

    print_NER_on_invoice(model, test_prod_input)

