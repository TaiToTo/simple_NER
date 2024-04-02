import os
import pandas as pd
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

# from transformers import AutoTokenizer, BertTokenizer
import transformers


from prepare_invoice_ner_dataset import label_idx_dict
from prepare_invoice_ner_dataset import split_tokenize_label_dataset
from prepare_invoice_ner_dataset import split_tokenize_label_file
from prepare_invoice_ner_dataset import form_input


def train_fn(data_loader, model, optimizer):
    '''
    Functiont to train the model
    '''
    print("Training phase")
    train_loss = 0
    for index, dataset in enumerate(data_loader):
        batch_input_ids = dataset['ids'].to(config['device'], dtype=torch.long)
        batch_att_mask = dataset['att_mask'].to(config['device'], dtype=torch.long)
        batch_tok_type_id = dataset['tok_type_id'].to(config['device'], dtype=torch.long)
        batch_target = dataset['target'].to(config['device'], dtype=torch.long)

        output = model(batch_input_ids,
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       labels=batch_target)

        step_loss = output[0]
        prediction = output[1]

        if ((index + 1) % 10 == 0):
            print("Step {}, train loss {}".format(index + 1, step_loss))

        # print(prediction.shape)

        step_loss.sum().backward()
        optimizer.step()
        train_loss += step_loss
        optimizer.zero_grad()

    return train_loss.sum()


def eval_fn(data_loader, model):
    '''
    Functiont to evaluate the model on each epoch.
    We can also use Jaccard metric to see the performance on each epoch.
    '''

    model.eval()

    eval_loss = 0
    predictions = np.array([], dtype=np.int64).reshape(0, config['MAX_LEN'])
    true_labels = np.array([], dtype=np.int64).reshape(0, config['MAX_LEN'])
    print("Evaluation phase")
    with torch.no_grad():
        for index, dataset in enumerate(data_loader):
            batch_input_ids = dataset['ids'].to(config['device'], dtype=torch.long)
            batch_att_mask = dataset['att_mask'].to(config['device'], dtype=torch.long)
            batch_tok_type_id = dataset['tok_type_id'].to(config['device'], dtype=torch.long)
            batch_target = dataset['target'].to(config['device'], dtype=torch.long)

            output = model(batch_input_ids,
                           token_type_ids=None,
                           attention_mask=batch_att_mask,
                           labels=batch_target)

            step_loss = output[0]
            eval_prediction = output[1]

            if ((index + 1) % 10 == 0):
                print("Step {}, train loss {}".format(index + 1, step_loss))

            eval_loss += step_loss

            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis=2)
            actual = batch_target.to('cpu').numpy()

            predictions = np.concatenate((predictions, eval_prediction), axis=0)
            true_labels = np.concatenate((true_labels, actual), axis=0)

    return eval_loss.sum(), predictions, true_labels


def train_engine(epoch, train_data, valid_data):
    model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_idx_dict))
    model = nn.DataParallel(model)
    model = model.to(config['device'])

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=3e-5)

    best_eval_loss = 1000000
    for i in range(epoch):
        train_loss = train_fn(data_loader=train_data,
                              model=model,
                              optimizer=optimizer)
        eval_loss, eval_predictions, true_labels = eval_fn(data_loader=valid_data,
                                                           model=model)

        # print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss

            print("Saving the model")
            torch.save(model.state_dict(), config['model_name'])

    return model, eval_predictions, true_labels

if __name__ == "__main__":
    train_split_path = 'split/train.txt'
    val_split_path = 'split/val.txt'
    test_split_path = 'split/test.txt'

    with open(train_split_path, "r", encoding="utf-8") as f:
        train_file_path_list = f.read().splitlines()

    model_checkpoint = "bert-base-cased"
    train_split_path = './split/train.txt'
    val_split_path = './split/val.txt'
    test_split_path = './split/test.txt'

    with open(train_split_path, "r", encoding="utf-8") as f:
        train_file_path_list = f.read().splitlines()

    from transformers import AutoTokenizer, BertTokenizer

    model_checkpoint = "bert-base-cased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)

    config = {'MAX_LEN': 128,
              'tokenizer': tokenizer,
              'batch_size': 32,
              'Epoch': 3,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'model_name': 'model1_bert_base_uncased_3_epochs.bin'
              }

    final_train_split_and_tokenized_file_list, \
        final_train_split_and_tokenized_labels, \
        final_train_split_word_id_list, \
        final_train_split_token_ids_list = split_tokenize_label_dataset(train_split_path, tokenizer)

    final_val_split_and_tokenized_file_list, \
        final_val_split_and_tokenized_labels, \
        final_val_split_word_id_list, \
        final_val_split_token_ids_list = split_tokenize_label_dataset(val_split_path, tokenizer)

    train_prod_input = form_input(final_train_split_and_tokenized_file_list, final_train_split_and_tokenized_labels,
                                  final_train_split_token_ids_list, final_train_split_word_id_list, config,
                                  data_type='train')

    val_prod_input = form_input(final_val_split_and_tokenized_file_list, final_val_split_and_tokenized_labels,
                                final_val_split_token_ids_list, final_val_split_word_id_list, config, data_type='train')

    train_prod_input_data_loader = DataLoader(train_prod_input, batch_size=config['batch_size'], shuffle=True)
    val_prod_input_data_loader = DataLoader(val_prod_input, batch_size=config['batch_size'], shuffle=True)

    model, val_predictions, val_true_labels = train_engine(epoch=config['Epoch'],
                                                           train_data=train_prod_input_data_loader,
                                                           valid_data=val_prod_input_data_loader)