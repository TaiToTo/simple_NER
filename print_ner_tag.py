from color_print import Color
import numpy as np
import scipy.stats as stats


label_color_dict = {'O': Color.BLACK, 
    'SUPP_N': Color.RED, # Supplier name
    'SUPP_G': Color.GREEN, # Supplier GST
    'BUY_N': Color.YELLOW, # Buyers name
    'BUY_G': Color.BLUE, # Buyers GST
    'GSTL': Color.MAGENTA, # GST label
    'INV_NO': Color.CYAN, # Invoice nubmer
    'INV_L': Color.BG_RED, # Invoice nubmer label
    'INV_DL': Color.BG_GREEN, # Invoice data label
    'INV_DT': Color.BG_YELLOW, # Invoice data
    'GT_AMTL': Color.BG_BLUE, # Grand total amount label
    'GT_AMT': Color.BG_MAGENTA} # Grand total amount

def print_ner_labels(token_list, label_list):
    assert len(token_list)==len(label_list), print(" ")
    
    for (token, label) in zip(token_list, label_list):
    #    print("token: {}, label: {}".format(token, label))
        print(label_color_dict[label]+token+ Color.RESET, end=' ')
        
        
def print_ner_labels_detokenized(token_list, label_list, bert_token_word_ids):
    assert len(token_list)==len(label_list), print(" ")
    token_list_cleaned = np.array([token.replace('##', '') for token in token_list])
    
    detokenized_token_list = []
    detokenized_label_list = []
    
    for word_id in range(bert_token_word_ids.min(), bert_token_word_ids.max() + 1):
        indexes_to_detokenize = bert_token_word_ids==word_id
        tokens_temp = token_list_cleaned[indexes_to_detokenize]
        label_temp = label_list[indexes_to_detokenize]
        mode_val, mode_num = stats.mode(label_temp)

        detokenized_token_list.append(''.join(tokens_temp) )
        detokenized_label_list.append(mode_val[0])

    for (token, label) in zip(detokenized_token_list, detokenized_label_list):
        print(label_color_dict[label]+token+ Color.RESET, end=' ')