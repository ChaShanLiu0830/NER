
from ast import Pass


def get_clean_label(text): 
    if text.startswith('-DOCSTART-') or text.startswith('\n'):
        return False
    else:
        token, _, _, NER_tag = text.split()
        return token, NER_tag.replace('\n','')

def read_file(path):
    with open(path, 'r') as f:
        text = f.readlines()
        # print(text[0:8])
        return text

def make_label(text):
    sent = []
    sent_label = []
    token = ""
    label = []
    start_pos = 0
    end_pos = 0
    for word in text:
        if get_clean_label(word):
            word, ner_tag = get_clean_label(word)
            token += (word + " ")
            end_pos =  start_pos + len(word)
            inform = dict(start = start_pos, end = end_pos, ner_tag = ner_tag)
            label.append(inform)
            start_pos = end_pos+1
        else:
            sent.append(token) if token != '' else Pass
            sent_label.append(label) if label != [] else Pass
            token = ""
            label = []
            start_pos = 0
            end_pos = 0
    return sent, sent_label
    
# def make_tokenize_label(tokenizer, sent, sent_label):
    
def tokenize_label(sent, sent_label, tokenizer):
    token = tokenizer(sent)
    label = ['O']*len(token['input_ids'])
    # print(token)
    for each_tag in sent_label:
        for start2end in range(each_tag['start'], each_tag['end']):
            index = token.char_to_token(start2end)
            label[index] = each_tag['ner_tag']
    return label

if __name__ == '__main__':
    print(get_clean_label('-DOCSTART-'))
    test_sent, test_label = make_label(read_file('./dataset/test.txt'))
    print(test_sent[0:2], test_label[0:2])
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print(tokenize_label(test_sent[0], test_label[0], tokenizer))
    