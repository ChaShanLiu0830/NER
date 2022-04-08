from make_label import ttv_dataset
from torch.utils.data import Dataset, DataLoader, Subset
import random
class Conll03_DS(Dataset):
    def __init__(self, data, label, tokenizer, max_len):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_encode = self.tokenizer.encode(self.data[index], add_special_tokens = True, padding='max_length', max_length = self.max_len, truncation=True, return_tensors = 'pt')
        label_return = self.label[index]
        if len(label_return) < self.max_len: #make truncation 
             label_return += [0]*(self.max_len - len(label_return) )
        else:
            label_return = label_return[:self.max_len]
        return {'Data':data_encode, 'Label':label_return}
    
def load_dataset(dataset_fun, tokenizer, tt, batch_size, max_len,split_size = 1):
    
    TT_data, TT_label = ttv_dataset(tokenizer, tt) #train or test dataset
    Val_data, Val_label = ttv_dataset(tokenizer, 'valid')
    TT_DS = dataset_fun(TT_data, TT_label, tokenizer, max_len)
    Val_DS = dataset_fun(Val_data, Val_label, tokenizer, max_len)
    if split_size != 1 and tt == 'train': # make subset
        val_index = random.sample(range(0, len(Val_DS)), int(split_size*len(Val_DS)))
        tt_index =  random.sample(range(0, len(TT_DS)), int(split_size*len(TT_DS)))
        Val_DS = Subset(Val_DS, indices= val_index)
        TT_DS = Subset(TT_DS, indices= tt_index)
    
    TT_DL = DataLoader(TT_DS, batch_size = batch_size)
    Val_DL = DataLoader(Val_DS, batch_size = batch_size)

    return TT_DL, Val_DL

if __name__ == '__main__':
    from transformers import BertTokenizerFast
    from make_label import ttv_dataset
    import torch
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer.add_tokens(['O', 'B-MISC', 'B-LOC'],special_tokens = True)
    # print(tokenizer.encode(['O', 'B-MISC', 'B-LOC']))
    test_dl, vv_dl = load_dataset(Conll03_DS, tokenizer, 'train', 5, 200, 0.1)
    # print(test_ds[0])
    