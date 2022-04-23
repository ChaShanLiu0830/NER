import torch.nn as nn
import torch
import copy
from transformers import AutoConfig, AutoModel
class NER(nn.Module):
    def __init__(self, pretrain_name):
        super().__init__()
        # model = EncoderDecoderModel.from_pretrained("t5-base")
        config = AutoConfig.from_pretrained(pretrain_name)

        # config = AutoConfig.from_pretrained(pretrain_name)
        self.encoder = AutoModel.from_pretrained( pretrain_name, config = config )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9) #number of class
            ) # for BI) tag
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, text):
        output = self.encoder(text)
        # print(output.last_hidden_state.shape)
        output = self.classifier(output.last_hidden_state)
        # print(output.shape)
        output = self.softmax(output)
        return output

if __name__ == '__main__':
    test_data = torch.randint(0,200, (1,20))
    test_model = NER('bert-base-uncased')
    test_output = test_model(test_data)
    print(test_output.shape)