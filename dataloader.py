import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import pickle, pandas as pd
import json

"""
数据存储：
Json文件：bug_data,other_data
格式：{
    "ids": [] #列表存放所有对话id
    "dialog": [[],[],....] #列表存放所有对话内容，每一个对话用列表存储所有句子
    "role": [[],[],....] #列表存放对话中每一个句子的角色：Questioner:0和Answer:1
    "dialog_label":[]  #列表存放对话的标签：other:0，bug:1
    "sen_label":[[],[],....] #列表存放对话中每一个句子的标签
    "graph_edge":[[],[],....] #列表存放每一个对话中句子的连接关系，[1,3]表示第一个句子与第三个句子相连
}    

assert len(ids) == len(dialog) == len(role) == len(dialog_label) == len(sen_label) == len(graph_edge)
"""


class dialogDataset(Dataset):
    def __init__(self, file_name, tokenizer_address='bert-base-uncased', max_length=40):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_address)

        with open(file_name, 'r') as r:
            data_dict = json.load(r)

        self.ids = data_dict['ids']
        self.max_length = max_length
        self.dialog = data_dict['dialog']
        self.role = data_dict['role']
        self.dialog_label = data_dict['dialog_label']
        self.sen_label = data_dict['sen_label']
        self.graph_edge = data_dict['graph_edge']
        self.len = len(self.ids)

    def __getitem__(self, index):

        # self.tokenized_dialog在映射id时已经返回为tensor了
        return self.dialog[index], \
               torch.tensor([1] * len(self.sen_label[index])), \
               torch.tensor(self.sen_label[index]), \
               torch.tensor(self.dialog_label[index]), \
               self.role[index], \
               self.graph_edge[index], \
               self.ids[index]

    def __len__(self):
        return self.len

    def tokenize_in_minibatch(self, dialog_list):
        # 对文本进行id的映射
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_attention_mask = []

        max_len = 0
        for dialog in dialog_list:
            for sen in dialog:
                max_len = max(len(sen), max_len)
        max_length = min(max_len, self.max_length)
        for dialog in dialog_list:
            tokenized_input = self.tokenizer(dialog, is_split_into_words=True, padding="max_length",
                                             truncation=True, return_tensors="pt", max_length=max_length + 2)
            dialog_input_ids.append(tokenized_input['input_ids'])
            dialog_token_type_ids.append(tokenized_input['token_type_ids'])
            dialog_attention_mask.append(tokenized_input['attention_mask'])

        assert len(dialog_list) == len(dialog_input_ids) == len(dialog_token_type_ids) == len(dialog_attention_mask)
        return [dialog_input_ids, dialog_token_type_ids, dialog_attention_mask]

    # dataloader自定义padding操作
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i == 0:
                for token in self.tokenize_in_minibatch(dat[i].tolist()):
                    return_list.append(pad_sequence(token, True))
            elif i < 3:
                return_list.append(pad_sequence(dat[i].tolist(), True))
            elif i < 4:
                return_list.append(torch.tensor(dat[i]))
            else:
                return_list.append(dat[i].tolist())

        return return_list