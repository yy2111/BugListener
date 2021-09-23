import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math
from transformers import BertTokenizer, BertModel



# 计算边的权重
class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.no_cuda = no_cuda

    def forward(self, M, edge_ind):
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=1).permute(0, 2, 1)
        if not self.no_cuda:
            mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
            mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()

        else:
            mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
            mask_copy = Variable(torch.zeros(alpha.size())).detach()

        edge_ind_ = []
        # i是batch号，
        for i, j in enumerate(edge_ind):
            # 每一个dialog中边的连边
            for x in j:
                edge_ind_.append([i, x[0], x[1]])

        # edge_ind_ -> 3 * all_num_edges
        edge_ind_ = np.array(edge_ind_).transpose()
        mask[edge_ind_] = 1
        mask_copy[edge_ind_] = 1
        masked_alpha = alpha * mask
        _sums = masked_alpha.sum(-1, keepdim=True)
        scores = masked_alpha.div(_sums) * mask_copy

        return scores


def edge_perms(l, window_past=-1, window_future=-1):
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def batch_graphify(features, role, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    some code is borrowed from https://github.com/declare-lab/conv-emotion
    """
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    # features -> batch_size * seq_num * embedd_dim
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    # 每一个对话都构建一个边的关系矩阵
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    # edge_ind -> batch_size * edges_num
    # scores are the edge weights
    scores = att_model(features, edge_ind)

    for j in range(batch_size):
        # features -> batch_size *seq_num  * embedd_dim
        # node_feature的作用是去掉padding的句子
        node_features.append(features[j, :lengths[j], :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        # 将一个batch的图构成一个完整的图
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        # 记录每个dialog的边数
        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])

            speaker0 = role[j][item1[0]]
            speaker1 = role[j][item1[1]]

            if item1[0] < item1[1]:
                # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    # node_features -> all_seq_num * embedd_dim
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    # if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    """
    node_features:节点特征，即句子embedding
    edge_index:一个batch中所有的对话构成一个图，存放边的索引
    edge_norm：存放边的权重
    edge_type：存放边的类型：总共8种类型
    edge_index_lengths：存放batch中每个对话的边数
    """

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def graph_classification(features, seq_lengths, linear_layer, dropout_layer, smax_fc_layer, avec):
    cur_num = 0
    node_sum = None
    max_pool_node = None
    for index in range(len(seq_lengths)):
        node_embedd = torch.sum(features[cur_num:seq_lengths[index] + cur_num, :], dim=0).unsqueeze(0)
        max_node_embedd = torch.max(features[cur_num:seq_lengths[index] + cur_num, :], dim=0)[0].unsqueeze(0)
        cur_num += seq_lengths[index]
        if node_sum is None:
            node_sum = node_embedd
        else:
            node_sum = torch.cat((node_sum, node_embedd), 0)
        if max_pool_node is None:
            max_pool_node = max_node_embedd
        else:
            max_pool_node = torch.cat((max_pool_node, max_node_embedd), 0)

    hidden = F.relu(linear_layer(torch.cat([node_sum, max_pool_node], dim=-1)))
    #hidden = F.relu(linear_layer(node_sum))
    hidden = dropout_layer(hidden)
    hidden = smax_fc_layer(hidden)

    if avec:
        return hidden

    graph_log_prob = F.log_softmax(hidden, -1)
    # graph_log_prob -> batch * num_class
    return graph_log_prob


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, graph_class_num, num_relations, hidden_size=64, dropout=0.5,
                 no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()

        self.conv1 = GraphConv(num_features, hidden_size)
        self.conv2 = RGCNConv(hidden_size, hidden_size, num_relations, num_bases=30)

        # self.node_linear = nn.Linear(num_features + hidden_size, hidden_size)
        self.graph_linear = nn.Linear(2*(num_features + hidden_size), hidden_size)
        self.dropout = nn.Dropout(dropout)
        # self.node_smax_fc = nn.Linear(hidden_size, node_class_num)
        self.graph_smax_fc = nn.Linear(hidden_size, graph_class_num)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, avec):
        out = self.conv1(x, edge_index, edge_norm)
        out = self.conv2(out, edge_index, edge_type)

        # 将结点的初始向量和经过图转移的向量拼接
        features = torch.cat([x, out], dim=-1)
        # 图预测
        graph_log_prob = graph_classification(features, seq_lengths, self.graph_linear, self.dropout,
                                              self.graph_smax_fc, avec)

        return graph_log_prob


class BugListener(nn.Module):

    def __init__(self, pretrained_model, D_bert, D_cnn, graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future, graph_class_num=2, dropout=0.5, avec=False, no_cuda=False):

        super(BugListener, self).__init__()

        self.pretrained_bert = BertModel.from_pretrained(pretrained_model)
        # mutil-task的参数调整
        # self.sigma = nn.Parameter(torch.ones(2))
        self.filter_sizes = [2,3,4,5]
        self.sen_encoder = nn.ModuleList([nn.Conv1d(D_bert, D_cnn, size) for size in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.filter_sizes) * D_cnn, 100)
        self.avec = avec
        self.no_cuda = no_cuda

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        # 计算边的权重
        self.att_model = MaskedEdgeAttention(100, max_seq_len, self.no_cuda)

        self.graph_net = GraphNetwork(100, graph_class_num, n_relations,
                                      graph_hidden_size, dropout, self.no_cuda)

        # 边的关系：edge_type_mapping['000']=0, edge_type_mapping['001']=1
        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

    def forward(self, input_ids, token_type_ids, attention_mask, role_id, seq_lengths, umask):
         # 预训练的BERT对Dialog编码
        # input_ids -> batch * seq_num * word_num
        b_, s_, w_ = input_ids.size()
        # -> (batch*sen_num) * word_num
        i_ids = input_ids.view(-1, w_)
        t_ids = token_type_ids.view(-1, w_)
        a_ids = attention_mask.view(-1, w_)

        # word_output = (batch*sen_num) * word_num * D_bert
        word_output = self.pretrained_bert(input_ids=i_ids, token_type_ids=t_ids,
                                           attention_mask=a_ids)[0]

        # -> (batch*sen_num) * dim * word_num
        word_output = word_output.transpose(-2,-1).contiguous()

        convoluted = [F.relu(conv(word_output)) for conv in self.sen_encoder]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(b_, s_, -1)  # (num_utt * batch, 100) -> (batch, num_utt, 100)
        mask = umask.unsqueeze(-1).type(torch.FloatTensor).detach().cuda()  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.repeat(1, 1, 100)  # (batch, num_utt, 1) -> (batch, num_utt, 100)
        features = (features * mask)  # (batch, num_utt, 100) -> (batch, num_utt, 100)


        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(features, role_id,
                                                                                        seq_lengths,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model, self.no_cuda)

        # features(node_features) ->  all_seq_num * embedd_dim
        graph_log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, self.avec)

        return graph_log_prob, edge_index, edge_norm, edge_type, edge_index_lengths