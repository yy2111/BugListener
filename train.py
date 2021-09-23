import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.data import DataLoader

from FocalLoss import FocalLoss
from dataloader import dialogDataset
from model import BugListener


# def multiTask_loss(loss1,loss2,sigma1,sigma2,alpha):
#     total_loss = alpha * loss1 /sigma1**2 + (1-alpha) * loss2 /sigma2**2 + torch.log(sigma1*sigma2)

#     return total_loss,sigma1,sigma2

def flooding_loss(loss, beta):
    return (loss - beta).abs() + beta


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Config(object):
    def __init__(self):
        self.cuda = True
        self.project_name = "./final_data/appium"  # angular, appium, docker, dl4j, gitter, typescript
        self.pretrained_model = 'bert-base-uncased'
        self.windowp = -1
        self.windowf = -1
        self.dropout = 0.5
        self.batch_size = 16
        self.epochs = 60
        self.D_bert = 768
        self.D_cnn = 100
        self.D_graph = 64
        self.lr = 0.0001
        self.l2 = 0.00001
        self.graph_class_num = 2
        self.alpha = 0.5
        self.tensorboard = True


def get_dialog_loaders(project_name, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):
    trainset = dialogDataset(project_name + '_train.json', pretrained_model)

    test_set = dialogDataset(project_name + '_test.json', pretrained_model)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_or_eval_graph_model(model, graph_loss_function, dataloader, epoch, cuda, optimizer=None, train=False,
                              alpha=0.5):
    losses, node_preds, node_labels, graph_preds, graph_labels = [], [], [], [], []
    scores, vids = [], []
    false_positive = []
    false_negative = []
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # 将一个mini-batch的数据读取出来,并加载到GPU
        input_ids, token_type_ids, attention_mask_ids, umask, node_label, graph_label = \
            [d.cuda() for d in data[:-3]] if cuda else data[:-3]

        role_id = data[-3]
        vids += data[-1]
        # graph_edge = data[-2]
        # vid = data[-1]

        # 保存一个mini-barch中每个对话的句子数目
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # 模型的输入：1.对话文本，2.用户角色，3.和每个对话中句子数目等长的数组，4.每个对话的句子长度
        # qmask换成role_id， umask去掉
        graph_log_prob, e_i, e_n, e_t, e_l = \
            model(input_ids, token_type_ids, attention_mask_ids, role_id, lengths, umask)

        # 之前的label在dataloader中padding过，需要通过lengths还原
        node_label = torch.cat([node_label[j][:lengths[j]] for j in range(len(node_label))])

        loss = graph_loss_function(graph_log_prob, graph_label)

        # loss = alpha * node_loss + (1.0-alpha) * graph_loss

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l


        graph_preds.append(torch.argmax(graph_log_prob, 1).cpu().numpy())
        graph_labels.append(graph_label.cpu().numpy())

        losses.append(loss.item())

        if train:
            loss.backward()
            for name, params in model.named_parameters():
                if params.requires_grad:
                    writer.add_histogram(name, params.grad, epoch)
            optimizer.step()

    # 每一个epoch，将result拼接在一起
    if graph_preds != []:
        graph_preds = np.concatenate(graph_preds)
        graph_labels = np.concatenate(graph_labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    graph_labels = np.array(graph_labels)
    graph_preds = np.array(graph_preds)
    vids = np.array(vids)

    assert len(graph_labels) == len(graph_preds) == len(vids)

    for vid, target, pred in zip(vids, graph_labels, graph_preds):
        if target != pred and pred == 1:
            false_positive.append(vid)
        if target != pred and pred == 0:
            false_negative.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_graph_accuracy = round(accuracy_score(graph_labels, graph_preds) * 100, 2)
    graph_pre = round(precision_score(graph_labels, graph_preds) * 100, 2)
    graph_rec = round(recall_score(graph_labels, graph_preds) * 100, 2)
    graph_fscore = round(f1_score(graph_labels, graph_preds) * 100, 2)

    return avg_loss, avg_graph_accuracy, graph_pre, graph_rec, graph_fscore, false_positive, false_negative, \
           node_labels, graph_labels, ei, et, en, el


if __name__ == '__main__':

    config = Config()

    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if config.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    model = BugListener(config.pretrained_model,
                             config.D_bert, config.D_cnn,
                             config.D_graph,
                             n_speakers=2,
                             max_seq_len=35,
                             window_past=config.windowp,
                             window_future=config.windowf,
                             graph_class_num=config.graph_class_num,
                             dropout=config.dropout)

    # 模型装载至cuda
    if cuda:
        model.cuda()


    graph_loss = FocalLoss(gamma=2)

    # 冻结bert参数，训练时不更新
    for name, params in model.pretrained_bert.named_parameters():
        params.requires_grad = False

    print(get_parameter_number(model))

    # 过滤掉requires_grad = False的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.l2)

    train_loader, test_loader = get_dialog_loaders(config.project_name, config.pretrained_model,
                                                   batch_size=config.batch_size,
                                                   num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None

    all_graph_fscore, all_graph_acc, all_loss = [], [], []

    for e in range(config.epochs):
        start_time = time.time()

        optimizer = optimizer

        # 训练模型的主要组件：data_loader，model，optimizer，loss_function
        # train
        train_loss, train_graph_acc, _, _, train_graph_fscore, _, _, _, _, _, _, _, _ = \
            train_or_eval_graph_model(model, graph_loss, train_loader, e, cuda, optimizer, True, config.alpha)

        # test
        test_loss, test_graph_acc, test_graph_precision, test_graph_recall, test_graph_fscore, fp, fn, _, _, _, _, _, _ = \
            train_or_eval_graph_model(model, graph_loss, test_loader, e, cuda)



        # all_node_fscore.append(test_node_fscore)
        all_graph_fscore.append(test_graph_fscore)
        # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')

        if config.tensorboard:
            # writer.add_scalar('test: node_accuracy/loss', test_node_acc / test_loss, e)
            writer.add_scalar('test: graph_accuracy/loss', test_graph_acc / test_loss, e)
            # writer.add_scalar('train: node_accuracy/loss', train_node_acc / train_loss, e)
            writer.add_scalar('train: graph_accuracy/loss', train_graph_acc / train_loss, e)

        print(
            'epoch: {}, train_loss: {}, train_graph_acc: {}, train_graph_fscore: {}, test_loss: {}, test_graph_acc: {},test_graph_pre: {},test_graph_rec: {}, test_graph_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_graph_acc, train_graph_fscore,
                       test_loss, test_graph_acc, test_graph_precision, test_graph_recall, test_graph_fscore,
                       round(time.time() - start_time, 2)))
        print("false_postive", fp)
        print("false_negative", fn)
        print("==========================================================================")

    if config.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score=graph:', max(all_graph_fscore))