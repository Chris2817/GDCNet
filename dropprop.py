from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from pygcn.utils import *
from pygcn.models import DropAttr
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import pdb
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor, fill_diag, sum as sparsesum, mul
from torchmetrics import AUROC
from torchmetrics.classification import MulticlassF1Score


scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=90, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                  help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                  help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                 help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                 help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                 help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--group', type=int, default=5,
                   help='group in conv')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--num_layers', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--attn_drop', type=float, default=0.5, help='attn_drop')
parser.add_argument('--path_drop', type=float, default=0.7, help='path_drop')
parser.add_argument('--drop1', type=float, default=0.5, help='aggregation_drop in layer 1')
parser.add_argument('--drop2', type=float, default=0.5, help='aggregation_drop in layer 2')
parser.add_argument('--formermlp_drop', type=float, default=0.6, help='formermlp_drop')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset = args.dataset

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = fill_diag(adj, 1)
    deg = sparsesum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, deg_inv_sqrt.view(1, -1))
    return adj


def randomedge_sampler(adj, droprate):
    """
    Randomly drop edge and preserve percent% edges.
    """
    "Opt here"
    percent = 1-droprate 
    nnz = adj._nnz()
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz*percent)
    perm = perm[:preserve_nnz]
    adj = torch.sparse.FloatTensor(torch.stack((adj._indices()[0][perm],
        adj._indices()[1][perm])), 
        adj._values()[perm], 
        adj.shape)
    adj = normalize_adj(SparseTensor.from_torch_sparse_coo_tensor(adj))
    return adj

def main(run_k,splits):
    # seed = args.seed

    seed = args.seed+splits#+run_k
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    # adj, features, labels, idx_train, idx_val, idx_test = load_amazon(splits,dataset)
    # adj, features, labels, idx_train, idx_val, idx_test = load_ms(splits,dataset)
    # pdb.set_trace()
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
    # adj = sio.loadmat("/home/chenyong/workspace/meta_adj/{}/adj_{}".format(dataset,run_k+1))["matrix"]
    # adj = sio.loadmat("/home/chenyong/workspace/random_attack/{}/adj{}".format(dataset,i+1))["adj"]
    # adj = sp.csr_matrix(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)


    attr_adj = [[] for _ in range(args.num_layers)]
    ori_adj = [[] for _ in range(args.num_layers)]
    drop = [args.drop1]+[args.drop2]*(args.num_layers-1)
    norm_adj = normalize_adj(SparseTensor.from_torch_sparse_coo_tensor(adj)).cuda()
    for i in range(args.num_layers):
        for _ in range(args.group):
            ori_adj[i].append(norm_adj)
            new_adj = randomedge_sampler(adj,drop[i]).cuda()
            attr_adj[i].append(new_adj)
    # attr_adj[1] = attr_adj[0]
    # attr_adj = [[],[]]
    # for _ in range(args.group):
    #     new_adj = randomedge_sampler(adj,args.drop1).cuda()
    #     attr_adj[0].append(new_adj)
    # for _ in range(args.group):
    #     new_adj = randomedge_sampler(adj,args.drop2).cuda()
    #     attr_adj[1].append(new_adj)
    
    # pdb.set_trace()
    



    
    # Model and optimizer
    nfeat = features.shape[1]
    nnode = features.shape[0]
    nclass = labels.max().item() + 1
    model = DropAttr(nnode=nnode, 
        nfeat=nfeat, 
        nhid=args.hidden, 
        nclass=nclass,
        group=args.group, 
        num_layers=args.num_layers,
        input_droprate=args.input_droprate, 
        hidden_droprate=args.hidden_droprate, 
        drop1=args.drop1, 
        drop2=args.drop2, 
        sample=args.sample, 
        num_heads=1, 
        drop=args.formermlp_drop, 
        attn_drop=args.attn_drop, 
        drop_path=args.path_drop)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, 
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        features = features.cuda()

    def consis_loss(logps,temp=args.tem):

        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), 
            dim=1,keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))

        loss = loss/len(ps)
        return args.lam * loss
    
    def train(epoch):
        t = time.time()
        
        X = features
        model.train()
       
        optimizer.zero_grad()
       
        output_list = model(X,attr_adj)
        loss_consis = consis_loss(output_list)
        loss_train = 0
        loss_train = F.nll_loss(output_list[0][idx_train], labels[idx_train])
        acc_train = accuracy(output_list[0][idx_train],labels[idx_train])
        loss_train = loss_train + loss_consis




        loss_train.backward()
        optimizer.step()
    
        if not args.fastmode:
            
            model.eval()
            output_list = model(X,ori_adj)
            
        loss_val = F.nll_loss(output_list[0][idx_val], labels[idx_val]) 
        acc_val = accuracy(output_list[0][idx_val], labels[idx_val])

        # print('Epoch: {:04d}'.format(epoch+1),
        #   'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train.item()),
        #   'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val.item()),
        #   'time: {:.4f}s'.format(time.time() - t))

        return loss_val.item(), acc_val.item(),loss_train.item()
    def Train():
        # Train model
        t_total = time.time()
        loss_values = []
        acc_values = []
        bad_counter = 0
        # best = args.epochs + 1
        loss_best = np.inf
        acc_best = 0.0
        epochs = []
        loss = []

    
        loss_mn = np.inf
        acc_mx = 0.0
    
        best_epoch = 0
    
        for epoch in range(args.epochs):

            l, a, b= train(epoch)
            loss_values.append(l)
            acc_values.append(a)
            epochs.append(epoch)
            loss.append(b)

    
            #print(bad_counter)

            if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
                if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                    loss_best = loss_values[-1]
                    acc_best = acc_values[-1]
                    best_epoch = epoch
                    torch.save(model.state_dict(), 
                    "./trained_models/{}_best_model_run_{}.pkl".format(args.dataset, 
                        run_k))
                    
                    
                loss_mn = np.min((loss_values[-1], loss_mn))
                acc_mx = np.max((acc_values[-1], acc_mx))
                bad_counter = 0
            else:
                bad_counter += 1
   
           #print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
            if bad_counter == args.patience:
                print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
                print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
                break
    
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load(
            "./trained_models/{}_best_model_run_{}.pkl".format(args.dataset, run_k)))
       

        return l,a
    
    
    
    def test():
        model.eval()
        X = features
        output_list = model(X,ori_adj)
        output = output_list[0]
        f1_score = MulticlassF1Score(num_classes=output.shape[1],average='micro').cuda()
        auroc = AUROC(task="multiclass", num_classes=output.shape[1]).cuda()
        pred = output[idx_test].max(1)[1]
        f1 = f1_score(pred,labels[idx_test])
        auc = auroc(torch.exp(output[idx_test]),labels[idx_test])
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "f1_score= {:.4f}".format(f1.item()),
              "auc= {:.4f}".format(auc.item()))
        return acc_test.item(), f1.item(), auc.item()
    Train()
    acc, f1, auc = test()
    return acc, f1, auc


test_f1 = np.zeros([10])
test_auc = np.zeros([10])
test_acc = np.zeros([10])
for i in range(10):
    test_acc[i], test_f1[i], test_auc[i] = main(0,i)
print(np.mean(test_acc),'+',np.std(test_acc))
print(np.mean(test_auc),'+',np.std(test_auc))
print(np.mean(test_f1),'+',np.std(test_f1))

