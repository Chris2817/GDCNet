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
from pygcn.sample import *
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
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
args.normalization = "AugNormAdj"
dataset_name = args.dataset

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
    # seed = args.seed#+i
    seed = args.seed+run_k
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    dataset = PygNodePropPredDataset(name=dataset_name)
    data = dataset[0]
    edge_index, values = to_undirected(data.edge_index,
        torch.ones(data.edge_index.shape[1]),
        reduce="add")

    
    adj = torch.sparse.FloatTensor(edge_index, 
        values, 
        torch.Size((data.num_nodes,data.num_nodes))).cuda()
    ori_adj = [[] for _ in range(args.num_layers)]
    attr_adj = [[] for _ in range(args.num_layers)]
    drop = [args.drop1]+[args.drop2]*(args.num_layers-1)
    norm_adj = normalize_adj(SparseTensor.from_torch_sparse_coo_tensor(adj)).cuda()
    for i in range(args.num_layers):
        for _ in range(args.group):
            ori_adj[i].append(norm_adj)
            new_adj = randomedge_sampler(adj,drop[i]).cuda()
            attr_adj[i].append(new_adj)

    data = data.cuda()
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].cuda()
    val_idx = split_idx['valid'].cuda()
    test_idx = split_idx['test'].cuda()

    evaluator = Evaluator(name=dataset_name)

   

    # Model and optimizer
    nfeat = data.x.shape[1]
    nclass = data.y.squeeze(1).max().item()+1
    nnode = data.x.shape[0]
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
        drop_path=args.path_drop,
        use_bn=args.use_bn)
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    if args.cuda:
        model.cuda()


    def consis_loss(logps,temp=args.tem):

        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        # avg_p = torch.exp(logps)
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1,keepdim=True)).detach()
        # loss = torch.mean((avg_p-sharp_p).pow(2).sum(1))
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))

        loss = loss/len(ps)
        return args.lam * loss
    
    def train(epoch):
        t = time.time()
        
        X = data.x
        model.train()
        optimizer.zero_grad()

        output_list = model(X,attr_adj)

        loss_consis = consis_loss(output_list)
        output = output_list[0]
        loss_train = F.nll_loss(output_list[0][train_idx], data.y.squeeze(1)[train_idx])
        loss_train = loss_train + loss_consis
        y_pred = output.argmax(dim=-1, keepdim=True)
        acc_train = evaluator.eval({
        'y_true': data.y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']

        loss_train.backward()
        optimizer.step()
    
        if not args.fastmode:
            
            model.eval()



            output_list = model(X,ori_adj)
           
            output = output_list[0]
            
        loss_val = F.nll_loss(output[val_idx], data.y.squeeze(1)[val_idx]) 
        y_pred = output.argmax(dim=-1, keepdim=True)
        acc_val = evaluator.eval({
        'y_true': data.y[val_idx],
        'y_pred': y_pred[val_idx],
    })['acc']

        # print('Epoch: {:04d}'.format(epoch+1),
        #   'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train),
        #   'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val),
        #   'time: {:.4f}s'.format(time.time() - t))

        return loss_val.item(), acc_val,loss_train.item()
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
                    "./trained_models/{}_best_model.pkl".format(args.dataset))
                    
                loss_mn = np.min((loss_values[-1], loss_mn))
                acc_mx = np.max((acc_values[-1], acc_mx))
                bad_counter = 0
            else:
                bad_counter += 1
   
           #print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
            if bad_counter == args.patience:
                # print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
                # print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
                break
    
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
        
        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load(
            "./trained_models/{}_best_model.pkl".format(args.dataset)))


        return l,a
    
    
    
    def test():
        model.eval()
        X = data.x
    
        output_list = model(X,ori_adj)
        output = output_list[0]
        f1_score = MulticlassF1Score(num_classes=output.shape[1],average='micro').cuda()
        auroc = AUROC(task="multiclass", num_classes=output.shape[1]).cuda()
        pred = output[test_idx].max(1)[1]
        f1 = f1_score(pred,data.y.squeeze()[test_idx])
        auc = auroc(torch.exp(output[test_idx]),data.y.squeeze()[test_idx])
        loss_test = F.nll_loss(output[test_idx], data.y.squeeze(1)[test_idx])
        y_pred = output.argmax(dim=-1, keepdim=True)
        acc_test = evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test),
              "f1_score= {:.4f}".format(f1.item()),
              "auc= {:.4f}".format(auc.item()))
        return acc_test,f1.item(),auc.item()
    Train()
    acc,f1,auc =test()
    return acc,f1,auc

# main(0,0)
test_f1 = np.zeros([10])
test_auc = np.zeros([10])
test_acc = np.zeros([10])
for i in range(10):
    test_acc, test_f1[i], test_auc[i] = main(i,0)
print(np.mean(test_acc),'+',np.std(test_acc))
print(np.mean(test_auc),'+',np.std(test_auc))
print(np.mean(test_f1),'+',np.std(test_f1))

