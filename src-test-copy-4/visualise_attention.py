# coding=gbk
import argparse

import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from GNN import GNN
import time
from data import get_dataset
from run_GNN import get_optimizer, print_model_params, train, test
import networkx as nx
import matplotlib.pyplot as plt

from best_params import best_params_dict

def merge_cmd_args(cmd_opt, opt):
  # if cmd_opt['beltrami']:
  #   opt['beltrami'] = True
  # if cmd_opt['function'] is not None:
  #   opt['function'] = cmd_opt['function']
  # if cmd_opt['block'] is not None:
  #   opt['block'] = cmd_opt['block']
  # if cmd_opt['attention_type'] != 'scaled_dot':
  #   opt['attention_type'] = cmd_opt['attention_type']
  # if cmd_opt['self_loop_weight'] is not None:
  #   opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  # if cmd_opt['method'] is not None:
  #   opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  # if cmd_opt['time'] != 1:
  #   opt['time'] = cmd_opt['time']
  # if cmd_opt['epoch'] != 100:
  #   opt['epoch'] = cmd_opt['epoch']
  # if not cmd_opt['not_lcc']:
  #   opt['not_lcc'] = False
  # if cmd_opt['num_splits'] != 1:
  #   opt['num_splits'] = cmd_opt['num_splits']

def construct_graph(model):
  edges = model.odeblock.odefunc.edge_index
  edge_list = zip(edges[0], edges[1])
  g = nx.Graph(edge_list)
  nx.draw(g)

def main(opt):

  dataset = get_dataset(opt, '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, data = GNN(opt, dataset, device).to(device), dataset.data.to(device)
  print(opt)
  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))

  construct_graph(model)

  return train_acc, best_val_acc, test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='attention', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', type=bool, default=False,
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations allowed.")

  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # 增加的
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true', help='Add a fully connected layer to the encoder.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')


  args = parser.parse_args()

  opt = vars(args)
  # opt['beltrami'] = True
  cmd_opt = opt
  best_opt = best_params_dict[cmd_opt['dataset']]
  opt = {**cmd_opt, **best_opt}
  merge_cmd_args(cmd_opt, opt)

  main(opt)
