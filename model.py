#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Define model
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from utils import normalize_adj, normalize_features
START_TAG = "<START>"
STOP_TAG = "<STOP>"

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return self.softmax(x)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        
class GInNER(nn.Module):
  def __init__(self, word_embedding_dim, tag_to_idx, device, dropout, hidden_layer, nheads):
      super(GInNER, self).__init__()
      self.word_embedding_dim = word_embedding_dim
      self.input_size = self.word_embedding_dim
      self.hidden_layer = hidden_layer
      self.nheads = nheads
      self.dropout = dropout
      self.device = device
      self.tag_to_idx = tag_to_idx
      self.tagset_size = len(tag_to_idx)
      self.gat = GAT(nfeat=self.input_size, nhid=self.hidden_layer, nclass=self.tagset_size, dropout=self.dropout, nheads=self.nheads, alpha=0.2)
      self.device = device
      
      # Matrix of transition parameters.  Entry i,j is the score of
      # transitioning *to* i *from* j.
      self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

      # These two statements enforce the constraint that we never transfer
      # to the start tag and we never transfer from the stop tag
      self.transitions.data[tag_to_idx[START_TAG], :] = -10000
      self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000
        
  def forward(self, input_tensor, adj):
    
      logits = self._get_gat_features(input_tensor, adj)
      # Find the best path, given the features.
      #score, tag_seq = self._viterbi_decode(logits)
      return logits#score, tag_seq
    
  def neg_log_likelihood(self, input_tensor, adj, tags):
      feats = self._get_gat_features(input_tensor, adj)
      forward_score = self._forward_alg(feats)
      gold_score = self._score_sentence(feats, tags)
      return forward_score - gold_score
  
  def _viterbi_decode(self, feats):
      backpointers = []

      # Initialize the viterbi variables in log space
      init_vvars = torch.full((1, self.tagset_size), -10000.)
      init_vvars[0][self.tag_to_idx[START_TAG]] = 0

      # forward_var at step i holds the viterbi variables for step i-1
      forward_var = init_vvars
      #print("len feat", len(feats))
      for feat in feats:
          bptrs_t = []  # holds the backpointers for this step
          viterbivars_t = []  # holds the viterbi variables for this step

          for next_tag in range(self.tagset_size):
              # next_tag_var[i] holds the viterbi variable for tag i at the
              # previous step, plus the score of transitioning
              # from tag i to next_tag.
              # We don't include the emission scores here because the max
              # does not depend on them (we add them in below)
              next_tag_var = forward_var + self.transitions[next_tag]
              best_tag_id = argmax(next_tag_var)
              bptrs_t.append(best_tag_id)
              viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
          # Now add in the emission scores, and assign forward_var to the set
          # of viterbi variables we just computed
          forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
          backpointers.append(bptrs_t)

      # Transition to STOP_TAG
      terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
      best_tag_id = argmax(terminal_var)
      path_score = terminal_var[0][best_tag_id]

      # Follow the back pointers to decode the best path.
      best_path = [best_tag_id]
      #print("best path", best_path)
      #print("backpointers", len(backpointers))
      for bptrs_t in reversed(backpointers):
          best_tag_id = bptrs_t[best_tag_id]
          #print("best_tag_id", best_tag_id)
          best_path.append(best_tag_id)
      # Pop off the start tag (we dont want to return that to the caller)
      start = best_path.pop()
      assert start == self.tag_to_idx[START_TAG]  # Sanity check
      best_path.reverse()
      #print("best path", best_path, len(best_path))
      return path_score, best_path
  
  def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
  def _get_gat_features(self, input_tensor, adj):
      embedded = input_tensor
      adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
      adj = normalize_adj(adj + sp.eye(adj.shape[0]))
      adj = torch.FloatTensor(np.array(adj.todense()))
      features = normalize_features(embedded.detach().numpy())
      features = torch.FloatTensor(np.array(features))
    
      logits = self.gat(features, adj)
      
      return logits
  
  def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score