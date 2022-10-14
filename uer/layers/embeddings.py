# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, pos=None):
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                            dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb



class BertVisualEmbedding(nn.Module):
    """
    4部分组成：
    word embedding, 新增visual embedding, position embedding, segment embedding.
    """
    def __init__(self, args, vocab_size, photo_dim):
        super(BertVisualEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.visual_embedding = nn.Linear(in_features=photo_dim, out_features=args.emb_size) #·图片层
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, visual, src, seg, pos=None):#src = 诗+知识向量byBrivl
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                            dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)
        visual_emb = self.visual_embedding(visual)

        emb = word_emb + pos_emb + seg_emb + visual_emb 
        emb = self.dropout(self.layer_norm(emb))
        return emb


