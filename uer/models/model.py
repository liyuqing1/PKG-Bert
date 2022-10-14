# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class MultimodalTarget(nn.Module):
    """
    Map poetry representations into the vector space of visual information.
    Loss: cosine loss.
    """
    def __init__(self, args, method='cos'):
        super(MultimodalTarget, self).__init__()
        self.method = method
        self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size)
        self.mlm_linear_2 = nn.Linear(args.hidden_size, args.photo_emb_size)
        if method == 'cos':
            self.criterion = nn.CosineEmbeddingLoss()
        elif method == 'L2':
            self.criterion = nn.MSELoss()

    def forward(self, memory_bank, tgt):
        memory_bank = memory_bank[:,0,:]#Take [CLS]'s embedding as the representation of the whole sentence
        output_emb = gelu(self.mlm_linear_1(memory_bank))
        output_emb = self.layer_norm(output_emb)
        output_emb = self.mlm_linear_2(output_emb)

        if tgt is None:#when predict, there's no need to conduct MLM task.
            return output_emb
        if self.method == 'cos':
            ones = torch.ones(tgt.shape[0]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            loss = self.criterion(output_emb, tgt, ones)
        elif self.method == 'L2':
            loss = self.criterion(output_emb, tgt)
        return loss, output_emb


class ModalModel(nn.Module):
    """
    PKG-Bert consists of 3 parts:
        - Input embedding: image embedding, token embedding, position embedding, segment embedding
        - Encoder: Transformer
        - Target: MLM + Minimize Space Distance
    """
    def __init__(self, args, embedding, encoder, target):
        super(ModalModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.multi_target = MultimodalTarget(args=args)
        self.no_visual = args.no_visual
        self.no_vm = args.no_vm
        

    def forward(self, visual, src, tgt_mlm, tgt_brivl, seg, pos=None, vm=None, is_eval=False):
        # [batch_size, seq_length, emb_size]
        if self.no_vm:
            vm = None
        if self.no_visual:
            emb = self.embedding(src, seg, pos) 
        else:
            emb = self.embedding(visual, src, seg, pos) 

        output_tran = self.encoder(emb, seg, vm) #[batch_size × seq_length × emb_size]    
        if is_eval:
            output_emb = self.multi_target(output_tran, None) 
            return output_emb
        else:
            loss2, output_emb = self.multi_target(output_tran, tgt_brivl) #[batch_size × photo_emb_size]
            loss_mlm, correct_mlm, denominator = self.target(output_tran, tgt_mlm)
            return loss_mlm, loss2, correct_mlm, denominator
