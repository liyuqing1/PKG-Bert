# -*- encoding:utf-8 -*-
from uer.layers.embeddings import BertEmbedding,BertVisualEmbedding
from uer.models.model import ModalModel
from uer.encoders.bert_encoder import BertEncoder
from uer.targets.mlm_target import MlmTarget
import torch

def build_modal_model(args):
    """
    Build PKG-Bert
    """
    if args.no_visual:
        embedding = BertEmbedding(args, vocab_size=len(args.vocab))
        print("···No Visual Embedding in Input Layer.···")
    else:
        embedding = BertVisualEmbedding(args, vocab_size=len(args.vocab), photo_dim=args.photo_emb_size)
    encoder = globals()["BertEncoder"](args)
    target = globals()["MlmTarget"](args, len(args.vocab))
    model = ModalModel(args, embedding, encoder, target)

    return model


def save_model(model, model_path, epoch=None):
    if hasattr(model, "module"):
        if epoch is None:
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save({"epoch":epoch, "model_state_dict":model.module.state_dict()}, model_path)
    else:
        if epoch is None:
            torch.save(model.state_dict(), model_path)
        else:
            torch.save({"epoch":epoch, "model_state_dict":model.state_dict()}, model_path)
