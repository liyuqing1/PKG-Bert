# -*- encoding:utf-8 -*-
import sys
import torch
import argparse
import requests
import json
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_modal_model,save_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from brain import ModalKnowledgeGraph
import numpy as np
from uer.utils.data import mask_seq
from torch.utils.data import Dataset, DataLoader
import brain.config as config
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--only_predict", action="store_true", help="Without pretraining, only predict.")
    # Path options.
    parser.add_argument("--pretrained_bert_path", default='./models/google_model.bin', type=str, 
                        help="Path of the Google Bert.")
    parser.add_argument("--output_model_path", default="./outputs/kbert_visual_poem", type=str,
                        help="Path of the output model of pretraining.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str,  default='./datasets/poetry.txt',
                        help="Path of the train set of poetry corpus.")
    parser.add_argument("--dev_path", type=str, default='./datasets/poetry.txt',
                        help="Path of the validation set of poetry corpus.") 
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--no_visual", action='store_true', help="Drop the visual part.")
    parser.add_argument("--no_kg", action='store_true', help="Drop the whole KG.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256, help="The max number of characters in a sample.")
    parser.add_argument("--photo_emb_size", type=int, default=2048, help="The dimension of the output embeddings.")
    parser.add_argument("--early_stop", type=int, default=5, help="When performence continuously becomes bad for 'early_stop' epochs, stop pretrain in case of overfitting.")
    parser.add_argument("--loss_w", type=int, default=1, help="loss = loss_mlm + loss_w × loss_brivl")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--epochs_num", type=int, default=100,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print information such as loss.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--GPUids", type=str, default="",
                        help="Usable GPU ids, intervaled by ','.")
    parser.add_argument("--eval_interval", type=int, default=3,help="After how many epochs of pretrain, model will evaluate on dev.")

    # kg
    parser.add_argument("--kg_node", help="The path of file storing PKG's nodes",default='./brain/kg_info/PKG_node.json')
    parser.add_argument("--kg_edge", help="The path of file storing PKG's edges",default='./brain/kg_info/PKG_edge.json')
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix.")

    return parser.parse_args()


def _input_sents(path):
    """
    Read corpus from path, and return the list of sentences.
    """
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        line = f.readline()
        while line:
            sentences.append(line)
            line = f.readline()
            
    sentence_num = len(sentences)
    print(f"\tAlready input {sentence_num} sentences.")
    return sentences

def _insert_kg(params):
    """
    Insert knowledge into the input poetry.
    """
    p_id, sentences, kg, vocab, args = params

    dataset = []
    for line in sentences:
        text = CLS_TOKEN + line.strip() + SEP_TOKEN

        #insert knowledge
        if args.no_kg:
            tokens = kg.tokenizer.cut(text)
            pos = [i for i in range(len(tokens))] #hard position
            if len(tokens) < args.seq_length:#pad [PAD]
                pad_num = args.seq_length - len(tokens)
                tokens += [config.PAD_TOKEN] * pad_num
                pos += [len(tokens) - 1] * pad_num
            else:
                tokens = tokens[:args.seq_length]
                pos = pos[:args.seq_length]
            visual = [np.zeros(args.photo_emb_size)] * args.seq_length
            vm = np.zeros([args.seq_length, args.seq_length])
        else:
            tokens, visual, pos, vm, _ = kg.insert([text], photo_emb_size=args.photo_emb_size, max_length=args.seq_length)
            visual = visual[0]
            tokens = tokens[0] 
            pos = pos[0]
            vm = vm[0].astype("bool")

        token_ids = [vocab.get(t) for t in tokens]
        text_kg = ''
        text_kg = text_kg.join(tokens)
        text_kg = text_kg.replace(CLS_TOKEN, '').replace(SEP_TOKEN, '').replace(PAD_TOKEN, '')

        seg = []
        seg_tag = 1
        for t in tokens:
            if t == PAD_TOKEN:
                seg.append(0)
            else:
                seg.append(seg_tag)
            if t == SEP_TOKEN:
                seg_tag += 1

        dataset.append((token_ids, visual, seg, pos, vm, text_kg))

    return dataset


class My_dataset(Dataset):
    def __init__(self, sentences, kg,vocab, args, is_eval=False):
        super().__init__()
        self.sentences = sentences
        self.kg = kg
        self.vocab = vocab
        self.args = args
        self.insert_num = 20000
        self.data = None
        self.brivl_embs = None
        self.is_eval = is_eval
           
    def __getitem__(self, index):
        if index % self.insert_num==0:
            sentences = self.sentences[index:index+self.insert_num] if index+self.insert_num<len(self.sentences) else self.sentences[index:]
            params = (0, sentences, self.kg, self.vocab, self.args)
            self.data = _insert_kg(params)
            if self.is_eval == False:
                self.brivl_embs = []
                for txt in [data[5] for data in self.data]:
                    response = requests.get("http://buling.wudaoai.cn/text_query?text=" + txt)#The embedding in the vector space of visual information.
                    emb = response.json()['embedding']
                    self.brivl_embs.append(np.array(emb))
        
        if self.is_eval:
            return torch.LongTensor(self.data[index % self.insert_num][0]), torch.tensor(self.data[index % self.insert_num][1]).to(torch.float32), torch.LongTensor(self.data[index % self.insert_num][2]), torch.LongTensor(self.data[index % self.insert_num][3]), torch.LongTensor(self.data[index % self.insert_num][4])
        else:
            return torch.LongTensor(self.data[index % self.insert_num][0]), torch.tensor(self.data[index % self.insert_num][1]).to(torch.float32), torch.LongTensor(self.data[index % self.insert_num][2]), torch.LongTensor(self.data[index % self.insert_num][3]), torch.LongTensor(self.data[index % self.insert_num][4]), torch.tensor(self.brivl_embs[index % self.insert_num]).to(torch.float32)

    def __len__(self):
        return len(self.sentences) 
        

def pretrain():
    #read parameters
    args = get_args()
    args = load_hyperparam(args)
    if args.no_kg:#drop the whole KG
        args.no_visual = True
        args.no_vm = True
        print("···No KG Insertion.···")
    
    set_seed(args.seed)
    #set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUids
    gpus = args.GPUids.split(',')
    gpus = [int(id) for id in range(len(gpus))]

    # Load vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build model
    model = build_modal_model(args)

    # Load or initialize parameters.
    if args.pretrained_bert_path is not None:# Initialize with Bert by Google.
        model_dict = torch.load(args.pretrained_bert_path)
        #Drop nsp
        for key in list(model_dict.keys()):
            if 'nsp' in key:
                del model_dict[key]
        #Change the input and target layer
        for n, p in list(model.named_parameters()):
            if n not in list(model_dict.keys()):
                model_dict[n] = torch.nn.init.normal_(torch.Tensor(p.shape), mean=0.0, std=0.02)
        model.load_state_dict(model_dict)  
    else:#Initialize randomly
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpus)
    model = model.to(device)
    
    kg = ModalKnowledgeGraph(node_file=args.kg_node, edge_file=args.kg_edge, predicate=False)

    
    print("···Start training···")
    trainsents = _input_sents(args.train_path)
    devsents = _input_sents(args.dev_path)
    instances_num = len(trainsents)

    print("···Create train and validation dataset···")
    my_dataset = My_dataset(trainsents, kg, vocab, args)
    my_dataloader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=False)
    dev_dataset = My_dataset(devsents, kg, vocab, args)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    train_steps = int(instances_num * args.epochs_num / args.batch_size) + 1

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
    if torch.cuda.device_count() > 1:
        optimizer = nn.DataParallel(optimizer, device_ids=gpus)
    
    sum_loss1 = 0.
    sum_loss2 = 0.
    sum_correct = 0
    sum_mlmnum = 0    
    eval_loss_min = None
    early_stop = 0
    #checkpoint
    if os.path.exists(args.output_model_path):
        print("···Upload the pretrained model from {}···".format(args.output_model_path))
        params_dict = torch.load(args.output_model_path, map_location=torch.device('cpu'))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(params_dict['model_state_dict'])
        else:
            model.load_state_dict(params_dict['model_state_dict'])
        epoch_start = params_dict['epoch']+1
    else:
        epoch_start = 1
    for epoch in range(epoch_start, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch,visual_emb_batch, seg_ids_batch, pos_ids_batch, vms_batch, brivl_batch) in enumerate(my_dataloader):
            model.zero_grad()

            #MLM
            tgt_mlm = []
            for j in range(input_ids_batch.shape[0]):
                input_ids_batch[j],mlm1 = mask_seq(src=input_ids_batch[j], vocab_size=vocab.__len__())             
                tgt_mlm.append(np.array(mlm1))
            tgt_mlm = torch.tensor(tgt_mlm)

            input_ids_batch = input_ids_batch.to(device)
            visual_emb_batch = visual_emb_batch.to(device)
            seg_ids_batch = seg_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
            tgt_mlm = tgt_mlm.to(device)
            brivl_batch = brivl_batch.to(device)

            loss1, loss2, correct, denominator = model(visual=visual_emb_batch,src=input_ids_batch, seg=seg_ids_batch, pos=pos_ids_batch, vm=vms_batch, tgt_mlm=tgt_mlm, tgt_brivl=brivl_batch)
            if torch.cuda.device_count() > 1:
                loss1 = torch.mean(loss1)
                loss2 = torch.mean(loss2)
                correct = torch.mean(correct)
                denominator = torch.mean(denominator)
            loss = torch.add(loss1, args.loss_w, loss2)

            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            sum_correct += correct.item()
            sum_mlmnum += denominator.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch {}, training steps {}: train loss={:.3f},  loss(mlm): {:.3f},  loss(brivl): {:.3f}".format(epoch, i+1, (sum_loss1+sum_loss2)/args.report_steps, sum_loss1 / args.report_steps, sum_loss2 / args.report_steps))
                sys.stdout.flush()
                sum_loss1 = 0.
                sum_loss2 = 0.
            
            loss.backward()
            optimizer.step()

        if sum_loss1 > 1e-8:#Some training steps have not printed yet.
            print("Epoch {}, training steps {}: train loss={:.3f},  loss(mlm): {:.3f},  loss(brivl): {:.3f}".format(epoch, i+1, (sum_loss1+sum_loss2)/args.report_steps, sum_loss1 / args.report_steps, sum_loss2 / args.report_steps))
            sys.stdout.flush()
        sum_correct = 0
        sum_mlmnum = 0
        sum_loss1 = 0.
        sum_loss2 = 0.
        save_model(model, args.output_model_path+'-'+str(epoch), epoch)

        #early stop
        if epoch % args.eval_interval == 0:
            print("···Start evalution on dev···")
            with torch.no_grad():
                model.eval()
                for i, (input_ids_batch,visual_emb_batch, seg_ids_batch, pos_ids_batch, vms_batch, brivl_batch) in enumerate(dev_dataloader):
                    model.zero_grad()
                    
                    #MLM mask token
                    tgt_mlm = []
                    for j in range(input_ids_batch.shape[0]):
                        input_ids_batch[j],mlm1 = mask_seq(src=input_ids_batch[j], vocab_size=vocab.__len__())              
                        tgt_mlm.append(np.array(mlm1))
                    tgt_mlm = torch.tensor(tgt_mlm)

                    input_ids_batch = input_ids_batch.to(device)
                    visual_emb_batch = visual_emb_batch.to(device)
                    seg_ids_batch = seg_ids_batch.to(device)
                    pos_ids_batch = pos_ids_batch.to(device)
                    vms_batch = vms_batch.to(device)
                    tgt_mlm = tgt_mlm.to(device)
                    brivl_batch = brivl_batch.to(device)

                    loss1, loss2, correct, denominator = model(visual=visual_emb_batch,src=input_ids_batch, seg=seg_ids_batch, pos=pos_ids_batch, vm=vms_batch, tgt_mlm=tgt_mlm, tgt_brivl=brivl_batch)#loss~损失、correct~预测正确数量、denominator~mask总数
                    if torch.cuda.device_count() > 1:
                        loss1 = torch.mean(loss1)
                        loss2 = torch.mean(loss2)
                    loss = torch.add(loss1, args.loss_w, loss2)
                    
                    sum_loss1 += loss1.item()
                    sum_loss2 += loss2.item()
                    sum_loss = sum_loss1 + sum_loss2
                if eval_loss_min is None or eval_loss_min>sum_loss:
                    eval_loss_min = sum_loss
                    early_stop = 0
                    print("-Epoch "+str(epoch)+": dev loss="+str(sum_loss)+", loss(mlm)="+str(sum_loss1)+", loss(brivl)="+str(sum_loss2))
                else:
                    print(f"-Epoch {epoch}: dev loss="+str(sum_loss)+", loss(mlm)="+str(sum_loss1)+", loss(brivl)="+str(sum_loss2))
                    for eid in range(epoch-args.eval_interval+1, epoch+1):
                        os.remove(args.output_model_path+'-'+str(eid))
                    early_stop +=1
                sum_loss1 = 0.
                sum_loss2 = 0.
        if early_stop >= args.early_stop:
            print("Overfit, stop!")
            return


pre_params = {'model':None, 'device':None,'args':None, 'kg':None, 'vocab':None }
def predo():
    """
    Before prediction, upload the PKG-Bert.
    """
    global pre_params
    print("···Start preparation of PKG-Bert···")
    args = get_args()
    
    #load hyperparam
    args = load_hyperparam(args)
    if args.no_kg:
        args.no_visual = True
        args.no_vm = True
        print("···No KG···")
    set_seed(args.seed)

    # Load vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab


    model = build_modal_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params_dict = torch.load(args.output_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(params_dict['model_state_dict'])

    #check the number of parameters
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    kg = ModalKnowledgeGraph(node_file=args.kg_node, edge_file=args.kg_edge, predicate=False)
    
    pre_params = {'model':model, 'device':device,'args':args, 'kg':kg, 'vocab':vocab }
    print("···Model and KG are already uploaded···")

def get_sentences_emb(sentences):
    """
    input:  the list of sentences.
    return: the embedding from PKG-Bert of these sentences.
    """
    if pre_params['vocab'] is None:
        predo()
    model=pre_params['model']
    device=pre_params['device']
    args=pre_params['args']
    kg=pre_params['kg']
    vocab=pre_params['vocab']

    my_dataset = My_dataset(sentences, kg, vocab, args, is_eval=True)
    my_dataloader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=False)


    model.eval()
    result_emb = None
    
    for i, (input_ids_batch,visual_emb_batch, seg_ids_batch, pos_ids_batch, vms_batch) in enumerate(my_dataloader):
        with torch.no_grad():
            input_ids_batch = input_ids_batch.to(device)
            visual_emb_batch = visual_emb_batch.to(device)
            seg_ids_batch = seg_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            output = model(visual=visual_emb_batch,src=input_ids_batch, seg=seg_ids_batch, pos=pos_ids_batch, vm=vms_batch, tgt_mlm=None, tgt_brivl=None, is_eval=True)
            output = output.cuda().data.cpu().numpy() if torch.cuda.is_available() else output.numpy()
            if i == 0:
                result_emb = output 
            else:
                result_emb.extend(output)
            
    return np.array(result_emb)


if __name__ == "__main__":
    args = get_args()

    if args.only_predict == False:
        pretrain()
    else:
        predo()
        result = get_sentences_emb(["两岸青山相对出，孤帆一片日边来。", "浮云不共此山齐，山霭苍苍望转迷。"])# Test the poetry encoder by two poetry.
        print(result)

