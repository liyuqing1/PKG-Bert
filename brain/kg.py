# coding: utf-8
import os
import brain.config as config
import pkuseg
import numpy as np
import json
import requests
import sys

class ModalKnowledgeGraph(object):
    """
    input:  node_file: store nodes of PKG.
            edge_file: store edges of PKG like "[[nodeX,nodeY], [,],...]".

    Please Noice: 
    In order to respect the rights of the Unsplash website, we only publish the URL of the image nodes. 
    If you need to use the visual part of our model, please visit the official website (https://unsplash.com/documentation), and apply for an api to download these pictures. 
    The visual part will work after replacing the URL we published with your local path of these pictures.
    The encoder is the Chinese multi-modal pre-trained model, BriVL (https://github.com/chuhaojin/WenLan-api-document).
    """
    def __init__(self, node_file, edge_file, predicate=False):
        self.predicate = predicate
        self.name2nodeid = {}
        self.modal_nodes = self._input_modal_nodes(node_file)
        self.lookup_table = self._create_lookup_table(edge_file)
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _input_modal_nodes(self, node_file):
        """
        Read the information of nodes.
        """
        with open(node_file, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        node_index_id = {}
        
        for name in nodes:
            self.name2nodeid[name] = nodes[name]['id']
            id_dict1 = nodes[name]
            del id_dict1['id']
            id_dict1['name'] = name
            node_index_id[self.name2nodeid[name]] = id_dict1          
        return node_index_id

    def _create_lookup_table(self, edge_file):
        """
        Create the index with textual nodes' id.
        """
        lookup_table = {}
        with open(edge_file, 'r', encoding='utf-8') as fedge:
            edges_data = json.load(fedge)
        for edge in edges_data:
            if edge[0] not in self.modal_nodes or edge[1] not in self.modal_nodes:
                continue
            subj = self.modal_nodes[edge[0]]['name']
            obje = edge[1]
            if subj in lookup_table.keys():
                lookup_table[subj].add(obje)
            else:
                lookup_table[subj] = set([obje])
        return lookup_table

    def _get_image_embedding(self, nodeid, embsize):
        import re
        """
        Encode image nodes related with the input text node by BriVL.
        """
        image_paths = self.modal_nodes[nodeid]['downloads']
        embeddings = []

        for img_path in image_paths:
            if os.path.exists(img_path):
                files = {"image":open(img_path, 'rb')}
                emb_p_url = "http://buling.wudaoai.cn/image_query"
                response = requests.post(emb_p_url, files=files)
                emb = response.json()['embedding']
                embeddings.append(np.array(emb))
    
        #average of image embeddings
        if len(embeddings) == 0:
            return np.zeros(embsize)
        else:
            embeddings = np.array(embeddings)
            embeddings = np.mean(embeddings, axis=0)
            return embeddings
                

    def insert(self, sent_batch, photo_emb_size, max_entities=config.MAX_ENTITIES, max_length=128):
        """
        Insert knowledge into the input sentence.
        input:  sent_batch - list of input sentences.
                max_entities - the max number of inserted entities for each token.
                max_length - the max length of each input sentence.
        return: know_sent_batch - list of sentences with entites embedding.
                visual_batch - list of visual vector of each token.
                position_batch - list of position index of each token.
                visible_matrix_batch - list of visible matrixs.
                seg_batch - list of segment tags.
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        visual_batch = []

        for split_sent in split_sent_batch:
            sent_tree = []  
            pos_idx_tree = []   #soft position
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1 
            abs_idx_src = []
            for token in split_sent:
                #search entity
                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))

                #position
                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]#hard position
                
                #position for inserted knowledge
                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(self.modal_nodes[ent]['name'])+1)]
                    ent_abs_idx = [abs_idx + i  for i in range(1, len(self.modal_nodes[ent]['name'])+1)]

                    entities_pos_idx.append(ent_pos_idx)
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            know_sent = []
            pos = []
            seg = []    #=0~input word, =1~entity
            photo = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                    photo += [np.zeros(photo_emb_size)]
                else:
                    add_word = list(word)
                    know_sent += add_word 
                    seg += [0] * len(add_word)
                    if word not in self.name2nodeid: #This token is not an entity
                        photo += [np.zeros(photo_emb_size)] * len(add_word)
                    else:
                        photo += [self._get_image_embedding(nodeid=self.name2nodeid[word],embsize=photo_emb_size)] * len(add_word)
                pos += pos_idx_tree[i][0]
                
                for j in range(len(sent_tree[i][1])):
                    add_word = list(self.modal_nodes[sent_tree[i][1][j]]['name'])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])
                    photo += [self._get_image_embedding(nodeid=sent_tree[i][1][j],embsize=photo_emb_size)] * len(add_word)

            token_num = len(know_sent)

            # Count visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            #cut or pad the sentence
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                photo += [np.zeros(photo_emb_size)] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                photo = photo[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            visual_batch.append(photo)
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)         
        
        return know_sent_batch, visual_batch, position_batch, visible_matrix_batch, seg_batch

