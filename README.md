# PKG-Bert

Source code and data for the paper *A Multi-Modal Knowledge Graph for Classical Chinese Poetry* which is accepted by the Findings of EMNLP 2022.



## Requirements

```bash
Python3
Pytorch >= 1.6
pkuseg
argparse
requests
```



## Usage

#### Datasets

The multi-modal knowledge graph for classical Chinese poetry (PKG) can be found in [./brain/kg_info/](./brain/kg_info/). We store the image entity in the form of URL, please contact [Beautiful Free Images & Pictures | Unsplash](https://unsplash.com/) to download these images.

The labeled evaluation dataset for poetry-image retrieval task can be found in [./evaluation/](./evaluation/).

We also provide an example in [./datasets/poetry.txt](./datasets/poetry.txt) to show the expected format of poetry corpus for pre-training.

#### Run

After downloading the image entities in PKG and replacing the URL with your local address for these images, run PKG-Bert by:

```bash
python modal_main.py --only_predict
```

Options of `modal_main.py`:

```bash
usage: 	[--only_predict] #Whether or not predict without pretraining.
		[--output_model_path] #Path to the pretrained model of PKG-Bert.
		[--pretrained_bert_path] #Path to the model parameters of Google Bert.
		[--train_path] #Path to the train set of poetry corpus.
		[--dev_path] #Path to the validation set of poetry corpus.
		[--kg_node] #Path to the storage of nodes in PKG.
		[--kg_edge] #Path to the storage of edges in PKG.
		[--no_visual] #Drop the visual part.
		[--no_kg] #Drop the whole PKG.
        [--batch_size] #The batch size during pretraining.
        [--epochs_num] #The maxium epoches of pretraining.
        [--GPUids] #Your available GPU ids.
```



## Acknowledgements

The image encoder in this model is [BriVL](https://github.com/chuhaojin/WenLan-api-document).

Codes are partially based on [autoliuweijie/K-BERT: Source code of K-BERT (AAAI2020)](https://github.com/autoliuweijie/K-BERT).

