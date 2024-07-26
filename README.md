# DP-BERT: A Pre-trained Deep Language Model for Depression Prediction Using Microarray Data

## Introduction
In recent years, the increasing number of depression patients and growing awareness of depression in modern society have underscored the need for accurate depression diagnosis. Microarray data plays a crucial role in uncovering the genetic mechanisms underlying depression. However, existing methods for depression prediction using microarray data often rely on the selection of differentially expressed genes, which may overlook valuable information from other genes and be susceptible to batch effects. This leads to limited generalizability and model stability. To address these limitations, we propose DP-BERT, a depression prediction model based on Bidirectional Encoder Representations from Transformers (BERT). DP-BERT follows a pre-training-fine-tuning paradigm, leveraging a large amount of unlabeled microarray data from diverse sequencing platforms for pre-training to extract comprehensive genetic-level representations of psychiatric disorders. Subsequently, supervised fine-tuning is performed for depression prediction. Experimental results demonstrate that the attention mechanism-based pre-trained model achieves superior performance in depression prediction.

## Requirements
- python==3.8.19
- numpy==1.23.5
- numba==0.58.1
- pandas==2.0.3
- ipython==8.12.3
- torch==1.12.1
- scikit-learn==1.3.2
- einops==0.7.0

## Data
The raw data can be retrieved from the GEO database. Here we provide the test datasets used for both the pretraining and fine-tuning stages. These datasets have been preprocessed and binned.
- `./data/Psychiatric disorder-64_bin_test.csv`: Psychiatric Disorder data used for the pretraining phase.
- `./data/Depression-20_bin_test.csv`: Dataset including only samples with Depression.
## Usage
### 1. Data Preprocessing
Firstly, preprocess the input microarray data by normalizing and binning the data. For detailed implementation instructions, please refer to `/DataProcess/preprocess.py`.

### 2. Model Pretraining
The model's pretraining process involves initially applying random masking to the input data. For modifications to the masking process, please refer to specific steps in `/DP-BERT/pretrainData.py`. If no modifications are needed, you can initiate distributed training by entering the following command in the terminal (where "gpu numbers" represents the number of GPUs used):
```
python -m torch.distributed.launch --nproc_per_node "gpu numbers" pretrain.py
```
You can train your own model by modifying the following variables:
```python
num_tokens = 20002    # The size of the token embedding
dim = 200             # Embedding dimension
depth = 6             # Performer layers
max_seq_len = 24447   # Maximum sequence length
heads = 10            # The number of heads in the multi-head attention
vocab_size = 20002    # The vocabulary size, with additional tokens for 'padding' and 'mask'
num_hiddens = 2048    # The dimension of the hidden neurons in the MLP for the MLM pretraining task.
```
### 3. Fine-tuning the Pretrained Model
First, load the pretrained model. Then, you can choose to freeze different layers for fine-tuning your model. For specific details, please refer to`/DP-BERT/finetune.py` . If no modifications are needed, you can initiate distributed training by entering the following command in the terminal (where "gpu numbers" represents the number of GPUs used):
```
python -m torch.distributed.launch --nproc_per_node "gpu numbers" finetune.py
```
### 4. Model Output
The final output of the model includes the predicted class for each sample as well as overall performance evaluation metrics such as Accuracy, Precision, Recall, and F1-score.
## Contact
Please feel free to contact us for any further questions:
- Min Li limin@mail.csu.edu.cn
## References
A Pre-trained Deep Language Model for Depression Prediction Using Microarray Data
