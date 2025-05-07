# CodeVul+

CodeVul+ is a deep learning framework for code vulnerability detection that leverages Graph Neural Networks and transformer-based models. The architecture combines GraphCodeBERT embeddings with a dynamic graph convolutional network to capture both syntactic and semantic relationships in code.

## Installation

Clone the repository.<br>
git clone https://github.com/your-username/CodeVul+.git.<br>
cd CodeVul+

Install dependencies.<br>
pip install -r requirements.txt


# Training.<br>
<pre lang="markdown"> python train.py \ --dataset path/to/dataset.csv \ --save_path models/codevul_plus.pt \ --batch_size 8 \ --epochs 3 \ --learning_rate 1e-4  </pre>

# Inference.<br>
<pre lang="markdown"> python infer.py \ --dataset path/to/test_dataset.csv \ --model_path models/codevul_plus.pt \ --batch_size 8 </pre>

