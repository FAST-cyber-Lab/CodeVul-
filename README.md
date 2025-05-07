# CodeVul+

CodeVul+ is a deep learning framework for code vulnerability detection that leverages Graph Neural Networks and transformer-based models. The architecture combines GraphCodeBERT embeddings with a dynamic graph convolutional network to capture both syntactic and semantic relationships in code.


# Dataset Format.<br>
CodeVul+ expects datasets in CSV format with a column named **`functionSource`** containing the code snippets to analyze. The sample dataset included in the repository demonstrates the expected format. 

### Quick Start with Sample Dataset.<br>
Run both training and inference with the sample dataset.<br>
<pre lang="markdown"> python sample_run.py </pre>

### Run only training.<br>
<pre lang="markdown"> sample_run.py --train </pre>

### Run only inference 
<pre lang="markdown"> python sample_run.py --infer </pre>



## Installation

Clone the repository.<br>
<pre lang="markdown">git clone https://github.com/FAST-cyber-Lab/CodeVulplus.git<br> </pre>
<pre lang="markdown">cd CodeVulplus </pre>

Install dependencies.<br>
<pre lang="markdown">pip install -r requirements.txt</pre>


# Training.<br>
<pre lang="markdown"> python train.py \ --dataset path/to/dataset.csv \ --save_path models/codevul_plus.pt \ --batch_size 8 \ --epochs 3 \ --learning_rate 1e-4  </pre>

# Inference.<br>
<pre lang="markdown"> python infer.py \ --dataset path/to/test_dataset.csv \ --model_path models/codevul_plus.pt \ --batch_size 8 </pre>

