# CodeVul+

CodeVul+ is a deep learning framework for code vulnerability detection that leverages Graph Neural Networks and transformer-based models. The architecture combines GraphCodeBERT embeddings with a dynamic graph convolutional network to capture both syntactic and semantic relationships in code.


# Dataset Format.<br>
CodeVul+ expects datasets in CSV format with a column named **`functionSource`** containing the code snippets to analyze. The sample dataset included in the repository demonstrates the expected format. 

## Installation

Clone the repository.<br>
<pre lang="markdown">git clone https://github.com/FAST-cyber-Lab/CodeVulplus.git<br> </pre>

### Create a virtual environment (recommended)
<pre lang="markdown">!python -m venv codevul_env </pre>
source codevul_env/bin/activate  <br>
On Windows: codevul_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Install dependencies.<br>
<pre lang="markdown">pip install -r ./CodeVulplus/requirements.txt</pre>



### Quick Start with Sample Dataset.<br>
Run both training and inference with the sample dataset.<br>
<pre lang="markdown">!python sample_run.py </pre>

### Run only Fine-tuning.<br>
<pre lang="markdown">!python sample_run.py --train </pre>

### Run only inference 
<pre lang="markdown"> python sample_run.py --infer </pre>



# Training.<br>
<pre lang="markdown">!python root/CodeVulplus/train.py --dataset root/CodeVulplus/data/sample_dataset.csv --save_path root/new_finetuned.pt --batch_size 4 --epochs 2
  </pre>

# Inference.<br>
<pre lang="markdown"> !python root/CodeVulplus/infer.py --dataset root/sample_dataset.csv --model_path root/CodeVulplus/pretrained/pretrained.pt --output infered_embeddings.csv </pre>


# Make prediction with your vectorized dataset.<br>
<pre lang="markdown"> !python root/CodeVulplus/predict.py --embeddings root/vectorized_dataset.csv --download</pre>

