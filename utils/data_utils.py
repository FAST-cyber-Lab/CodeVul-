import torch
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer

def clean_code(code_str):

    if not isinstance(code_str, str):
        return ""
    

    code_str = re.sub(r'#.*$', '', code_str, flags=re.MULTILINE)
    code_str = re.sub(r'\/\/.*$', '', code_str, flags=re.MULTILINE)
    code_str = re.sub(r'\/\*[\s\S]*?\*\/', '', code_str)
    
    code_str = re.sub(r'\s+', ' ', code_str)
    

    code_str = re.sub(r';\s*', ';\n', code_str)
    code_str = re.sub(r'{\s*', '{\n', code_str)
    code_str = re.sub(r'}\s*', '}\n', code_str)
    
    return code_str.strip()

class CodeEmbedder:
    def __init__(self, model_name="microsoft/graphcodebert-base", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
    
    def embed_code_snippets(self, batch_code, max_length=None, clean=True):
        if clean:
            batch_code = [clean_code(code) for code in batch_code]
        
        if max_length is None:
            token_lengths = [len(self.tokenizer.encode(code, truncation=False)) for code in batch_code]
            max_length = int(np.percentile(token_lengths, 95))
        
        max_length = min(max_length, self.tokenizer.model_max_length)
        
        with torch.no_grad():
            tokenized_code = self.tokenizer(
                batch_code,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False
            ).to(self.device)
            outputs = self.model(**tokenized_code)
        
        return outputs.last_hidden_state
