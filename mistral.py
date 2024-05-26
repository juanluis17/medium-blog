from huggingface_hub import login
login()



#
from datasets import load_dataset

dataset = load_dataset('Unbabel/TowerBlocks-v0.2', split='train')
train = dataset.filter(lambda example: example['split']=="train" and example['task']=="machine_translation")
dev = dataset.filter(lambda example: example['split']=="dev" and example['task']=="machine_translation")
print(len(train), len(dev))

#
from transformers import AutoTokenizer

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

#
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)