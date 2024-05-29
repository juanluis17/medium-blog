# from huggingface_hub import login
# login()


#
from datasets import load_dataset

dataset = load_dataset('Unbabel/TowerBlocks-v0.2', split='train')
dataset = dataset.filter(lambda example: example['task'] == "machine_translation")
dataset = dataset.class_encode_column("lang")
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column="lang")
train = dataset["train"]
dev = dataset["test"]
print(len(train), len(dev))

#
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

#
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token


# Setup the tokenize function to make labels and input_ids the same. This is basically what self-supervised fine-tuning is:
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


# And convert each sample into a prompt that I found from this [notebook](https://github.com/samlhuillier/viggo-finetune/blob/main/llama/fine-tune-code-llama.ipynb).
def generate_prompt(instruction, response,
                    sep="\n\n### "):  # The prompt format is taken from the official Mixtral huggingface page
    # p = "<s> [INST]" + instruction + "[/INST]" + response + "</s>"
    p = instruction+"\n"+response
    return p


def generate_and_tokenize_prompt(data_point):
    messages = data_point["conversations"]
    if len(messages) > 2:
        raise Exception("Too many messages")
    full_prompt = generate_prompt(messages[0]["value"], messages[1]["value"])
    return tokenize(full_prompt)


# Reformat the prompt and tokenize each sample:
tokenized_train_dataset = train.map(generate_and_tokenize_prompt)
tokenized_val_dataset = dev.map(generate_and_tokenize_prompt)

# Check that input_ids is padded on the left with the eos_token (2) and there is an eos_token 2 added to the end, and the prompt starts with a bos_token (1).
# Check that a sample has the max length, i.e. 512.
print(len(tokenized_train_dataset[4]['input_ids']))

# How does the base model do?
# Let's grab a test input (meaning_representation) and desired output (target) pair to see how the base model does on it.
print("Input:", dev[1]['conversations'][0]['value'])
print("Output:",dev[1]['conversations'][1]['value'] + "\n")
# Re-init the tokenizer so it doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)
# to_evaluate = "<s> [INST]" + dev[1]['conversations'][0]['value'] + "[/INST]"
to_evaluate = dev[1]['conversations'][0]['value']
print("Input:",to_evaluate)
model_input = eval_tokenizer(to_evaluate, return_tensors="pt").to(
    "cuda")

model.eval()
with torch.no_grad():
    print("Output:",eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)




