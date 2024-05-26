from huggingface_hub import login
login()



#
from datasets import load_dataset

dataset = load_dataset('Unbabel/TowerBlocks-v0.2', split='train')
train = dataset.filter(lambda example: example['split']=="train" and example['task']=="machine_translation")
dev = dataset.filter(lambda example: example['split']=="dev" and example['task']=="machine_translation")
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
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

#And convert each sample into a prompt that I found from this [notebook](https://github.com/samlhuillier/viggo-finetune/blob/main/llama/fine-tune-code-llama.ipynb).
def generate_prompt(instruction,response,  sep="\n\n### "):  #The prompt format is taken from the official Mixtral huggingface page
    p =  "<s> [INST]" + instruction + "[/INST]" +  response + "</s>"
    return p

def generate_and_tokenize_prompt(data_point):
    messages = data_point["conversations"]
    if len(messages) > 2:
        raise Exception("Too many messages")
    full_prompt = generate_prompt(messages[0]["value"], messages[1]["value"])
    return tokenize(full_prompt)

#Reformat the prompt and tokenize each sample:
tokenized_train_dataset = train.map(generate_and_tokenize_prompt)
tokenized_val_dataset = dev.map(generate_and_tokenize_prompt)

#Check that input_ids is padded on the left with the eos_token (2) and there is an eos_token 2 added to the end, and the prompt starts with a bos_token (1).
print(tokenized_train_dataset[4]['input_ids'])
#Check that a sample has the max length, i.e. 512.
print(len(tokenized_train_dataset[4]['input_ids']))

#How does the base model do?
#Let's grab a test input (meaning_representation) and desired output (target) pair to see how the base model does on it.
print("Target Sentence: " + dev[1]['conversations'][0]['value'])
print("Meaning Representation: " + dev[1]['conversations'][1]['value'] + "\n")
#
# eval_prompt = """Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
# This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
# The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
#
# ### Target sentence:
# Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?
#
# ### Meaning representation:
# """
# # Re-init the tokenizer so it doesn't add padding or eos token
# eval_tokenizer = AutoTokenizer.from_pretrained(
#     base_model_id,
#     add_bos_token=True,
# )
#
# model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#
# model.eval()
# with torch.no_grad():
#     print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))
#
#
