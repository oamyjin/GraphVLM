import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoProcessor
import torch
from peft import AutoPeftModelForCausalLM

# root = ''
# output = '/scratch/jl11523/projects/Qwen-VL/eval-output'
dataset_name = 'Movies'

img_path = 'https://images-na.ssl-images-amazon.com/images/I/31V3NfjNACL.jpg'
# Promots
candidates = {
    "Movies": "Movies or Genre for Featured Categories or Studio Specials or Musicals & Performing Arts or A&E Home Video or TV or Science Fiction & Fantasy or Boxed Sets or Walt Disney Studios Home Entertainment or Paramount Home Entertainment or Blu-ray or Art House & International or Criterion Collection or Holidays & Seasonal or Music Artists or BBC or Fully Loaded DVDs or Independently Distributed or HBO or Classics"
}
prompts = [f'Which category does the product seem to belong to: {candidates[dataset_name]}?', \
            f"Given the information of the product: [TEXT_INPUT] . Which category does the product seem to belong to? Choose from the following options: {candidates[dataset_name]}."]
question = prompts[0]


torch.manual_seed(1234)

local_model = '/scratch/jl11523/projects/Qwen-VL/local_model/Qwen-VL'
path_to_adapter = '/scratch/jl11523/projects/Qwen-VL/local_model/finetuned/Movies_finetuned'

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, 
    device_map="auto", 
    trust_remote_code=True
).eval()


# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(local_model, device_map="cuda", trust_remote_code=True, fp16=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained(local_model, trust_remote_code=True)
query = tokenizer.from_list_format([
    {'image': 'https://images-na.ssl-images-amazon.com/images/I/51Z298XNMGL.jpg'},
    {'text': question},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred[0], skip_special_tokens=False)
print("response:", response)
# <img>http://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> playing with<ref> her dog</ref><box>(217,423),(582,897)</box> on the beach<|endoftext|>

# comments: response就是把iamge+text重新说了一遍