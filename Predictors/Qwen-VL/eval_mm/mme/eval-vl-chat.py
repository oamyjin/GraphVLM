import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoProcessor
import torch
from datasets import load_dataset
import numpy as np
import ast
import json
import argparse
import requests
import re
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/jl11523/projects/Qwen-VL/local_model/Qwen-VL-Chat")
    parser.add_argument("--path_to_adapter", type=str, default=None)
    parser.add_argument("--cvs-file", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--use-image", type=int, default=1)
    parser.add_argument("--use-text", type=int, default=1)
    parser.add_argument("--use-nb-images", type=int, default=0)
    parser.add_argument("--use-nb-titles", type=int, default=0)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    
    dataset_name = args.cvs_file.split("/")[-1].split(".")[0]
    checkpoint = args.model_path
    print("dataset_name:", dataset_name)
    print("local_model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    print("Tokenizer done.")

    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
    if args.path_to_adapter != None:
        model.load_adapter(args.path_to_adapter)
        print(f"Model from path_to_adapter: {args.path_to_adapter}")

    print("Model done")
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
    print("Model.generation_config done")
    model.generation_config.top_p = 0.01 # TODO: What is 0.01???
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    print("processor done")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("device:", device)


    # read the splits
    with open('/scratch/jl11523/projects/LLaVA/dataset/splits_id/' + dataset_name + '.json', 'r') as f:
        splits = json.load(f)
    test_ids = splits['test']
    print("len(test_ids):", len(test_ids))
    total_num = len(test_ids)

    # load prompt-file
    with open(args.prompt_file, "r") as file:
        prompt_data = json.load(file)
    
    # enable tqdm
    output_json = []
    for idx in tqdm(range(args.start_idx, len(test_ids)), desc="Processing", unit="item"):
        test_id = test_ids[idx]
        query_items = []

        prompt = prompt_data[str(test_id)]

        img_tag_pattern = r"<img>(.*?)</img>"
        image_paths = re.findall(img_tag_pattern, prompt)
        # images = [Image.open(img_path) for img_path in image_paths]  # 加载图像
        # inputs = processor(images=images, text=prompt, return_tensors="pt")
        # inputs = {key: value.to(device) for key, value in inputs.items()}  # 确保 inputs 移动到设备上
        #outputs = model.generate(**inputs)
        #response = processor.decode(outputs[0], skip_special_tokens=True)

        query_items = [{'image': img_path} for img_path in image_paths]
        query_items.append({'text': prompt})
        query = tokenizer.from_list_format(query_items)
        response, _ = model.chat(tokenizer, query=query, history=None)

        output_data = {}
        output_data["id"] = test_id
        output_data["node_idx"] = test_id
        output_data["res"] = response
        output_json.append(output_data)
        # save the output
        with open(args.output_file, "w") as file:
            json.dump(output_json, file, indent=4)  # 'indent=4' for pretty-printing
       
        print(f"[{idx}/{total_num}]")
        if len(output_json) == 1:
            print("Here is the first data to evaluate:")
            print(" prompt:", prompt)
            print(" image_paths:", image_paths)
            print(" query_items:", query_items)
            print(" output_data:", output_data["res"])
            print(" len(prompt):", len(prompt))

    print("len(output_json):", len(output_json))
    print(f"Saved to {args.output_file}")