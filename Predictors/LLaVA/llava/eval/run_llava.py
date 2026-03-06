import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


import requests
from PIL import Image
from io import BytesIO
import re

from datasets import load_dataset
import numpy as np
import ast
import json
import time
import threading

import torch
import time
import os
from tqdm import tqdm


def zombie_task():
    # 周期性地进行计算
    while True:
        # 创建更大的张量，并进行多个矩阵运算，增加 GPU 负载
        fake_data1 = torch.randn(8192, 8192, device='cuda')
        fake_data2 = torch.randn(8192, 8192, device='cuda')
        fake_result = torch.mm(fake_data1, fake_data2)
        
        # 进行一些额外的计算，增加占用时间
        for _ in range(10):
            fake_result = torch.sin(fake_result) + torch.cos(fake_result)
            fake_result = torch.mm(fake_result, fake_data1)
        
        # 强制同步，确保计算真正发生
        torch.cuda.synchronize()
        
        # 每隔30秒运行一次，占用较长的时间
        time.sleep(30)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    image_file = image_file.strip()
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        # print("iamge_file:", image_file)
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print("prompt:", prompt)

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print("model.device:", model.device)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    description_input_ids = (
        tokenizer_image_token(args.query_description, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    # print("size(description_input_ids):", description_input_ids.size(1))

    # all_tokens = input_ids.size(1)
    # print("all_tokens:", all_tokens)

    # num_image_tokens = images_tensor.size(1) # length of image tokens
    # print("num_image_tokens:", num_image_tokens)
    
    # # truncate the prompt if it is too long
    # max_length = model.config.max_length
    # print("max_length:", max_length)
    
    # Calculate remaining tokens for other text
    # remaining_tokens = max_length - num_image_tokens - category_tokens
    # if remaining_tokens > 0:
    #     input_ids = input_ids[:, :remaining_tokens]
    # else:
    #     input_ids = input_ids[:, :0]

    # # Concatenate category tokens to the end of the input sequence
    # input_ids = torch.cat([input_ids, category_input_ids['input_ids']], dim=1)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    return outputs


if __name__ == "__main__":
    # # 创建并启动僵尸计算线程
    # zombie_thread = threading.Thread(target=zombie_task)
    # zombie_thread.daemon = True  # 将线程设为守护线程，主任务结束时自动退出
    # zombie_thread.start()

    # main task
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cvs-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--use-image", type=bool, default=True)
    parser.add_argument("--use-text", type=bool, default=False)
    parser.add_argument("--true-label-file", type=str, required=True)
    parser.add_argument("--write-true-label-file", type=bool, required=True)
    parser.add_argument("--image-file", type=str, default="")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--query_question", type=str, default="")
    parser.add_argument("--query_description", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    dataset_name = args.cvs_file.split("/")[-1].split(".")[0]
    print("dataset_name:", dataset_name)
    print("use_image:", args.use_image, ", use_text:", args.use_text)

    args.model_name = get_model_name_from_path(args.model_path) #"/scratch/jl11523/projects/LLaVA/local_model/llava-v1.5-7b"
    
    print(f"Loading {args.cvs_file.split('/')[-1]}...")
    true_labels = []
    image_files = []
    texts = []
    raw_data = load_dataset("csv", data_files=args.cvs_file)
    # print(raw_data)
    for data in raw_data['train']:
        true_label = {}
        if dataset_name == 'Reddit':
            true_label[data['id']] = data['subreddit']
            image_files.append(data['url'])
            texts.append(data['caption'])
        else:
            true_label[data['id']] = data['second_category'] #ast.literal_eval(data['category'])[1]
            if dataset_name != 'Arts' and dataset_name != 'CD':
                image_files.append(ast.literal_eval(data['imageURLHighRes']))
            texts.append(data['text'])
        true_labels.append(true_label)
    total_num = len(true_labels)
    if total_num != len(texts):
        print(f"Error: total_num{total_num} != len(texts){len(texts)}")
    print("total num of tests:", total_num)
    if args.write_true_label_file:
        with open(args.true_label_file, "w") as file:
            json.dump(true_labels, file)

    # Promots
    candidates = {
        "Movies": ['Fully Loaded DVDs', 'Musicals & Performing Arts', 'TV', 'Holidays & Seasonal', 'Classics', 'Science Fiction & Fantasy', 'Walt Disney Studios Home Entertainment', 'Genre for Featured Categories', 'Paramount Home Entertainment', 'Boxed Sets', 'Blu-ray', 'BBC', 'Independently Distributed', 'HBO', 'Music Artists', 'Movies', 'Art House & International', 'Studio Specials', 'A&E Home Video', 'Criterion Collection'],
        "Toys": ['Novelty & Gag Toys', 'Baby & Toddler Toys', 'Dolls & Accessories', 'Building Toys', 'Action Figures & Statues', 'Learning & Education', 'Arts & Crafts', 'Tricycles, Scooters & Wagons', 'Hobbies', 'Stuffed Animals & Plush Toys', 'Toy Remote Control & Play Vehicles', 'Dress Up & Pretend Play', 'Games', 'Sports & Outdoor Play', "Kids' Electronics", 'Grown-Up Toys', 'Party Supplies', 'Puzzles'],
        "Grocery": ['Dried Beans, Grains & Rice', 'Canned, Jarred & Packaged Foods', 'Pasta & Noodles', 'Food & Beverage Gifts', 'Candy & Chocolate', 'Condiments & Salad Dressings', 'Produce', 'Sauces, Gravies & Marinades', 'Dairy, Cheese & Eggs', 'Beverages', 'Soups, Stocks & Broths', 'Frozen', 'Herbs, Spices & Seasonings', 'Fresh Flowers & Live Indoor Plants', 'Cooking & Baking', 'Breads & Bakery', 'Meat & Seafood', 'Jams, Jellies & Sweet Spreads', 'Snack Foods', 'Breakfast Foods'],
        "Reddit": ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling'],
        "Arts": ['Knitting & Crochet', 'Beading & Jewelry Making', 'Painting, Drawing & Art Supplies', 'Crafting', 'Model & Hobby Building', 'Sewing', 'Scrapbooking & Stamping'],
        "CD": ['Jazz', 'Pop', 'International Music', 'Indie & Alternative', 'Blues', 'Classic Rock', 'Rock', 'Metal', 'Dance & Electronic', 'Christian & Gospel', 'Today\'s Deals in Music', 'R&B', 'Country', 'Rap & Hip-Hop', 'Classical']
    } 
    candidates_str = '\n'.join(candidates[dataset_name])
    
    question = f'Which category does the product seem to belong to? Choose from the following options: {candidates_str}.'
    description = f"Given the information of the product: [TEXT_INPUT]."
    # prompts = [question, description + question]
    args.query_question = question
    
    # read the splits
    # '/scratch/jl11523/projects/LLaVA/dataset/splits_id/Movies.json'
    test_id_file = f"/scratch/jl11523/projects/LLaVA/dataset/splits_id/{dataset_name}.json"
    with open(test_id_file, 'r') as f:
        splits = json.load(f)
    test_ids = splits['test']
    print("len(test_ids):", len(test_ids))
    total_num = len(test_ids)

    # iterate the image files to evaluate for each of them
    # idx = #14728 #args.start_idx
    output_json = []
    for idx in tqdm(range(args.start_idx, len(test_ids)), desc="Processing", unit="item"): #len(image_files)):
        i = test_ids[idx] # longest text idx: 11041
        if args.use_image:
            if args.image_folder is not None:
                args.image_file = os.path.join(args.image_folder, f"{i}.jpg")
            else:
                if dataset_name == 'Reddit':
                    if "http://farm3.staticflickr.com/2842/10202410356_7b36827fc6_h.jpg" == image_files[i]:
                        print(f"{i}=test_ids[idx={idx}] http://farm3.staticflickr.com/2842/10202410356_7b36827fc6_h.jpg IS NOT A VALID IAMGE")
                        continue
                    args.image_file = image_files[i]
                else:
                    new_image_files = []
                    for img in image_files[i]:
                        try:
                            response = requests.get(img)
                        except:
                            print(f"{i}=test_ids[idx={idx}] {img} CANNNOT GET RESPONSE")
                            continue
                        if response.status_code != 200:
                            print(f"{i}=test_ids[idx={idx}] {img} IS NOT A VALID IAMGE")
                        else:
                            new_image_files.append(img)
                    if len(new_image_files) == 0:
                        print(f"{i}=test_ids[idx={idx}] NO VALID IAMGE")
                        continue
                    image_file = ",".join(new_image_files)
                    args.image_file = image_file
        if args.use_text:
            # Replace [TEXT_INPUT] with the actual content
            args.query_description = description.replace("[TEXT_INPUT]", texts[i])
            args.query = args.query_description + question # prompts[1].replace("[TEXT_INPUT]", texts[i])
        else:
            args.query = question # prompts[0]
        output_data = {}
        output_data["id"] = i #idx
        output_data["node_idx"] = i #idx
        # evaluate the model
        output_data["res"] = eval_model(args)
        output_json.append(output_data)
        # idx += 1
        if idx == 0 or idx == 1:
            print("args.output_file:", args.output_file)
        with open(args.output_file, "w") as file:
            json.dump(output_json, file, indent=4)  # 'indent=4' for pretty-printing
        print(f"[{idx}/{total_num}]") #idx
        if len(output_json) == 1:
            print("Here is the first data to evaluate:")
            print(" args.query:", args.query)
            print(" args.image_file:", args.image_file)
            print(" output_data:", output_data["res"])
            print(" len(prompt):", len(args.query))

    print("len(output_json):", len(output_json))
