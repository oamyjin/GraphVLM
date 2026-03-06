import json
import os.path as osp
import os
import torch as th
import re
import pandas as pd
from tqdm import tqdm 
import re


path = "/scratch/jl11523/projects/LLaVA/eval-results/"
dataset_name = "CD" # Movies, Toys, Grocery, Reddit, Arts
dataset = dataset_name.lower() + "/"
folders = [path + dataset + f for f in ["onlyimage", 'imagetext']]
output_data_save = []
output_path = path + dataset
output_file = output_path + "eval_results_" + dataset_name + ".json"

for folder in folders:
    data_list = []
    correct_idx_list = []
    wrong_idx_list = []
    print(folder.split("/")[-1])
    print("folder:", folder)
    for filename in os.listdir(folder):
        if filename.endswith('.json'): 
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list.extend(data)
    # print("result demo:", data_list[1])

    with open('/scratch/jl11523/projects/LLaVA/dataset/true_labels_'+dataset_name+'.csv', 'r') as f:
        labels = json.load(f)
    # print("len(labels):", len(labels))

    correct = 0
    actual_total = len(data_list)

    candidates = {
        "Movies": ['Fully Loaded DVDs', 'Musicals & Performing Arts', 'TV', 'Holidays & Seasonal', 'Classics', 'Science Fiction & Fantasy', 'Walt Disney Studios Home Entertainment', 'Genre for Featured Categories', 'Paramount Home Entertainment', 'Boxed Sets', 'Blu-ray', 'BBC', 'Independently Distributed', 'HBO', 'Music Artists', 'Movies', 'Art House & International', 'Studio Specials', 'A&E Home Video', 'Criterion Collection'],
        "Toys": ['Novelty & Gag Toys', 'Baby & Toddler Toys', 'Dolls & Accessories', 'Building Toys', 'Action Figures & Statues', 'Learning & Education', 'Arts & Crafts', 'Tricycles, Scooters & Wagons', 'Hobbies', 'Stuffed Animals & Plush Toys', 'Toy Remote Control & Play Vehicles', 'Dress Up & Pretend Play', 'Games', 'Sports & Outdoor Play', "Kids' Electronics", 'Grown-Up Toys', 'Party Supplies', 'Puzzles'],
        "Grocery": ['Dried Beans, Grains & Rice', 'Canned, Jarred & Packaged Foods', 'Pasta & Noodles', 'Food & Beverage Gifts', 'Candy & Chocolate', 'Condiments & Salad Dressings', 'Produce', 'Sauces, Gravies & Marinades', 'Dairy, Cheese & Eggs', 'Beverages', 'Soups, Stocks & Broths', 'Frozen', 'Herbs, Spices & Seasonings', 'Fresh Flowers & Live Indoor Plants', 'Cooking & Baking', 'Breads & Bakery', 'Meat & Seafood', 'Jams, Jellies & Sweet Spreads', 'Snack Foods', 'Breakfast Foods'],
        "Reddit": ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling'],
        "Arts": ['Knitting & Crochet', 'Beading & Jewelry Making', 'Painting, Drawing & Art Supplies', 'Crafting', 'Model & Hobby Building', 'Sewing', 'Scrapbooking & Stamping'],
        "CD": ['Jazz', 'Pop', 'International Music', 'Indie & Alternative', 'Blues', 'Classic Rock', 'Rock', 'Metal', 'Dance & Electronic', 'Christian & Gospel', 'Today\'s Deals in Music', 'R&B', 'Country', 'Rap & Hip-Hop', 'Classical']
    } 

    pattern = {}
    for c in candidates:
        pattern[c] = r'\b(' + '|'.join(candidates[c]) + r')\b'

    # pattern = {
    #     "Movies": r'\b(Movies|Genre for Featured Categories|TV|Classics|Boxed Sets|Blu-ray|Independently Distributed|Holidays & Seasonal|A&E Home Video|Fully Loaded DVDs|Musicals & Performing Arts|Criterion Collection|BBC|Art House & International|Walt Disney Studios Home Entertainment|HBO|Studio Specials|Science Fiction & Fantasy|Music Artists|Paramount Home Entertainment)\b',
    #     "Toys": r'\b(Novelty & Gag Toys|Baby & Toddler Toys|Dolls & Accessories|Building Toys|Action Figures & Statues|Learning & Education|Arts & Crafts|Tricycles & Scooters & Wagons|Hobbies|Stuffed Animals & Plush Toys|Toy Remote Control & Play Vehicles|Dress Up & Pretend Play|Games|Sports & Outdoor Play|Kids\' Electronics|Grown-Up Toys|Party Supplies|Puzzles)\b',
    #     "Grocery":r'\b(Dried Beans & Grains & Rice|Canned & Jarred & Packaged Foods|Pasta & Noodles|Food & Beverage Gifts|Candy & Chocolate|Condiments & Salad Dressings|Produce|Sauces & Gravies & Marinades|Dairy & Cheese & Eggs|Beverages|Soups & Stocks & Broths|Frozen|Herbs & Spices & Seasonings|Fresh Flowers & Live Indoor Plants|Cooking & Baking|Breads & Bakery|Meat & Seafood|Jams & Jellies & Sweet Spreads|Snack Foods|Breakfast Foods)\b'
    # }

    all_node_ids = []
    actual_total = 0
    for entry in tqdm(data_list): 
        # node_idx = entry["node_idx"]
        node_idx = entry["id"]
        if node_idx in all_node_ids:
            continue
        else:
            all_node_ids.append(node_idx)
            actual_total += 1

        #print("id:", node_idx)
        prediction = entry["res"]
        # print("node_idx:", node_idx)
        # print("  prediction:", prediction)

        # 使用 re.findall() 方法查找所有匹配项，忽略大小写
        matches = list(set(re.findall(pattern[dataset_name], prediction, re.IGNORECASE))) 
        sorted_matches = sorted(matches, key=lambda x: prediction.index(x)) # 按照出现顺序排序

        label = labels[node_idx][str(node_idx)]#.replace("_", " ") # Replace underscore with space
        
        # print("  sorted_matches:", sorted_matches)
        if len(sorted_matches) == 0:
            wrong_idx_list.append(node_idx)
            continue
        if label.lower().strip() == sorted_matches[0].lower().strip():
            correct += 1
            correct_idx_list.append(node_idx)
        else:
            wrong_idx_list.append(node_idx)
            # todo: maybe we can check those correct predictions are accurate or not

    if actual_total == 0:
        acc = 0
    else:
        acc = correct / actual_total

    output_data = {}
    output_data["folder"] = folder.split("/")[-1]
    output_data["correct_idx_list"] = correct_idx_list
    output_data["wrong_idx_list"] = wrong_idx_list
    output_data["Accuracy"] = acc
    output_data["actual_total"] = actual_total
    output_data_save.append(output_data)
    # print("correct_idx_list:", correct_idx_list)
    # print("wrong_idx_list:", wrong_idx_list)
    print("Accuracy:", acc)
    print("actual_total:", actual_total)
    print()

# Save the output data
with open(output_file, "w") as file:
    json.dump(output_data_save, file)  # 'indent=4' for pretty-printing
    
    
    # report = classification_report(trues, preds, digits=6)
    # print(report)



def eval_amazon_computers_nc(res_path):
    data=torch.load("dataset/amazon-computers_semi/processed_data.pt")
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error.append((ans, label))
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
