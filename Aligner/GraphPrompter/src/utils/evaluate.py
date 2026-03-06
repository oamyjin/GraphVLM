import json
import pandas as pd
import re
import argparse


def get_accuracy_videogame(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Legacy Systems', 'PC', 'Nintendo Switch', 'PlayStation 4', 'Xbox One', 'PlayStation 5']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_cd(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Pop', "Today's Deals in Music", 'Rock', 'Indie & Alternative', 'Classic Rock', 'Country', 'International Music', 'Jazz', 'Metal', 'R&B', 'Classical', 'Rap & Hip-Hop', 'Christian & Gospel', 'Blues', 'Dance & Electronic']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_cd_aug(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Pop', "Today's Deals in Music", 'Rock', 'Indie & Alternative', 'Classic Rock', 'Country', 'International Music', 'Jazz', 'Metal', 'R&B', 'Classical', 'Rap & Hip-Hop', 'Christian & Gospel', 'Blues', 'Dance & Electronic']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_movies_aug(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Fully Loaded DVDs', 'Musicals & Performing Arts', 'TV', 'Holidays & Seasonal', 'Classics', 'Science Fiction & Fantasy', 'Walt Disney Studios Home Entertainment', 'Genre for Featured Categories', 'Paramount Home Entertainment', 'Boxed Sets', 'Blu-ray', 'BBC', 'Independently Distributed', 'HBO', 'Music Artists', 'Movies', 'Art House & International', 'Studio Specials', 'A&E Home Video', 'Criterion Collection']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_arts(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    # classes = ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling']
    classes = ['Knitting & Crochet', 'Beading & Jewelry Making', 'Painting, Drawing & Art Supplies', 'Crafting', 'Model & Hobby Building', 'Sewing', 'Scrapbooking & Stamping']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_arts_aug(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    # classes = ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling']
    classes = ['Knitting & Crochet', 'Beading & Jewelry Making', 'Painting, Drawing & Art Supplies', 'Crafting', 'Model & Hobby Building', 'Sewing', 'Scrapbooking & Stamping']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_reddit(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes =  ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_toys_aug(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes =  ['Novelty & Gag Toys', 'Baby & Toddler Toys', 'Dolls & Accessories', 'Building Toys', 'Action Figures & Statues', 'Learning & Education', 'Arts & Crafts', 'Tricycles, Scooters & Wagons', 'Hobbies', 'Stuffed Animals & Plush Toys', 'Toy Remote Control & Play Vehicles', 'Dress Up & Pretend Play', 'Games', 'Sports & Outdoor Play', "Kids' Electronics", 'Grown-Up Toys', 'Party Supplies', 'Puzzles']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_toys(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes =  ['Novelty & Gag Toys', 'Baby & Toddler Toys', 'Dolls & Accessories', 'Building Toys', 'Action Figures & Statues', 'Learning & Education', 'Arts & Crafts', 'Tricycles, Scooters & Wagons', 'Hobbies', 'Stuffed Animals & Plush Toys', 'Toy Remote Control & Play Vehicles', 'Dress Up & Pretend Play', 'Games', 'Sports & Outdoor Play', "Kids' Electronics", 'Grown-Up Toys', 'Party Supplies', 'Puzzles']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_grocery(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Dried Beans, Grains & Rice', 'Canned, Jarred & Packaged Foods', 'Pasta & Noodles', 'Food & Beverage Gifts', 'Candy & Chocolate', 'Condiments & Salad Dressings', 'Produce', 'Sauces, Gravies & Marinades', 'Dairy, Cheese & Eggs', 'Beverages', 'Soups, Stocks & Broths', 'Frozen', 'Herbs, Spices & Seasonings', 'Fresh Flowers & Live Indoor Plants', 'Cooking & Baking', 'Breads & Bakery', 'Meat & Seafood', 'Jams, Jellies & Sweet Spreads', 'Snack Foods', 'Breakfast Foods']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_grocery_aug(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Dried Beans, Grains & Rice', 'Canned, Jarred & Packaged Foods', 'Pasta & Noodles', 'Food & Beverage Gifts', 'Candy & Chocolate', 'Condiments & Salad Dressings', 'Produce', 'Sauces, Gravies & Marinades', 'Dairy, Cheese & Eggs', 'Beverages', 'Soups, Stocks & Broths', 'Frozen', 'Herbs, Spices & Seasonings', 'Fresh Flowers & Live Indoor Plants', 'Cooking & Baking', 'Breads & Bakery', 'Meat & Seafood', 'Jams, Jellies & Sweet Spreads', 'Snack Foods', 'Breakfast Foods']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_movies(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Fully Loaded DVDs', 'Musicals & Performing Arts', 'TV', 'Holidays & Seasonal', 'Classics', 'Science Fiction & Fantasy', 'Walt Disney Studios Home Entertainment', 'Genre for Featured Categories', 'Paramount Home Entertainment', 'Boxed Sets', 'Blu-ray', 'BBC', 'Independently Distributed', 'HBO', 'Music Artists', 'Movies', 'Art House & International', 'Studio Specials', 'A&E Home Video', 'Criterion Collection']
    # classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

def get_accuracy_cora(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_pubmed(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_citeseer(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_arxiv(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row)) + '\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        print(f'prediction: {pred}')

        # Remove everything after the first open parenthesis (if any) for cleaner matching
        clean_pred = re.sub(r'\(.*\)', '', pred.strip())
        clean_label = re.sub(r'\(.*\)', '', label.strip())
        print(clean_label)
        matches = re.findall(r"cs\.[a-zA-Z]{2}", clean_pred)

        if len(matches) > 0 and clean_label == matches[0]:
            correct += 1
            print('correct')
        print(f'gt: {clean_label}')
        print('\n')
    print(len(df))
    return correct / len(df)


def get_accuracy_sports(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_computers(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_photo(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_products(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Home & Kitchen',
               'Health & Personal Care',
               'Beauty',
               'Sports & Outdoors',
               'Books',
               'Patio, Lawn & Garden',
               'Toys & Games',
               'CDs & Vinyl',
               'Cell Phones & Accessories',
               'Grocery & Gourmet Food',
               'Arts, Crafts & Sewing',
               'Clothing, Shoes & Jewelry',
               'Electronics',
               'Movies & TV',
               'Software',
               'Video Games',
               'Automotive',
               'Pet Supplies',
               'Office Products',
               'Industrial & Scientific',
               'Musical Instruments',
               'Tools & Home Improvement',
               'Magazine Subscriptions',
               'Baby Products',
               'NaN',
               'Appliances',
               'Kitchen & Dining',
               'Collectibles & Fine Art',
               'All Beauty',
               'Luxury Beauty',
               'Amazon Fashion',
               'Computers',
               'All Electronics',
               'Purchase Circles',
               'MP3 Players & Accessories',
               'Gift Cards',
               'Office & School Supplies',
               'Home Improvement',
               'Camera & Photo',
               'GPS & Navigation',
               'Digital Music',
               'Car Electronics',
               'Baby',
               'Kindle Store',
               'Buy a Kindle',
               'Furniture & Decor',
               '#508510']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)

eval_funcs = {
    'movies': get_accuracy_movies,
    'grocery': get_accuracy_grocery,
    'toys': get_accuracy_toys,
    'reddit': get_accuracy_reddit,
    'toys_aug': get_accuracy_toys_aug,
    'arts': get_accuracy_arts,
    'movies_aug': get_accuracy_movies_aug,
    'cd': get_accuracy_cd,
    'cd_aug': get_accuracy_cd_aug,
    'videogame': get_accuracy_videogame,
    'grocery_aug': get_accuracy_grocery_aug,
    'arts_aug': get_accuracy_arts_aug,
    
    'cora_sup': get_accuracy_cora,
    'citeseer': get_accuracy_citeseer,
    'pubmed_sup': get_accuracy_pubmed,
    'arxiv_sup': get_accuracy_arxiv,
    'products_sup': get_accuracy_products,
    'cora_semi': get_accuracy_cora,
    'pubmed_semi': get_accuracy_pubmed,
    'arxiv_semi': get_accuracy_arxiv,
    'products_semi': get_accuracy_products,
    "sports_semi": get_accuracy_sports,
    "sports_sup": get_accuracy_sports,
    "computers_semi": get_accuracy_computers,
    "computers_sup": get_accuracy_computers,
    "photo_semi": get_accuracy_photo,
    "photo_sup": get_accuracy_photo,
}