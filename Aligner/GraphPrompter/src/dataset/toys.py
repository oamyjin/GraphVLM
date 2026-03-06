import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np
import pdb

# Promots
candidates = {
    "Movies": ['Fully Loaded DVDs', 'Musicals & Performing Arts', 'TV', 'Holidays & Seasonal', 'Classics', 'Science Fiction & Fantasy', 'Walt Disney Studios Home Entertainment', 'Genre for Featured Categories', 'Paramount Home Entertainment', 'Boxed Sets', 'Blu-ray', 'BBC', 'Independently Distributed', 'HBO', 'Music Artists', 'Movies', 'Art House & International', 'Studio Specials', 'A&E Home Video', 'Criterion Collection'],
    "Toys": ['Novelty & Gag Toys', 'Baby & Toddler Toys', 'Dolls & Accessories', 'Building Toys', 'Action Figures & Statues', 'Learning & Education', 'Arts & Crafts', 'Tricycles, Scooters & Wagons', 'Hobbies', 'Stuffed Animals & Plush Toys', 'Toy Remote Control & Play Vehicles', 'Dress Up & Pretend Play', 'Games', 'Sports & Outdoor Play', "Kids' Electronics", 'Grown-Up Toys', 'Party Supplies', 'Puzzles'],
    "Grocery": ['Dried Beans, Grains & Rice', 'Canned, Jarred & Packaged Foods', 'Pasta & Noodles', 'Food & Beverage Gifts', 'Candy & Chocolate', 'Condiments & Salad Dressings', 'Produce', 'Sauces, Gravies & Marinades', 'Dairy, Cheese & Eggs', 'Beverages', 'Soups, Stocks & Broths', 'Frozen', 'Herbs, Spices & Seasonings', 'Fresh Flowers & Live Indoor Plants', 'Cooking & Baking', 'Breads & Bakery', 'Meat & Seafood', 'Jams, Jellies & Sweet Spreads', 'Snack Foods', 'Breakfast Foods'],
    "Reddit": ['foodporn', 'carporn', 'crossstitch', 'rabbits', 'crafts', 'interestingasfuck', 'succulents', 'fountainpens', 'cats', 'blackcats', 'dogpictures', 'hiking', 'woodworking', 'crochet', 'cozyplaces', 'eyebleach', 'guineapigs', 'mycology', 'baking', 'thriftstorehauls', 'earthporn', 'aquariums', 'houseplants', 'food', 'germanshepherds', 'pitbulls', 'embroidery', 'sneakers', 'abandonedporn', 'beerporn', 'gardening', 'photocritique', 'guns', 'breadit', 'corgi', 'cityporn', 'rarepuppers', 'watches', 'cactus', 'beardeddragons', 'mechanicalkeyboards', 'pics', 'natureisfuckinglit', 'itookapicture', 'bettafish', 'knives', 'mildlyinteresting', 'battlestations', 'plants', 'bicycling']
} 
# candidates_str = '\n'.join(candidates[dataset_name])

# question = f'Which category does the product seem to belong to? Choose from the following options: {candidates_str}.'

class ToysDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        # self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        
        candidates_str = ', '.join(candidates["Toys"])
        # if reddit: product -> post
        self.prompt = f"Which category does the product seem to belong to? Choose from the following options: {candidates_str}.\n\nAnswer:"
        # pdb.set_trace()
        self.graph_type = 'Text Attributed Graph'

        self.num_features = 768 * 2 # modify this for different embedding .pt files
        self.num_classes = 18
        print(f'label mapping: {self.graph.label_texts}')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['../../datasets/Toys/Toys_toy_ori_text_aug_imgText_ori_img_graph_data.pt']

    def get_idx_split(self):    
        json_path = "../../datasets/Toys/Toys_split.json"
        with open(json_path, 'r') as file:
            loaded_data_dict = json.load(file)

        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]

        print(f"Loaded data from {json_path}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")

        return {'train': train_ids, 'test': test_ids, 'val': val_ids}




if __name__ == '__main__':
    dataset = ToysDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
