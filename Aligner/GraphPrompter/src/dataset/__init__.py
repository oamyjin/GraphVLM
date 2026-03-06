from src.dataset.cora import CoraSupDataset, CoraSemiDataset
from src.dataset.citeseer import CiteseerDataset
from src.dataset.pubmed import PubmedSupDataset,PubmedSemiDataset
from src.dataset.arxiv import ArxivSupDataset, ArxivSemiDataset
from src.dataset.products import ProductsSupDataset, ProductsSemiDataset
from src.dataset.sports import SportsSemiDataset, SportsSupDataset
from src.dataset.photo import PhotoSemiDataset, PhotoSupDataset
from src.dataset.computers import ComputersSemiDataset, ComputersSupDataset

from src.dataset.movies import MoviesDataset
from src.dataset.grocery import GroceryDataset
from src.dataset.toys import ToysDataset
from src.dataset.reddit import RedditDataset
from src.dataset.toys_aug import ToysAugDataset
from src.dataset.arts import ArtsDataset
from src.dataset.movies_aug import MoviesAugDataset
from src.dataset.cd import CDDataset
from src.dataset.videogame import VideoGameDataset
from src.dataset.grocery_aug import GroceryAugDataset
from src.dataset.arts_aug import ArtsAugDataset
from src.dataset.cd_aug import CDAugDataset

load_dataset = {
    'movies': MoviesDataset,
    'grocery': GroceryDataset,
    'toys': ToysDataset,
    'reddit': RedditDataset,
    'toys_aug': ToysAugDataset,
    'arts': ArtsDataset,
    'movies_aug': MoviesAugDataset,
    'cd': CDDataset,
    'videogame': VideoGameDataset,
    'grocery_aug': GroceryAugDataset,
    'arts_aug': ArtsAugDataset,
    'cd_aug': CDAugDataset,

    'cora_sup': CoraSupDataset,
    'pubmed_sup': PubmedSupDataset,
    'arxiv_sup': ArxivSupDataset,
    'products_sup': ProductsSupDataset,
    'cora_semi': CoraSemiDataset,
    'pubmed_semi': PubmedSemiDataset,
    'arxiv_semi': ArxivSemiDataset,
    'products_semi': ProductsSemiDataset,
    'citeseer': CiteseerDataset,
    'sports_semi': SportsSemiDataset,
    'sports_sup': SportsSupDataset,
    'photo_semi': PhotoSemiDataset,
    'photo_sup': PhotoSupDataset,
    'computers_semi': ComputersSemiDataset,
    'computers_sup': ComputersSupDataset,
}
