import csv
from dataclasses import dataclass
from typing import Union
import tqdm
import markdown
from bs4 import BeautifulSoup
from preprocess.fetch import CratesIOCSVPath, dump_crate_io
@dataclass
class Category:
    id: str
    name: str
    description: str

@dataclass
class Crate:
    id: str
    name: str
    description: str
    readme: str
    category_indices: list[int]

class Categories:
    def __init__(self, paths: CratesIOCSVPath) -> None:
        categories = list(csv.DictReader(open(paths.categories, 'r', encoding='utf-8')))
        self._list = [Category(c['id'], c['category'], c['description']) for c in categories]
        self._id2idx = {c.id: i for i, c in enumerate(self._list)}
        self._id2item = {c.id: c for c in self._list}
    
    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, int):
            return self._list[item]
        elif isinstance(item, str):
            return self._id2item[item]
        else:
            raise TypeError(f'Expected int or str, got {type(item)}')

    def __len__(self):
        return len(self._list)

    def index(self, id: str) -> int:
        return self._id2idx[id]

class CratesData:
    def __init__(self, force_download = False) -> None:
        paths = dump_crate_io(force_download)
        self.categories = Categories(paths)
        self.id2crates = {}
        self.name2crates = {}
        csv.field_size_limit(100000000)
        with open(paths.crates, 'r', encoding='utf-8') as f:
            raw_crates = list(csv.DictReader(f))
        for crate in tqdm.tqdm(raw_crates, desc='Loading crates'):
            crate_id = crate['id']
            name = crate['name']
            description = crate['description']
            readme = crate['readme']
            crate = Crate(crate_id, name, description, readme, [])
            self.id2crates[crate_id] = crate
            self.name2crates[crate.name] = crate
        
        with open(paths.crates_categories, 'r', encoding='utf-8') as f:
            crates_categories = list(csv.DictReader(f))
        for crate_category in tqdm.tqdm(crates_categories, desc='Merging crates-categories relationship'):
            crate_id = crate_category['crate_id']
            category = crate_category['category_id']
            if crate_id in self.id2crates and category in self.categories._id2idx:
                self.id2crates[crate_id].category_indices.append(self.categories.index(category))

    def remove_no_category_(self):
        for crate in list(self.id2crates.values()):
            if len(crate.category_indices) == 0:
                del self.id2crates[crate.id]
                del self.name2crates[crate.name]

    def process_readme_(self):
        # convert markdown to a list of words
        for crate in tqdm.tqdm(self.id2crates.values(), desc='Processing readme'):
            html = markdown.markdown(crate.readme)
            text = BeautifulSoup(html, 'html.parser').get_text()
            text = text.replace("\n", " ")
            crate.readme = text

    def all_crates(self):
        return self.id2crates.values()

    def __getitem__(self, name: str) -> Crate:
        return self.name2crates[name]

    def __len__(self):
        return len(self.id2crates)

    def __iter__(self):
        return iter(self.id2crates.values())

    def get_from_id(self, id: str) -> Crate:
        return self.id2crates[id]

    def num_categories(self) -> int:
        return len(self.categories)



  
