import csv
from dataclasses import dataclass
from typing import List, Union
import tqdm
from preprocess.fetch import CratesIOCSVPath, dump_crate_io
import datetime
import md2txt
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
    dependencies: List[str]
    keywords: List[str]
    category_indices: List[int]
    processed: bool = False

    def processed_string(self):
        assert self.processed
        return " ".join(["description: ", self.description, "readme: ", self.readme])

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

    def get_label_name(self, pos: int):
        return self._list[pos].name

class CratesData:
    def __init__(self, force_download = False) -> None:
        paths = dump_crate_io(force_download)
        self.categories = Categories(paths)
        self.id2crates: dict[str, Crate] = {}
        self.id2idx = {}
        self.name2crates = {}
        self.crates: list[Crate] = []
        csv.field_size_limit(100000000)
        with open(paths.crates, 'r', encoding='utf-8') as f:
            raw_crates = list(csv.DictReader(f))
        for crate in tqdm.tqdm(raw_crates, desc='Loading crates'):
            crate_id = crate['id']
            name = crate['name']
            description = crate['description']
            readme = crate['readme']
            crate = Crate(crate_id, name, description, readme, [],[],[], False)
            self.id2crates[crate_id] = crate
            self.name2crates[crate.name] = crate
            self.crates.append(crate)
            self.id2idx[crate_id] = len(self.crates) - 1
        
        with open(paths.crates_categories, 'r', encoding='utf-8') as f:
            crates_categories = list(csv.DictReader(f))
        for crate_category in tqdm.tqdm(crates_categories, desc='Merging crates-categories relationship'):
            crate_id = crate_category['crate_id']
            category = crate_category['category_id']
            if crate_id in self.id2crates and category in self.categories._id2idx:
                self.id2crates[crate_id].category_indices.append(self.categories.index(category))

        with open(paths.crates_keywords, 'r', encoding='utf-8') as f:
            crates_keywords = list(csv.DictReader(f))
        with open(paths.keywords, 'r', encoding='utf-8') as f:
            keywords = list(csv.DictReader(f))
        print(f"Loading Keywords")
        id2keyword = {k['id']: k['keyword'] for k in keywords}
        for crate_keyword in tqdm.tqdm(crates_keywords, desc='Merging crates-keywords relationship'):
            crate_id = crate_keyword['crate_id']
            keyword_id = crate_keyword['keyword_id']
            if crate_id in self.id2crates:
                self.id2crates[crate_id].keywords.append(id2keyword[keyword_id])

        print(f"Loading version ids")
        version_id_to_crate_id = {}
        crate_id_to_latest_version_id = {}
        crate_id_to_time = {}
        with open(paths.versions, 'r', encoding='utf-8') as f:
            for version in tqdm.tqdm(csv.DictReader(f)):
                crate_id = version['crate_id']
                version_id = version['id']
                update_time_str = version['updated_at'] # example: 2022-10-07 23:40:17.141
                update_time = datetime.datetime.strptime(update_time_str, '%Y-%m-%d %H:%M:%S.%f')
                if crate_id in crate_id_to_latest_version_id:
                    if update_time > crate_id_to_time[crate_id]:
                        old_version_id = crate_id_to_latest_version_id[crate_id]
                        crate_id_to_latest_version_id[crate_id] = version_id
                        crate_id_to_time[crate_id] = update_time
                        del version_id_to_crate_id[old_version_id]
                    else:
                        continue
                version_id_to_crate_id[version_id] = crate_id
                crate_id_to_latest_version_id[crate_id] = version_id
                crate_id_to_time[crate_id] = update_time
        del crate_id_to_time
        del crate_id_to_latest_version_id
        with open(paths.dependencies, 'r', encoding='utf-8') as f:
            for dependency in tqdm.tqdm(csv.DictReader(f), desc='Merging dependencies'):
                dependency_crate_id = dependency['crate_id']
                version_id = dependency['version_id']
                if version_id in version_id_to_crate_id:
                    crate_id = version_id_to_crate_id[version_id]
                    if crate_id in self.id2crates and dependency_crate_id in self.id2crates:
                        self.id2crates[crate_id].dependencies.append(self.id2crates[dependency_crate_id].name)
            for crate in self.id2crates.values():
                crate.dependencies = list(set(crate.dependencies))
            

    def remove_no_category_(self):
        for crate in list(self.id2crates.values()):
            if len(crate.category_indices) == 0:
                del self.id2crates[crate.id]
                del self.name2crates[crate.name]
    
    def remove_with_category_(self):
        for crate in list(self.id2crates.values()):
            if len(crate.category_indices) > 0:
                del self.id2crates[crate.id]
                del self.name2crates[crate.name]

    def pre_normalize_(self):
        print('Pre-normalizing using Rust Backend')
        unprocessed_readme = [crate.readme for crate in self.id2crates.values()]
        processed_readme = md2txt.batch_markdown_to_text(unprocessed_readme)
        unprocessed_desc = [crate.description for crate in self.id2crates.values()]
        processed_desc = md2txt.batch_normalize_text_simple(unprocessed_desc)
        for crate, readme, desc in zip(self.id2crates.values(), processed_readme, processed_desc):
            crate.readme = readme
            crate.description = desc
            crate.processed = True
        print('Done Pre-normalizing')

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



  
