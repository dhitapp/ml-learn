from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import requests
import random

DATASET_MAP: dict = {
    'COCO': 'yerevann/coco-karpathy',
    'flickr8k': 'jxie/flickr8k'
}

class ContrastiveDataset(Dataset):
    def __init__(self, split: str  = 'train', sources: list = ['flickr8k']):
        super().__init__()
        self.sources = sources
        self.split = split
        self.load()

    def __len__(self):
        return len(self.data)

    def preprocess_data(self, raw_data):
        data = []
        for source in raw_data:
            if list(source.keys())[0] == 'COCO':
                for row in source['COCO'][self.split]:
                        data.append({
                            'imgid': row['imgid'],
                            'filename': row['filename'],
                            'url': row['url'],
                            'text': row['sentences'][0],
                            'source': 'COCO'
                        })
            else:
                for row in source['flickr8k'][self.split]:
                    k = random.randrange(5)
                    data.append({
                        'img': row['image'],
                        'text': row[f'caption_{k}'],
                        'source': 'flickr8k'
                    })
        return data
    
    def load(self):
        raw_data = []
        for source in self.sources:
            if source in DATASET_MAP:
                raw_data.append({
                    source: load_dataset(DATASET_MAP[source])
                })
        self.data = self.preprocess_data(raw_data)

    def __getitem__(self, idx):
        source = self.data[idx]['source']
        if source == 'COCO':
            url = self.data[idx]['url']
            image = Image.open(requests.get(url, stream=True).raw)
        else:
            image = self.data[idx]['img']
        text = self.data[idx]['text']
        image = image.convert('RGB')
        return image, text

