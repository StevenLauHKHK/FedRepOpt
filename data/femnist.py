import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class femnist(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.json_file = json_file
        self.image_paths, self.labels = self._load_json_file()
        self.transform = transform
        

    def _load_json_file(self):
        img_files = []
        img_labels = []
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                img_files.append(d['name'])
                img_labels.append(d['label'])
        return img_files, img_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label