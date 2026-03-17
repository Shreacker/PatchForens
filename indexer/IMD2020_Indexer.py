import json
from pathlib import Path
from typing import Dict, List

class IMD2020_Indexer:
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.index: Dict[str, Dict] = {}

    def build_index(self) -> Dict:
        """
        Main function to build dataset index.
        """
        if not self.dataset_root.exists():
            raise FileNotFoundError(f'Dataset root not found: {self.dataset_root}')
        
        for folder in sorted(self.dataset_root.iterdir()):
            if folder.is_dir():
                folder_data = self._process_folder(folder)
                if folder_data['original'] is not None and len(folder_data['manipulated']) > 0:
                    self.index[folder.name] = folder_data

        return self.index
    
    def save_to_json(self, output_path: str):
        """
        Save index to JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(self.index, f, indent=4)

    def load_from_json(self, json_path: str):
        """
        Load previously saved index.
        """
        with open(json_path, 'r') as f:
            self.index = json.load(f)

        return self.index
    
    def _process_folder(self, folder_path: Path) -> Dict:
        """
        Process a single IMD2020 folder.
        """
        folder_dict = {
            'original': None,
            'manipulated': []
        }

        files = list(folder_path.glob('*'))

        for file in files:
            stem = file.stem.lower()

            if stem.endswith('_orig'):
                folder_dict['original'] = str(file)

        for file in files:
            stem = file.stem.lower()
            
            if stem.endswith('_orig'):
                continue

            if stem.endswith('_mask'):
                continue

            parts = stem.split('_')
            if len(parts) >= 2 and (parts[-1].isdigit() or parts[-1] == 'fake'):
                base_name = stem
                mask_name = f'{base_name}_mask'

                mask_path = None
                for candidate in files:
                    if candidate.stem.lower() == mask_name:
                        mask_path = str(candidate)
                        break

                folder_dict['manipulated'].append({
                    'image': str(file),
                    'mask': mask_path
                })

        return folder_dict

dataset_path = './IMD2020'
indexer = IMD2020_Indexer(dataset_path)
index_dict = indexer.build_index()
indexer.save_to_json('./data/json/imd2020_index.json')