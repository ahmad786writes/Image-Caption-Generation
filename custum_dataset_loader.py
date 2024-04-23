import os
import pandas as pd 
import spacy # for tokenizer
from torch.nn.utils.rnn import pad_sequence # pad batch
from torch.utils.data import Dataset, DataLoader
from PIL import Image # Load images
import torchvision.transforms as transforms # to transform numpy to tensor
import torch

spacy_eng = spacy.load(r'C:\Users\Ahmad Ansari\anaconda3\envs\pytorch\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.7.1')

class vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return self.itos
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    # "I am good Boy" -> ["i", "am", "good", "boy"]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text=text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    
class flikr8KDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None, freq_threshold=5):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Get images, Caption Columns
        self.imgs = self.annotations["image"]
        self.Captions = self.annotations["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.Captions.tolist())

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        caption = self.Captions[index]
        image_id = self.imgs[index]
        image = Image.open(os.path.join(self.root_dir, image_id)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(root_folder, annotation_file, transform, batch_size=8, num_workers=8, shuffle=True, pin_memory=True):
    dataset = flikr8KDataset(annotation_file, root_folder, transform=transform)
    print("I am Here")
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader
# if __name__ == '__main__':
#     transform = transforms.Compose([
#     transforms.Resize((64,64)),
#     transforms.ToTensor()
#     ])
#     dataloader = get_loader('dataset/flickr8k/Images/', annotation_file = "dataset/flickr8k/captions.txt", transform = transform)   

#     for idx, (imgs, captions) in enumerate(dataloader):
#         print(imgs.shape)
#         print(captions.shape)