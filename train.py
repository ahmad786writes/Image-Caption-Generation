import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from custum_dataset_loader import get_loader
from model import CNNtoRNN
    

def train():
    transform = transforms.Compose([
            transforms.Resize((356,356)),
            # because i am using inception and it take input of 299*299
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader, dataset = get_loader('/kaggle/input/flickr8kimagescaptions/flickr8k/images', annotation_file = '/kaggle/input/flickr8kimagescaptions/flickr8k/captions.txt', transform = transform, num_workers= 2)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    

    # hyperParameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab.itos)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
#     load_model = True

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model
    model = CNNtoRNN(embed_size=embed_size, hidden_size=hidden_size,
                      vocab_size=vocab_size, num_layer=num_layers).to(device=device)
    
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load Checkpoint if any
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()
    for epoch in range(num_epochs):
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                'state_dict' : model.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'step' : step
                }
            save_checkpoint(checkpoint)
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training Loss", loss.item(), global_step=step)
            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
if __name__ == "__main__":
    train()



