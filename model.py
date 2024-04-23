import torch
import torch.nn as nn
import torchvision.models as models

class EncorderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncorderCNN, self).__init__()
        self.train_CNN = train_CNN
        # Load pretrained model
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract the Features
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
            
        return self.dropout(self.relu(features))
    

class DecorderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecorderRNN, self).__init__()
        # Define RNN layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layer):
        super(CNNtoRNN, self).__init__()
        self.encorderCNN = EncorderCNN(embed_size=embed_size)
        self.decorderRNN = DecorderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layer)
    
    def forwar(self, images, captions):
        features = self.encorderCNN(images)
        outputs = self.decorderRNN(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length = 50):
        result_caption = []

        with torch.no_grad():
            x = self.encorderCNN(image).unsqueeze(0)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decorderRNN.lstm(x, states)
                output = self.decorderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decorderRNN.embed(predicted).unsequeeze(0)
                if vocabulary.itos[predicted.item()] == "<EOS>" :
                    break

            return [vocabulary.itos[idx] for idx in result_caption]








    







                


