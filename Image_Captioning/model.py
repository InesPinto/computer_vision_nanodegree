import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        
        # embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers, batch_first = True)
       
        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        #self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        #Decode image feature vectors and generates captions
        
        # Remove end tag
        captions = captions[:, :-1]
        embeddings = self.embedding_layer(captions)
        
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.lstm(embeddings)
        outupt = self.linear(hiddens)
        
        return outupt        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            predicted = out.max(1)[1]
            predicted_sentence.append(predicted.item())
            inputs = self.embedding_layer(predicted).unsqueeze(1)
        return predicted_sentence    
            
            