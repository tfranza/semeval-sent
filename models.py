import sys
import torch
import torch.nn as nn

class FirstNeuralNetwork(nn.Module):
    def __init__(self, embeddings, embedding_size, hidden_size, num_classes):
        super(FirstNeuralNetwork, self).__init__()
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.loss_fn= nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, data_input):
        x = self.embedding_layer(data_input)
        x, (hidden_state, _) = self.lstm(x)
        hidden_state.squeeze(0)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x.squeeze()

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU. 
        """
        return self.input_layer.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = FirstNeuralNetwork(**args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        params = {
            'args': dict(
                embeddings = self.embeddings,
                embedding_size = self.embedding_size,
                hidden_size = self.hidden_size, 
                num_classes = self.num_classes
            ),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

        