import torch 
import torch.nn as nn


class LSTMNet(nn.Module):

    def __init__(self, input_dim, lstm_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
                
        super().__init__()

        # Input Dims
        self.input_dim = input_dim

        # LSTM hidden dims
        self.lstm_dim = lstm_dim

        # Hidden Dims
        self.hidden_dim = hidden_dim

        # Output Dims
        self.output_dim = output_dim

        self.n_layers = n_layers

        # RNN layer
        self.lstm_layer = nn.LSTM(self.input_dim, 
                                self.lstm_dim, 
                                self.n_layers, 
                                batch_first=True)
        
        # Fully-connected output layer
        self.fc1 = nn.Linear(self.lstm_dim*2,
                            self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim,
                            self.output_dim)

        self.do = nn.Dropout(p=dropout)

        # Initialize weights
        self.init_weights()

    def forward(self, x):

        # Validate input shape
        assert len(x.shape)==3, f"Expected input to be 3-dim, got {len(x.shape)}"

        # Get dimensions of the input
        batch_size, seq_size, input_size = x.shape

        # Initialize hidden_state
        hidden, cell = self.init_zero_hidden(batch_size)

        # Pass through the recurrent layer
        out, (hidden, cell) = self.lstm_layer(x, (hidden, cell))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = torch.tanh(out[:, -2:, :].contiguous().view(batch_size, -1))
        out = self.do(out)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = torch.unsqueeze(out, -1)

        return out

    
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
                Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        h_0 = torch.zeros(self.n_layers, batch_size, self.lstm_dim, requires_grad=False)
        c_0 = torch.zeros(self.n_layers, batch_size, self.lstm_dim, requires_grad=False)
        return h_0, c_0
    

    def init_weights(self):
        for p in self.lstm_layer.parameters():
            nn.init.normal_(p)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)




class MultiStepLSTMNet(nn.Module):

    def __init__(self, input_dim, lstm_dim, hidden_dim, output_dim, n_layers, forecast_horizon, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.forecast_horizon = forecast_horizon

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, lstm_dim, n_layers, batch_first=True)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(output_dim, lstm_dim, n_layers, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(lstm_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state for encoder
        encoder_hidden, encoder_cell = self.init_zero_hidden(batch_size)

        # Encode input sequence
        _, (hidden, cell) = self.encoder_lstm(x, (encoder_hidden, encoder_cell))

        # Prepare decoder input (first input is zeros)
        decoder_input = torch.zeros((batch_size, 1, self.output_dim), device=x.device)

        outputs = []
        for t in range(self.forecast_horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            out = self.fc(out.squeeze(1))  # Pass through fully connected layer
            outputs.append(out)
            decoder_input = out.unsqueeze(1)  # Use predicted output as next input

        return torch.stack(outputs, dim=1)  # Shape: (batch_size, forecast_horizon, output_dim)

    def init_zero_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.lstm_dim, device='cuda' if torch.cuda.is_available() else 'cpu'),
                torch.zeros(self.n_layers, batch_size, self.lstm_dim, device='cuda' if torch.cuda.is_available() else 'cpu'))
