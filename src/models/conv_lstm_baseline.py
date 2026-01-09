import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMBaseline(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=64, kernel_size=(3,3), num_layers=3):
        super(ConvLSTMBaseline, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_channels if i == 0 else hidden_dim
            self.encoder_layers.append(ConvLSTMCell(input_dim, hidden_dim, kernel_size))

        # Decoder layers (inverse order)
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(ConvLSTMCell(hidden_dim * 2 if i < num_layers - 1 else hidden_dim, hidden_dim, kernel_size))

        # Final convolution to predict output channels
        self.final_conv = nn.Conv2d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, channels, H, W)
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states for encoder
        encoder_states = []
        for layer in self.encoder_layers:
            encoder_states.append(layer.init_hidden(batch_size, (height, width)))

        # Encoder: process sequence
        skip_connections = []
        for t in range(seq_len):
            input_t = x[:, t]  # (batch, channels, H, W)
            for i, layer in enumerate(self.encoder_layers):
                h, c = encoder_states[i]
                h, c = layer(input_t, (h, c))
                encoder_states[i] = (h, c)
                input_t = h  # for next layer
            skip_connections.append(h)  # last layer output as skip

        # Decoder: predict next timestep
        # Use last hidden state as input to decoder
        decoder_input = encoder_states[-1][0]  # last layer h

        # Initialize decoder states
        decoder_states = []
        for layer in self.decoder_layers:
            decoder_states.append(layer.init_hidden(batch_size, (height, width)))

        # For simplicity, single step prediction
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-1] if i == 0 else decoder_states[i-1][0]  # use skip from encoder
            combined_input = torch.cat([decoder_input, skip], dim=1) if i < self.num_layers - 1 else decoder_input
            h, c = decoder_states[i]
            h, c = layer(combined_input, (h, c))
            decoder_states[i] = (h, c)
            decoder_input = h

        # Final prediction
        output = self.final_conv(decoder_input)  # (batch, input_channels, H, W)
        return output.unsqueeze(1)  # (batch, 1, channels, H, W) for forecast_horizon=1