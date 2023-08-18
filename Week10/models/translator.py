import torch
import torch.nn as nn

from Week10.models.encoder import Encoder
from Week10.models.decoder import Decoder


class Translator(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate: float = 0.0,
    ) -> None:
        super(Translator, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)
        self.final_layer = nn.Softmax(nn.Linear(target_vocab_size))
        
        self.d_model = d_model
        
    def forward(self, encoder_inputs, decoder_inputs):
        context = self.encoder(encoder_inputs)  # (batch_size, context_len, d_model)
        output = self.decoder(decoder_inputs, context)  # (batch_size, target_len, d_model)
        output = self.final_layer(output)  # (batch_size, target_len, target_vocab_size)
        return output

    # Load model from file
    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    # Save model to file
    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    # Summary
    def summary(self):
        print(self)

    # Test with input data
    def predict(self, x_test):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(x_test)
        return output