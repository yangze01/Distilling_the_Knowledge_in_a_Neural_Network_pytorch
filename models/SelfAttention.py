from models.BasicModule import *
class SelfAttention(BasicModule):
    def __init__(self, input_hidden_dim):
        super().__init__()
        self.hidden_dim = input_hidden_dim
        self.fc = nn.Linear(self.hidden_dim, 1)
    def forward(self, encode_output):
        # (B, L, H) -> (B, L, 1)
        energy = self.fc(encode_output)
        weights = F.softmax(energy.squeeze(-1), dim=1)

        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encode_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights
