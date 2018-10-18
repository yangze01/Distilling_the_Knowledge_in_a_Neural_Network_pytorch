#coding=utf8

from models.BasicModule import *
# from models.__init__ import *
from models.DynamicLSTM import DynamicLSTM
from models.SelfAttention import SelfAttention
from models.Highway import Highway
class LSTMSelfAttention(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.embed_num = args.embed_num
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.lstm_hidden_dim
        self.class_num = args.class_num
        self.num_layers = args.lstm_num_layers
        self.dropout_rate = args.dropout_rate
        self.embed = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=0)
        if args.add_pretrain_embedding:
            pretrained_embeddings = np.loadtxt(args.pretrain_embedding_path)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embed.weight.requires_grad = args.static != True

        self.lstm = DynamicLSTM(input_dim=self.embed_dim,
                            output_dim=self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention = SelfAttention(input_hidden_dim=self.hidden_dim)
        self.highway = Highway(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_dim, self.class_num)

    def forward(self, inputs, lengths):
        """
        :param inputs: (B, L) the text ids of data
        :param lengths: (B, 1)
        :return:(B, C)
        """
        # (B, L, E)
        embed_input = self.embed(inputs)
        # (B, L, E) -> (B, L, 2H)
        out, (ht, ct) = self.lstm(embed_input, lengths)

        # (B, L, 2H) -> (B, L, H)
        output = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (B, L, H) -> (B, H), (B, L, 1)
        weighted_out, weights = self.attention(output)
        highway_out = self.highway(weighted_out)
        logits = self.fc(highway_out)

        return logits




