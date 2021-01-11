import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
import torch.nn.functional as F 

import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_channel: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 pe: bool = False,
                 mask: bool = False):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_input = d_input
        self._d_channel = d_channel
        self._d_model = d_model
        self._pe = pe
        self._h = h
        self._q = q

        self.layers_encoding_1 = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask = mask
                                                      ) for _ in range(N)])

        self.layers_encoding_2 = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask = True) for _ in range(N)])

        # self.layers_decoding = nn.ModuleList([Decoder(d_model,
        #                                               q,
        #                                               v,
        #                                               h,
        #                                               dropout=dropout) for _ in range(N)])

        self._embedding_channel = nn.Linear(self._d_channel, d_model)
        self._embedding_input = nn.Linear(self._d_input, d_model)

        
        self._linear = nn.Linear(d_model * d_input + d_model * d_channel, d_output)
        self._layerNorm = nn.LayerNorm(d_model * d_input + d_model * d_channel)
        # self._gate = nn.Linear(d_model * d_input + d_model * d_channel,2)
        self._gate = nn.Linear(d_model * d_input + d_model * d_channel,2)

        # self._W_q = nn.Linear(d_model * d_input + d_model * d_channel,q * h)
        # self._W_k = nn.Linear(d_model * d_input + d_model * d_channel,q * h)
        # self._W_v = nn.Linear(d_model * d_input + d_model * d_channel,v * h)

        self._attention_linear = nn.Linear(d_model * d_input + d_model * d_channel,8 * d_input +8  * d_channel)
        # self._linear_avg = nn.Linear(d_model+d_model, d_output)

        self._W_q = nn.Linear(1,q * h)
        self._W_k = nn.Linear(1,q * h)
        self._W_v = nn.Linear(1,v * h)

        # Output linear function
        # self._W_o = nn.Linear(v * h, d_model * d_input + d_model * d_channel)
        self._W_o = nn.Linear(v * h, 1)
        self._dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # encoding_channel = self._embedding_channel(x)

        encoding_channel = F.relu_(self._embedding_channel(x))

        # 位置编码
        # if self._pe:
        #     pe = torch.ones_like(encoding_channel[0])
        #     position = torch.arange(0, self._d_input).unsqueeze(-1)
        #     temp = torch.Tensor(range(0, self._d_model, 2))
        #     temp = temp * -(math.log(10000)/self._d_model)
        #     temp = torch.exp(temp).unsqueeze(0)
        #     temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        #     pe[:, 0::2] = torch.sin(temp)
        #     pe[:, 1::2] = torch.cos(temp)

        #     encoding_channel = encoding_channel + pe

        # Encoding stack
        for layer in self.layers_encoding_1:
            encoding_channel,score_channel = layer(encoding_channel)

        encoding_input = self._embedding_input(x.transpose(-1, -2))

        # 位置编码
        if self._pe:
            pe = torch.ones_like(encoding_input[0])
            position = torch.arange(0, self._d_channel).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_input = encoding_input + pe

        for layer in self.layers_encoding_2:
            encoding_input,score_input = layer(encoding_input)

                
        # 三维变两维
        encoding_channel = encoding_channel.reshape(encoding_channel.shape[0], -1)
        encoding_input = encoding_input.reshape(encoding_input.shape[0], -1)
        
        # encoding_channel = F.adaptive_avg_pool2d(encoding_channel, (1, encoding_channel.shape[-1])).squeeze()
        # encoding_input = F.adaptive_avg_pool2d(encoding_input, (1, encoding_input.shape[-1])).squeeze()

        
        gate = F.softmax(self._gate(torch.cat((encoding_input, encoding_channel), dim=1)), dim=1)

        encoding_gate = torch.cat((encoding_input * gate[:, 0:1], encoding_channel * gate[:, 1:2]), dim=1)


        
        
        encoding_gate = self._layerNorm(encoding_gate)
        # encoding_gate = self._dropout(encoding_gate)
        output = self._linear(encoding_gate)

        return output,score_channel,score_input
