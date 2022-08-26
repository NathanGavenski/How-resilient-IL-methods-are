import torch
import torch.nn as nn


class Self_Attn1D(nn.Module):
    """ Self attention Layer """

    def __init__(self, in_dim, activation, k=8):
        super(Self_Attn1D, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim // k,
            kernel_size=1,
        )
        self.key_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim // k,
            kernel_size=1,
        )
        self.value_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> (torch.Tensor, torch.Tensor):
        """
            inputs :
                x : input feature maps(B X C X T)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*T)
        """

        controller = False
        if len(x.size()) == 1:
            controller = True
            x = x[None]
        
        B, C = x.size()
        T = 1
        x = x.view(B, C, T)

        # B X C X (N)
        proj_query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)
        # B X C x (W*H)
        proj_key = self.key_conv(x).view(B, -1, T)
        energy = torch.bmm(proj_query, proj_key)
        # B X (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x).view(B, -1, T)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x
        out = out.squeeze(2)

        if return_attn:
            return out, attention
        else:
            return out, None