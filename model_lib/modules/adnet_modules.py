import torch
import torch.nn as nn

class ADNetBlock(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(ADNetBlock, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False)

        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False)

        self.conv_h1 = nn.Conv2d(num_hidden, num_hidden*4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False)

        self.conv_uvx = nn.Conv2d(in_channel, num_hidden * 2, kernel_size=1,
                        stride=stride, padding=0, bias=False)
                                    
        self.dx = nn.Conv2d(num_hidden, num_hidden, kernel_size=5,
                        stride=1, padding=5//2, groups=num_hidden, bias=False)

        self.dy = nn.Conv2d(num_hidden, num_hidden, kernel_size=5,
                        stride=1, padding=5//2, groups=num_hidden, bias=False)

        self.dxdx = nn.Conv2d(num_hidden, num_hidden, kernel_size=5,
                        stride=1, padding=5//2, groups=num_hidden, bias=False)

        self.dydy = nn.Conv2d(num_hidden, num_hidden, kernel_size=5,
                        stride=1, padding=5//2, groups=num_hidden, bias=False)

        self.a = MSCA(num_hidden)
        self.b = MSCA(num_hidden)


    def forward(self, x_t, h_t, c_t, h1_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        h1_concat = self.conv_h1(h1_t)
        uv_concat = self.conv_uvx(x_t)

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_h1, f_h1, g_h1, o_h1 = torch.split(h1_concat, self.num_hidden, dim=1)
        ux, vx= torch.split(uv_concat, self.num_hidden, dim=1)
        
        i_t = torch.sigmoid(i_x + i_h + i_h1)
        f_t = torch.sigmoid(f_x + f_h + f_h1)
        g_t = torch.tanh(g_x + g_h + g_h1)
        u_fild = torch.tanh(ux)
        v_fild = torch.tanh(vx)

        c_new = f_t * c_t + i_t * g_t  -  self.a(c_t)*(u_fild*self.dx(c_t) + v_fild*self.dy(c_t)) +  self.b(c_t)*(self.dxdx(c_t) + self.dydy(c_t))

        o_t = torch.sigmoid(o_x + o_h + o_h1)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new

class MSCA(nn.Module):

    def __init__(self, in_channel):
        super(MSCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False) 
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 1, in_channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x)

        y_local = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).expand_as(x)
        y_global = self.fc(y.view(b, c)).view(b, c, 1, 1).expand_as(x)
        y = y_local + y_global

        y = self.sigmoid(y)

        return y
