import torch
import torch.nn as nn
import torchvision.models as models


class OrientationEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super(OrientationEncoder, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1
        )

    def forward(self, x):
       
        return self.encoder(x)


class FourierUnit(nn.Module):
   
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        
        return output


class FreqDomainMixer(nn.Module):
  
    def __init__(self, dim):
     
        super(FreqDomainMixer, self).__init__()
        self.dim = dim
        self.FFC = FourierUnit(self.dim, self.dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
       
        
        x_freq = self.FFC(x)
        x = x_freq + x
        x = self.relu(self.bn(x))

        return x


class OXFM_core(nn.Module):
   
    def __init__(self, dim):
        super(OXFM_core, self).__init__()
        self.dim = dim

       
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )

        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        
        self.conv_branch_1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.conv_branch_2 = nn.Conv2d(self.dim, self.dim, kernel_size=1)

 
        self.fdm = FreqDomainMixer(dim=self.dim * 2)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                      groups=dim, padding_mode='reflect'),
            nn.GELU()
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
     
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.conv_branch_1(self.dw_conv_1(x[0]))
        x_local_2 = self.conv_branch_2(self.dw_conv_2(x[1]))
        f_ms = torch.cat([x_local_1, x_local_2], dim=1)

        f_freq = self.fdm(f_ms)

        f_freq_prime = self.ca_conv(f_freq)

        a_w = self.ca(f_freq_prime)
        f_ca = a_w * f_freq_prime

        return f_ca


class OXFM(nn.Module):

    def __init__(self, in_channels=96):
        super(OXFM, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.oxfm = OXFM_core(in_channels)

    def forward(self, fused_feature):

        f_norm = self.norm(fused_feature)

        f_ca = self.oxfm(f_norm)

        f_out = f_norm + f_ca

        return f_out


