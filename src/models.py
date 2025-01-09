import torch
import torch.nn as nn

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 latent_space_dim,
                 encoder_conv_channels,
                 encoder_conv_kernel_size,
                 encoder_conv_stride,
                 encoder_conv_padding,
                 decoder_conv_t_channels,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_stride,
                 decoder_conv_t_padding,
                 eps=0.001):
        assert len(encoder_conv_channels) == len(encoder_conv_kernel_size) == len(encoder_conv_stride) == len(encoder_conv_padding)
        assert len(decoder_conv_t_channels) == len(decoder_conv_t_kernel_size) == len(decoder_conv_t_stride) == len(decoder_conv_t_padding)
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.eps = eps

        #Encoder
        self.encoder = nn.Sequential()
        for i in range(len(encoder_conv_channels)):
            in_channels = input_shape[0] if i == 0 else encoder_conv_channels[i - 1]
            out_channels = encoder_conv_channels[i]
            assert(encoder_conv_kernel_size[i]%2==1)
            padding = encoder_conv_padding[i]
            self.encoder.append(
                nn.Conv2d(in_channels, out_channels,
                          encoder_conv_kernel_size[i],
                          encoder_conv_stride[i],
                          padding=padding)
            )
            self.encoder.append(nn.ReLU())

        #Linear
        dummy_output = self.encoder(torch.zeros(1, *input_shape))
        self.shape_before_flatten = dummy_output.shape[1:]
        features_num = torch.prod(torch.tensor(self.shape_before_flatten))

        self.mean_linear = nn.Linear(features_num, latent_space_dim)
        self.logvar_linear = nn.Linear(features_num, latent_space_dim)

        self.decoder_linear = nn.Linear(latent_space_dim, features_num)

        #Decoder
        self.decoder = nn.Sequential()
        for i in range(len(decoder_conv_t_channels)):
            in_channels = encoder_conv_channels[-1] if i == 0 else decoder_conv_t_channels[i - 1]
            out_channels = decoder_conv_t_channels[i]
            assert(decoder_conv_t_kernel_size[i]%2==1)
            padding = decoder_conv_t_padding[i]
            output_padding = 1 if decoder_conv_t_stride[i]>1 else 0
            self.decoder.append(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   decoder_conv_t_kernel_size[i],
                                   decoder_conv_t_stride[i],
                                   padding=padding,
                                   output_padding=output_padding)
            )
            if i==len(decoder_conv_t_channels)-1:
                self.decoder.append(nn.Sigmoid())
            else:
                self.decoder.append(nn.ReLU())

    def encode(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        mean = self.mean_linear(x)
        logvar = self.logvar_linear(x)
        return mean, logvar

    def sample(self, mean, logvar, eps):
        standard_normal = torch.normal(torch.zeros(mean.shape), torch.ones(logvar.shape))
        return mean + torch.exp(logvar/2) * standard_normal * eps

    def decode(self, x):
        x = nn.ReLU()(self.decoder_linear(x))
        x = nn.Unflatten(-1, self.shape_before_flatten)(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        sampled = self.sample(mean, logvar, self.eps)
        output = self.decode(sampled)
        return output, mean, logvar