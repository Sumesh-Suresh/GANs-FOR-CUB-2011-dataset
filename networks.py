import torch
import torch.nn as nn
import torch.nn.functional as F

device ='cuda'

class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor
        self.pixel_shift = nn.PixelShuffle(self.upscale_factor)

    @torch.jit.script_method
    def forward(self, x):
        # nearest neighbor upsampling
        # x channel-wise upscale_factor^2 times
        # torch.nn.PixelShuffle forms an output of dimension (batch, channel, height*upscale_factor, width*upscale_factor)
        # Applying convolution and return output
       
        x = x.repeat(1,int(self.upscale_factor**2),1,1)
        y = self.pixel_shift(x)
        z = self.conv(y)
    
        # print('shape of input after upsampling', x.shape)
        # print("shape after pixel shift ", y.shape)
        # print('shape of after convolution',z.shape)
        

        return z 

class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio

        self.unshuffle = nn.PixelUnshuffle(self.downscale_ratio)

    @torch.jit.script_method
    def forward(self, x):
       
        # Implementing spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension (batch, channel*downscale_factor^2, height, width)
        # 2. Then split channel-wise and reshape into (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution, and return the output
       
        x = self.unshuffle(x)   # b,c*r**2,h,w
        b, c, h, w = x.shape   # b,c,h*r,w*r
        y = x.view(int(self.downscale_ratio**2) ,b,-1,h,w)  #
        y = y.mean(dim=0)
        z = self.conv(y)
        return z
        


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d( input_channels, n_filters, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(input_channels=n_filters,n_filters=n_filters,kernel_size=3,padding=1)
        )
        self.upsample_conv_res = UpSampleConv2D(input_channels=input_channels,kernel_size=1, n_filters=n_filters)
       

    @torch.jit.script_method
    def forward(self, x):
        
        y = self.layers(x)
        res= self.upsample_conv_res(x)
        out = y + res
        return out
       
class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        
        out_channels = n_filters
        self.layers = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                DownSampleConv2D(out_channels, kernel_size=3, n_filters=out_channels, downscale_ratio=2, padding=1)
            )
        self.res = DownSampleConv2D(input_channels, n_filters=out_channels, kernel_size=1)

       

    @torch.jit.script_method
    def forward(self, x):
        
        y = self.layers(x)
        res = self.res(x)
        out = y + res
        return out
      


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
       
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels,n_filters,kernel_size=kernel_size,stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(n_filters,n_filters,kernel_size=kernel_size,stride=(1,1),padding=(1,1)),
        )
      
    @torch.jit.script_method
    def forward(self, x):
   
       
        return self.layers(x)+x
      

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
     
        self.starting_img_size = starting_image_size
        self.layers = torch.nn.Sequential(
            ResBlockUp(input_channels=128),
            ResBlockUp(input_channels=128),
            ResBlockUp(input_channels=128),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh()
            )
        self.dense = nn.Linear(128,2048,bias=True)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
      
        batch, _ = z.shape
        g = self.dense(z.to('cuda'))
        reshaped = g.view(batch,128,self.starting_img_size,self.starting_img_size)
        z = self.layers(reshaped)
        return z
     

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
      
        z = torch.randn((n_samples,128), device=torch.device("cuda")).half()
        g = self.forward_given_samples(z)
        return g 
     
class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
   
        self.layers = nn.Sequential(ResBlockDown(3),
                                    ResBlockDown(128),
                                    ResBlock(128),
                                    ResBlock(128),
                                    nn.ReLU())
        self.dense = nn.Linear(128,1,bias= True)

    @torch.jit.script_method
    def forward(self, x):
       
        d = self.layers(x)
        s = torch.sum(d,dim=(-2,-1))
        z = self.dense(s)
        return z
    
