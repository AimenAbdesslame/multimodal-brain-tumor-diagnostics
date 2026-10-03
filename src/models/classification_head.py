# Input(B, 1, 240, 240) => {Encoder} => {Bottleneck Feature Map } (B, C, H', W') => Hook Captured {GAP } (B, C, 1, 1)=> {Logit } (B, 1)

import torch 
import torch.nn as nn



class UNetClassificationWrapper(nn.Module): 
    def __init__(self , unet_model ,bottleneck_layer , in_channels = 512) : 
        super().__init__() 
        self.unet = unet_model 
        self.extraction_feature = None # this is for storing the bottleneck feature map
        self.hook_handle = bottleneck_layer.register_forward_hook(self.hook_fn)
        # 2. ResNet-style GAP + 2-layer MLP classification head (~66k parameters)
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1, 1),  # Global Average Pooling
            nn.Flatten(),  # Flatten the feature map to a vector
            nn.Linear(in_channels,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128,1),
            
        )
        def hook_fn(module, input, output):
            self.extraction_feature = output
        def forward(self ,x ) : 
            _=self.unet(x) 
            cls_logit = self.mlp_head(self.extraction_feature)
            return cls_logit
        def remove_hook(self):
            self.hook_handle.remove()
        