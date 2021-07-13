import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FPN(nn.Module):
    def __init__(self,in_channels=512):
        super(FPN, self).__init__()   
        # Top layer
        self.toplayer = nn.Conv2d(512, in_channels, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, in_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, in_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, in_channels, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
        
    def forward(self, x):
        # Top-down
        c5=x[0]
        c4=x[1]
        c3=x[2]
        c2=x[3]
        p5 = self.toplayer(c5)
#        p4 = self._upsample_add(p5, self.latlayer1(c4))
#        p3 = self._upsample_add(p4, self.latlayer2(c3))
#        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
#        p4 = self.smooth1(p4)
#        p3 = self.smooth2(p3)
#        p2 = self.smooth3(p2)
        p4 = self.latlayer1(c4)
        p3 = self.latlayer2(c3)
        p2 = self.latlayer3(c2)
        return p2, p3, p4, p5

def main():
        c5=torch.randn(1,512,7,7)
        c4=torch.randn(1,256,14,14)
        c3=torch.randn(1,128,28,28)
        c2=torch.randn(1,64,56,56)
        
        inputs=[c5,c4,c3,c2]
        fpn = FPN(512)
        l=fpn(inputs)
        print(l[0].shape)
if __name__ == '__main__':
    main()
        
