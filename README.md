Semantic Segmentation with 3 different networks:
1) Deeplab v3+ 
2) EMANet
3) MFNet (New!)

How to Run:

train.py --model="MFNet" 


Prerequisites:
1) Install pytorch and all dependecies
2) Download data from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data
3) Set dataset path in mypath.py
4) Download ResNet pretrained models:
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
and put them in ./pretrained directory
