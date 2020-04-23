# AI_Tailor

Environment: 

Pyhon 2.7
pytorch (pip install future)
visdom
matplotlib
chumpy
opencv-python
conda install -c menpo osmesa	(or sudo apt-get install libosmesa6-dev)
pip install opendr
pip install neural_renderer_pytorch

Use SMPL shape + pose + vertices + measurements as loss

for multiple view fusion, use max pool of features instead of a FC layer of double the size (2048 x 2 for 2 views)
