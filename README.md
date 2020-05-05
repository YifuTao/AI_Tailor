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

	------chagne gcc in conda  without root
conda install -c psi4 gcc-5
conda install libgcc
conda install -c anaconda libstdcxx-ng

https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6



ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /homes/53/yifu/anaconda3/envs/tailor/lib/python2.7/site-packages/neural_renderer/cuda/load_textures.so)
	-solution:  export LD_LIBRARY_PATH=/scratch/local/ssd/yifu/anaconda3/envs/tailor/lib/

Use SMPL shape + pose + vertices + measurements as loss

for multiple view fusion, use max pool of features instead of a FC layer of double the size (2048 x 2 for 2 views)
