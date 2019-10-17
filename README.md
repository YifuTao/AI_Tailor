# AI_Tailor

Environment: pip install opendr,pytorch cuda 9.2

Use SMPL shape + pose + vertices + measurements as loss

for multiple view fusion, use max pool of features instead of a FC layer of double the size (2048 x 2 for 2 views)