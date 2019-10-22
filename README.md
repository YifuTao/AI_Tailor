# AI_Tailor

Environment: pip install opendr,pytorch cuda 9.2

Use SMPL shape + pose + vertices + measurements as loss

for multiple view fusion, use max pool of features instead of a FC layer of double the size (2048 x 2 for 2 views)

# changes: 
1. in c_loss etc, do not scale by height ratio because initial guess of height leads to unrealistic height ratio and the c_loss x height_ratio  will blow up.
2. Network:
cat(features_per_view, features_global)----->FC----->delta_feature---->new_feature per view---maxpool---new feature_global 
3. initialisation of 
	feature_global: change from using zeros to maxpool the features of different views 
	prameter_global: change from using zeros to FC the feature_global 
and this works better!

Gender:  male
Batch size:  40
Pose 0.100 Shape 0.100 Ver 0.100
Height 0.000 Chest 0.010 Waist 0.010 Neck 0.010 Arm 0.010


