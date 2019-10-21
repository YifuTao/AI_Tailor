# AI_Tailor

Environment: pip install opendr,pytorch cuda 9.2

Use SMPL shape + pose + vertices + measurements as loss

for multiple view fusion, use max pool of features instead of a FC layer of double the size (2048 x 2 for 2 views)

# changes: 
1. in c_loss etc, do not scale by height ratio because initial guess of height leads to unrealistic height ratio and the c_loss x height_ratio  will blow up.

# Notes:
1. with small data(300) the iterative network learns faster and fitted better
2. with more data (1000) the iterative network seems not learning faster and better fitted
Gender:  male
Batch size:  40
Pose 0.100 Shape 0.100 Ver 0.100
Height 0.000 Chest 0.010 Waist 0.010 Neck 0.010 Arm 0.010
--------------------------
# dataset:300
# Baseline
feature---->maxpool----->fc---->SMPL

Epoch 5/24
----------
t Loss: 0.1901 RMS Shape 1.3672 Pose 0.1405 Ver 0.1033 Chest 7.07cm Waist 9.06cm Neck 2.00cm Arm4.07cm H16.02cm
v Loss: 0.2550 RMS Shape 1.5793 Pose 0.1736 Ver 0.1491 Chest 9.22cm Waist 14.44cm Neck 3.04cm Arm3.60cm H13.39cm
Epoch 24/24
----------
t Loss: 0.0441 RMS Shape 0.6411 Pose 0.1463 Ver 0.0888 Chest 4.28cm Waist 5.75cm Neck 1.21cm Arm1.91cm H9.43cm
v Loss: 0.1117 RMS Shape 1.0439 Pose 0.1319 Ver 0.0981 Chest 4.47cm Waist 6.32cm Neck 1.51cm Arm2.71cm H10.35cm
--------------------------------------------------------------------------------------------------------
# Iterative (iteration = 3)
contatenate(feature + SMPL_estimation)----> FC ----> delta_SMPL ---> SMPL_estimation += delta_SMPL

Epoch 5/24
----------
t Loss: 0.1345 RMS Shape 1.1191 Pose 0.2484 Ver 0.1686 Chest 9.69cm Waist 12.89cm Neck 2.75cm Arm3.47cm H31.43cm
v Loss: 0.2332 RMS Shape 1.4790 Pose 0.3145 Ver 0.2109 Chest 5.52cm Waist 9.02cm Neck 2.87cm Arm3.90cm H42.46cm
Epoch 24/24
----------
t Loss: 0.0137 RMS Shape 0.3142 Pose 0.1749 Ver 0.0857 Chest 3.07cm Waist 3.49cm Neck 1.01cm Arm1.42cm H10.70cm
v Loss: 0.0647 RMS Shape 0.7759 Pose 0.1725 Ver 0.1194 Chest 3.92cm Waist 5.85cm Neck 1.17cm Arm2.19cm H14.89cm

# dataset:1,000
# Baseline 
Epoch 1/24
----------
t Loss: 0.2223 RMS Shape 1.4795 Pose 0.1454 Ver 0.1037 Chest 7.83cm Waist 9.83cm Neck 2.29cm Arm4.36cm H16.48cm
v Loss: 0.2401 RMS Shape 1.5291 Pose 0.1877 Ver 0.1560 Chest 10.55cm Waist 14.38cm Neck 2.83cm Arm4.64cm H17.26cm
Epoch 5/24
----------
t Loss: 0.0399 RMS Shape 0.6110 Pose 0.1355 Ver 0.0811 Chest 3.84cm Waist 4.58cm Neck 1.07cm Arm2.10cm H9.54cm
v Loss: 0.0539 RMS Shape 0.7157 Pose 0.1380 Ver 0.0832 Chest 3.58cm Waist 4.31cm Neck 0.92cm Arm1.85cm H8.72cm
Epoch 24/24
----------
t Loss: 0.0089 RMS Shape 0.2736 Pose 0.1047 Ver 0.0583 Chest 2.44cm Waist 3.17cm Neck 0.74cm Arm1.09cm H5.16cm
v Loss: 0.0318 RMS Shape 0.5478 Pose 0.1115 Ver 0.0701 Chest 3.49cm Waist 4.00cm Neck 0.85cm Arm1.56cm H7.06cm

# Iterative (3)
Epoch 1/24
----------
t Loss: 0.1614 RMS Shape 1.2327 Pose 0.2486 Ver 0.1735 Chest 8.99cm Waist 11.76cm Neck 2.56cm Arm3.68cm H29.34cm
v Loss: 0.1587 RMS Shape 1.2228 Pose 0.2488 Ver 0.1675 Chest 6.17cm Waist 8.70cm Neck 1.84cm Arm3.36cm H19.03cm
Epoch 5/24
----------
t Loss: 0.0465 RMS Shape 0.6564 Pose 0.1512 Ver 0.1007 Chest 6.00cm Waist 6.70cm Neck 1.43cm Arm2.14cm H10.57cm
v Loss: 0.0659 RMS Shape 0.7872 Pose 0.1585 Ver 0.1165 Chest 3.70cm Waist 4.46cm Neck 0.98cm Arm2.97cm H16.23cm
Epoch 24/24
----------
t Loss: 0.0092 RMS Shape 0.2774 Pose 0.1065 Ver 0.0566 Chest 2.50cm Waist 3.07cm Neck 0.70cm Arm0.85cm H4.93cm
v Loss: 0.0319 RMS Shape 0.5470 Pose 0.1180 Ver 0.0747 Chest 3.08cm Waist 3.84cm Neck 0.84cm Arm1.73cm H7.69cm

# dataset:10,000 (iterave worse)
# Baseline
Epoch 1/24
----------
t Loss: 0.0264 RMS Shape 0.5081 Pose 0.0611 Ver 0.0403 Chest 3.93cm Waist 5.19cm Neck 1.09cm Arm1.45cm H4.57cm
v Loss: 0.0276 RMS Shape 0.5205 Pose 0.0571 Ver 0.0363 Chest 1.72cm Waist 2.32cm Neck 0.50cm Arm1.77cm H3.86cm
Epoch 5/24
----------
t Loss: 0.0155 RMS Shape 0.3909 Pose 0.0384 Ver 0.0250 Chest 2.56cm Waist 3.40cm Neck 0.71cm Arm1.24cm H3.53cm
v Loss: 0.0084 RMS Shape 0.2867 Pose 0.0382 Ver 0.0243 Chest 2.15cm Waist 3.88cm Neck 0.66cm Arm1.68cm H4.84cm
Epoch 24/24
----------
t Loss: 0.0050 RMS Shape 0.2196 Pose 0.0302 Ver 0.0187 Chest 1.29cm Waist 1.64cm Neck 0.37cm Arm0.71cm H2.14cm
v Loss: 0.0027 RMS Shape 0.1590 Pose 0.0304 Ver 0.0182 Chest 1.00cm Waist 1.22cm Neck 0.28cm Arm0.51cm H1.58cm
# Iterative (3)
Epoch 1/24
----------
t Loss: 0.0297 RMS Shape 0.5380 Pose 0.0603 Ver 0.0583 Chest 4.17cm Waist 5.65cm Neck 1.19cm Arm1.55cm H6.34cm
v Loss: 0.0155 RMS Shape 0.3858 Pose 0.0539 Ver 0.0430 Chest 5.96cm Waist 7.81cm Neck 1.58cm Arm0.75cm H3.33cm
Epoch 5/24
----------
t Loss: 0.0189 RMS Shape 0.4309 Pose 0.0387 Ver 0.0366 Chest 3.20cm Waist 4.26cm Neck 0.89cm Arm1.36cm H4.09cm
v Loss: 0.0217 RMS Shape 0.4635 Pose 0.0392 Ver 0.0295 Chest 1.55cm Waist 3.71cm Neck 0.57cm Arm0.66cm H1.95cm
Epoch 24/24
----------
t Loss: 0.0081 RMS Shape 0.2830 Pose 0.0303 Ver 0.0179 Chest 1.62cm Waist 2.19cm Neck 0.46cm Arm0.87cm H2.47cm
v Loss: 0.0032 RMS Shape 0.1753 Pose 0.0302 Ver 0.0169 Chest 1.02cm Waist 1.22cm Neck 0.28cm Arm0.63cm H1.78cm

