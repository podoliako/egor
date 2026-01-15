# Improving Earthquake Prediction Accuracy using Velocity Model

## Main idea
The distribution of seismic wave velocities with a time component (4D tomography) can be one of the key parameters that allow us to significantly improve the accuracy of earthquake forecasting.

It is proposed to compare the accuracy of the prediction algorithm using 4D tomography as an additional feature with a model without it. We can also include a third model in the comparison: a model that takes into account raw data on arrival times and station locations. By doing this, we can separate the tomography effect of the improvement from the additional information. 

## 4D tomography
While 3D seismic tomography has been studied by many research groups and has some excellent methods to use, there is little work on 4D tomography. This fact justifies the creation of a seismic tomography method optimized for our task.
