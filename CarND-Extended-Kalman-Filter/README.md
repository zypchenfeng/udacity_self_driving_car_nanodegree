# Extended Kalman Filter 
## Code
Code can be found in the src directory. 
The following source files are changed

#### 1. FusionEKF.cpp
This is used for initialising the variables needed for the actual kalman filter.
The variables initialised include:

##### state x, prediction F, measurement noise H, prediciton noise P

#### 2. KalmanFilter.cpp
This class implements the predict and update function

Liidar uses linear equations, the update step will use the basic Kalman filter equations.  While radar uses non-linear equations, so the update step involves linearizing the equations with the Jacobian matrix. 

#### 3. Tools.cpp
This implements functions to calculate root mean squared (RMSE) error and the Jacobian matrix.

## Build Instructions
To compile the package, go to root directory

1. mkdir build &&  cd build 
3. cmake ..
4. make
5. ./ExtendedKF ( Make sure that simulator is running)

## Results
Obtained RMSE values are 

px  = 0.0973178
py = 0.0854597
vx = 0.451267
vy = 0.439935



![](/home/arjun/Personal_Projects/CarND-Extended-Kalman-Filter-Project_arjun/results.png)