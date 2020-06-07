# CarND Path Planning Project

## Model Documentation

The goal of this project is to drive around the highway, with traffic present on the three-lane road without crossing any speed limit of 50 mph, without driving out of line, to make a lane change when necessary and to avoid collision with other vehicles.

I have used the approach mentioned in project walkthrough video. As the speed limit was 50 mph, the maximum speed allowed is set to `49.5mph`.

 Below is the pseudo-code,

```
if there is no car infront:
    increase the speed
else:
    if left lane is safe:
        change the lane to left
    else if right lane is safe:
        change the lane to right
    else:
        decrease the speed
```

As you can see, I have used a simple approach.

`getBehaviour()` returns a vector of two elements. The first element tells which lane should be used and the second element tells whether we should increase or decrease the speed.

### Trajectory Generation

A spline is used to create a smooth jerk minimizing trajectory. For creating spline, `spline.h` is used. 

The simulator sends back the points that are not yet driven. If such points are present, then the last two points in those points are selected as the starting points of the spline, else, the car's current x and y points and a point tangential to this point are selected. Points at 30, 60 and 90 meters away are also selected. Using these five points, the spline is set. This results in a smooth curve and also minimizes the jerk.

The trajectory is defined by 50 points. The points send back from the simulator are added as the starting points. The rest points are sampled from the spline. Considering the previous points make the trajectory smooth.

### Velocity Management

Intuition is, if a car is in front of the ego car, then speed is decremented else, speed is incremented to reach the maximum allowed speed(49.5, in our case). For controlling the speed, `getBehaviour()`  is used. The speed of the car is incremented or decremented if the second element in the return vector is zero or one respectively. Each increment or decrements is done by `0.224`. 

### Checking lane safety

The lane checking is done in `getBehaviour()`. Three lanes - right lane ( represented by 2), left lane( represented by 0) and middle lane( represented by 1) - are checked for safety. `getBehaviour()` returns the lane number which is safe for driving. Below is the pseudo-code for `getBehaviour()`,

```
if there is no car infront: // current lane is safe for driving
    return current_lane
if left lane is valid and left lane is safe:
    return current_lane - 1
if right lane is valid and right lane is safe:
    return current_lane + 1
```

A lane is considered to be safe if, there is no car within a buffer distance of 30 m. This check is done using the data in `sensor_fusion` vector.