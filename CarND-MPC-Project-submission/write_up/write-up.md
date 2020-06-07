
## MPC Project

### Model

In this project, kinematic model of vehicle is used. The state of the car is given by position (ie, x and y coordinate), heading orientation (psi), velocity, cross track error (cte) and psi error (epsi). Cross track error and psi error is added to state to know how good the state. Actuator outputs are acceleration and delta (steering angle). The update of state from time t, to t+1 is done using the following equations,

      x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      psi_[t+1] = psi[t] - v[t] / Lf * delta[t] * dt
      v_[t+1] = v[t] + a[t] * dt
      cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      epsi[t+1] = psi[t] - psides[t] - v[t] * delta[t] / Lf * dt

### Timestep Length and Elapsed Duration (N & dt)

N and dt were two of the hyper parameters that had to be tuned. Even though I tried various N and dt values, I have settled with the values which were discussed in the Udacity's Q&A video on this project. The values are as follows,

    N = 10

    dt = 0.1

### Polynomial Fitting and MPC Preprocessing

The waypoints transformed to car's coordinates, thus making (x, y) at origin (0, 0) and orientation angle, psi, zero. Code is at 103 - 108 in main.cpp

### Model Predictive Control with Latency

Following are the equations used to update the initial state for accomodating the latency.

          pred_px = v * latency
          pred_psi = -1.0 * ((v * delta * latency) / Lf)
          pred_v = v + acceleration * latency
          pred_cte = cte + v * sin(epsi) * latency
          pred_epsi = epsi - ((v * delta * latency) / Lf)
          
### Output
<br/>
<center>
    <video width="960" height="540" controls src="output_video.mp4" />
</center>
       
