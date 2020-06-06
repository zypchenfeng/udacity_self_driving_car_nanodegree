/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (is_initialized){
		return;
	}
	num_particles = 100;

	default_random_engine rand_gen;

	is_initialized = false;

	normal_distribution<double> dist_x(x, std[0]); // std[0] - standard deviation of x
    normal_distribution<double> dist_y(y, std[1]); // std[1] - standard deviation of y
    normal_distribution<double> dist_theta(theta, std[2]); // std[2] - standard deviation of theta

	// initialise each particles
    for(int i=0; i<num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(rand_gen);
        particle.y = dist_y(rand_gen);
        particle.theta = dist_theta(rand_gen);
		particle.weight = 1.0;
		weights.push_back(particle.weight);
        particles.push_back(particle);
        }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

    for(int i=0;i<num_particles;i++){
		if (fabs(yaw_rate) < 0.00001) {  
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);} 
		else {
			particles[i].x += velocity*( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) / yaw_rate;
			particles[i].y += velocity*( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) / yaw_rate; 
			particles[i].theta += yaw_rate*delta_t; }

		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	/*
	PLAN:
	for each particle, do
		find the distance between the landmark and transformed observatio
		find the probability and update the weights
	*/
	double particle_weight;
	double trans_x, trans_y, shortest_distance, distance, mu_x, mu_y;

	double normaliser = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
	double exponent, new_weight;
	int nearest_landmark; 
	for(int i = 0; i < num_particles; i++){
		particle_weight = 1.0;
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		for( int j = 0; j < observations.size(); j++){
			// transform the observation from car for each particle
			trans_x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
			trans_y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
			
			// find the nearest landmark
			shortest_distance = numeric_limits<double>::max();
			for (int k=0; k < map_landmarks.landmark_list.size(); k++){
				distance  = pow((trans_x - (double) map_landmarks.landmark_list[k].x_f),2);
				distance += pow((trans_y - (double) map_landmarks.landmark_list[k].y_f),2);
				// distance  = sqrt(distance);
				if(distance < shortest_distance){
					shortest_distance  = distance;
					mu_x 			   = (double) map_landmarks.landmark_list[k].x_f;
					mu_y 			   = (double) map_landmarks.landmark_list[k].y_f;
					nearest_landmark   =   map_landmarks.landmark_list[k].id_i;
				}
			}
			// find the probability and next weight of particle
			exponent = ((pow(trans_x - mu_x,2))/(2 * pow(std_landmark[0],2))) + ((pow(trans_y - mu_y,2))/(2 * pow(std_landmark[1],2)));
			new_weight = normaliser * exp(-exponent);
			particle_weight *= new_weight;
			particles[i].associations.push_back(nearest_landmark);
			particles[i].sense_x.push_back(trans_x);
			particles[i].sense_y.push_back(trans_y);
		}

		weights[i] = particle_weight;
		// update the particle
		particles[i].weight = particle_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine rand_gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resampled_particles;

	for(int i = 0; i < num_particles; i++){
		resampled_particles.push_back(particles[distribution(rand_gen)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
