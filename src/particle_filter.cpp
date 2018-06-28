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

  srand(time(NULL));
  
  num_particles = 500;
  particles = std::vector<Particle>(num_particles);

  default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
	
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);


  for (int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  double x_pred;
  double y_pred;
  double t_pred;
  normal_distribution<double> dist_x(x_pred, std_x);
  normal_distribution<double> dist_y(y_pred, std_y);
  normal_distribution<double> dist_theta(t_pred, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    if (yaw_rate != 0.0) {
      x_pred = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      y_pred = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      t_pred = particles[i].theta + yaw_rate*delta_t;
    } else {
      x_pred = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y_pred = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      t_pred = particles[i].theta;
    }
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  std::vector<LandmarkObs> obslist = observations;
  std::vector<LandmarkObs>::iterator itobs;
  std::vector<LandmarkObs>::iterator itprd;

  for (itprd = predicted.begin(); itprd != predicted.end(); ++itprd) {
    std::vector<LandmarkObs>::iterator nearestobs = obslist.begin();
    for (itobs = obslist.begin() + 1; itobs != obslist.end(); ++itobs) {
      if (dist((*itprd).x, (*itprd).y, (*nearestobs).x, (*nearestobs).y) > dist((*itprd).x, (*itprd).y, (*itobs).x, (*itobs).y)) {
	nearestobs = itobs;
      }
    }
    (*itprd).id = (*itobs).id;
    obslist.erase(nearestobs);
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  
  //std::vector<LandmarkObs> predicted;

  for (int i = 0; i < num_particles; ++i) {

    // Set associations to nearest observation
    //predicted.clear();
    //for (int j = 0; j < particles[i].associations.size(); ++j) {
    //  LandmarkObs pred;
    //  pred.x = particles[i].sense_x[j];
    //  pred.y = particles[i].sense_y[j];
    //  pred.id = particles[i].associations[j];
    //  predicted.push_back(pred); 
    // }
    //dataAssociation(predicted, observations);
    
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    
    double weight = 1;
    const double gaussfactor = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
    const double sqrland0 = 2*std_landmark[0]*std_landmark[0];
    const double sqrland1 = 2*std_landmark[1]*std_landmark[1];


    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double land_x_map = map_landmarks.landmark_list[j].x_f;
      double land_y_map = map_landmarks.landmark_list[j].y_f;

      if (dist(land_x_map, land_y_map, particles[i].x, particles[i].y) < sensor_range) {
	if (observations.size() > 0 ) {
	  // Transformation
          double obs_x_map = particles[i].x + cos(particles[i].theta) * observations[0].x - sin(particles[i].theta) * observations[0].y;
          double obs_y_map = particles[i].y + sin(particles[i].theta) * observations[0].x + cos(particles[i].theta) * observations[0].y;
	  double min_dist = dist(land_x_map, land_y_map, obs_x_map, obs_y_map);
	  for (int k = 1; k < observations.size(); ++k) {
	    // Transformation
            double x_map = particles[i].x + cos(particles[i].theta) * observations[k].x - sin(particles[i].theta) * observations[k].y;
            double y_map = particles[i].y + sin(particles[i].theta) * observations[k].x + cos(particles[i].theta) * observations[k].y;
	    if (dist(land_x_map, land_y_map, x_map, y_map) < min_dist) {
	      obs_x_map = x_map;
	      obs_y_map = y_map;
	      min_dist = dist(land_x_map, land_y_map, x_map, y_map);
	    }
	  }
	  particles[i].associations.push_back(map_landmarks.landmark_list[j].id_i);
	  particles[i].sense_x.push_back(obs_x_map);
	  particles[i].sense_y.push_back(obs_y_map);
	  
          // Multi-variate Gaussian distribution
          weight *=  gaussfactor * exp(-(obs_x_map - land_x_map)*(obs_x_map - land_x_map)/sqrland0 - (obs_y_map - land_y_map)*(obs_y_map - land_y_map)/sqrland1);
	}
      }
      
    }
    particles[i].weight = weight;
  }

  
}

void ParticleFilter::resample() {

  std::vector<Particle> temp_particles;

  weights.clear();
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }

  unsigned int index = rand() % num_particles;
  double beta = 0.0;
  double maxweight = *max_element(begin(weights), end(weights));
									      
  for (int i = 0; i < num_particles; ++i) {
    beta += (double)rand() / RAND_MAX * 2.0 * maxweight;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    temp_particles.push_back(particles[index]);
  }

  for (int i = 0; i < num_particles; ++i) {
    particles[i] = temp_particles[i];
  }

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
