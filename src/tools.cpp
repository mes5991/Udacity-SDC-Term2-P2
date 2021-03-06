#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	
	//check validity of the following inputs
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()){
		std::cout << "Error: Estimation vector size does not equal ground truth vector size" << std::endl;
			return rmse;
	}

	if (estimations.size() == 0) {
		std::cout << "Error: Estimation vector size is zero" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); i++) {
		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}