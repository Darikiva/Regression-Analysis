#pragma once
#include "Sample.hpp"
#include"Eigen"
#include <vector>
using Eigen::MatrixXd;
using std::vector;

class RegressionModel {
public:
    RegressionModel() = default;

    void setSample(Sample& s);

    [[nodiscard]]
    MatrixXd getAlpha() const;

    [[nodiscard]]
    vector<std::pair<double, double>> confidenceIntervals(const double statSignif);

    void significance(const double statSignif);

    [[nodiscard]]
    MatrixXd defineModel(const double& statSignif) const;

private:   
    void _countAlpha();
private:
    Sample* _s;
    MatrixXd _alpha;
};