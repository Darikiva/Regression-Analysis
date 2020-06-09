#pragma once
#include "Eigen"
#include "stats.hpp"
#include <vector>
#include <future>
#include <vector>
using Eigen::MatrixXd;

class Sample {
public:
    Sample(const MatrixXd& X, const MatrixXd& Y);
    Sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    Sample(const Sample&) = default;

    [[nodiscard]]
    MatrixXd X() const;

    [[nodiscard]]
    MatrixXd Y() const;

    [[nodiscard]]
    MatrixXd Xi(const int& i) const;

    [[nodiscard]]
    double Yi(const int& i) const;

    bool isDependant(double alpha);
private:
    MatrixXd _x;
    MatrixXd _y;
};