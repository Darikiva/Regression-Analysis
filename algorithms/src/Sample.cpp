#include "Sample.hpp"
#include <vector>

double corelationCoef(const Eigen::VectorXd& ksi, const Eigen::VectorXd& theta) {
    double x_hat{0}, y_hat{0};
    int n = ksi.size();
    for(auto i = 0; i < ksi.size(); ++i) {
        x_hat += ksi[i];
    }
    x_hat /= n;
    for(auto i = 0; i < theta.size(); ++i) {
        y_hat += theta[i];
    }
    y_hat /= n;
    double r{0};
    double denom{0}, numeral_sum_x{0}, numeral_sum_y{0};
    for(auto i = 0; i < n; ++i) {
        denom += (ksi[i] - x_hat) * (theta[i] - y_hat);
        numeral_sum_x += std::pow(ksi[i] - x_hat, 2);
        numeral_sum_y += std::pow(theta[i] - y_hat, 2);
    }
    if (numeral_sum_x * numeral_sum_y == 0) {
        if(denom == 0) return 1;
        else return INT_MAX;
    }
    r = denom / sqrt(numeral_sum_x * numeral_sum_y);
    return r;
}

Sample::Sample(const MatrixXd& X, const MatrixXd& Y) :_x{ X }, _y{ Y } {}

Sample::Sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    _x = MatrixXd(X[0].size(),X.size());
    for (int i = 0; i < X[0].size(); ++i)
        for (int j = 0; j < X.size(); ++j)
            _x(i, j) = X[j][i];

    _y = MatrixXd(Y.size(),1);
    for (int i = 0; i < Y.size(); ++i)
        _y(i, 0) = Y[i];
}

MatrixXd Sample::X() const {
    return _x;
}

MatrixXd Sample::Y() const {
    return _y;
}

MatrixXd Sample::Xi(const int& i) const {
    return _x.row(i);
}

double Sample::Yi(const int& i) const {
    return _y(i, 0);
}

bool Sample::isDependant(double alpha) {
    auto mda = std::async(std::launch::async, [this, alpha]() -> bool
    {
        MatrixXd wow(this->X().rows(), this->X().cols() + 1);
        wow << this->Y(), this->X();
        MatrixXd R(this->X().cols() + 1, this->X().cols() + 1);
        for (int i = 0; i < R.cols(); ++i) {
            for (int j = 0; j < R.cols(); ++j) {
                R(i,j) = corelationCoef(wow.col(i), wow.col(j));
            }
        }
        double r = 1 - R.determinant() / R.block(1, 1, R.cols() - 1, R.cols() - 1).determinant();
        if(abs(r) <= 1 && abs(r) > 0.999) return true;
        double F = (r / (1 - r)) * ((double) (this->X().rows(), - this->X().cols() - 1) / (double) this->X().cols());
        return F < stats::qf(alpha, this->X().cols(), this->X().rows() - this->X().cols() - 1);
    });
    return mda.get();
}