#include "Sample.hpp"
#include <vector>

Sample::Sample(const MatrixXd& X, const MatrixXd Y) :_X{ X }, _Y{ Y } {}

Sample::Sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    _X = MatrixXd(X[0].size(),X.size());
    for (int i = 0; i < X[0].size(); ++i)
        for (int j = 0; j < X.size(); ++j)
            _X(i, j) = X[i][j];

    _Y = MatrixXd(Y.size(),1);
    for (int i = 0; i < Y.size(); ++i)
        _Y(i, 0) = Y[i];
}

MatrixXd Sample::X() const {
    return _X;
}

MatrixXd Sample::Y() const {
    return _Y;
}

MatrixXd Sample::Xi(const int i) const {
    return _X.row(i);
}

double Sample::Yi(const int i) const {
    return _Y(i, 0);
}