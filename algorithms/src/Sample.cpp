#include "Sample.hpp"
#include <vector>

Sample::Sample(const MatrixXd& X, const MatrixXd Y) :_x{ X }, _y{ Y } {}

Sample::Sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    _x = MatrixXd(X[0].size(),X.size());
    for (int i = 0; i < X[0].size(); ++i)
        for (int j = 0; j < X.size(); ++j)
            _x(i, j) = X[i][j];

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

MatrixXd Sample::Xi(const int i) const {
    return _x.row(i);
}

double Sample::Yi(const int i) const {
    return _y(i, 0);
}