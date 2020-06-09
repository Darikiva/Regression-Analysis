#include "RegressionModel.hpp"
#include "stats.hpp"
#include <future>
#include <thread>

void RegressionModel::setSample(Sample& s) {
    _s = &s;
    _countAlpha();
}

MatrixXd RegressionModel::getAlpha() const {
    return _alpha;
}

namespace {
    double varianceEstimator(const MatrixXd& y, const MatrixXd& xa, const int p) {
        double measure = 0;
        
        for (int i = 0; i < y.rows(); ++i)
            measure += pow(y(i, 0) - xa(i, 0), 2);
        measure /= y.rows() - p;
        
        return measure;
    }

    std::pair<double, double> countInterval(const MatrixXd& alpha, const MatrixXd& X, const double sigma, const double statSignif, const int i) {
        double t = stats::qt(1 - statSignif / 2, X.rows() - X.cols());
        double tmp = sigma * sqrt(X(i, i))*t;
        return { alpha(i,0) - tmp,alpha(i,0) + tmp };
    }

    bool significant(const MatrixXd& alpha, const MatrixXd& X, const double variance, const double statSignif, const int i) {
        double F = stats::qf(1 - statSignif / 2, 1, X.rows() - X.cols());
        double Fi = pow(alpha(i, 0), 2) / (variance*X(i, i));
        return Fi >= F;
    }

    std::pair<double, double> meanVarianceEstimator(const MatrixXd& m) {
        double mean = m.mean();

        double sum = 0;
        for (int i = 0; i < m.rows(); ++i)
            sum += pow(m(i, 0) - mean, 2);

        return { mean,sum / (m.rows()-1) };
    }

    double correlationCoeffEstimator(const MatrixXd& a, const MatrixXd& b) {
        const auto&[mean1, var1] = meanVarianceEstimator(a);
        const auto&[mean2, var2] = meanVarianceEstimator(b);

        double tmp = 0;
        for (int i = 0; i < a.rows(); ++i)
            tmp += (a(i, 0) - mean1)*(b(i, 0) - mean2);
        tmp /= a.rows() + b.rows();

        return tmp / sqrt(var1*var2);
    }
}

vector<std::pair<double, double>> RegressionModel::confidenceIntervals(const double statSignif) {
    auto s = std::async([this, statSignif]() -> vector<std::pair<double, double>> {
        MatrixXd xa = _s->X()*_alpha;
        double sigma = sqrt(varianceEstimator(_s->Y(), xa, _s->X().cols()));

        vector<std::pair<double, double>> result;
        for (int i = 0; i < _s->X().cols(); ++i)
            result.push_back(countInterval(_alpha, _s->X(), sigma, statSignif, i));
        return result;
    });

    return s.get();
}

void RegressionModel::significance(const double statSignif) {
    auto s = std::async([this, statSignif]() {
        double variance = varianceEstimator(_s->Y(), _s->X()*_alpha, _s->X().cols());
        for (int i = 0; i < _s->X().cols(); ++i)
            if (!significant(_alpha, _s->X(), variance, statSignif, i))
                _alpha(i, 0) = 0.;
    });
    return s.get();
}

MatrixXd RegressionModel::defineModel(const double &statSignif) const {
    return _alpha;
}


void RegressionModel::_countAlpha() {
    auto s = std::async([this] {
        MatrixXd tmp = _s->X().transpose()*_s->X();
        _alpha = tmp.inverse()*_s->X().transpose()*_s->Y();
    });
    s.get();
}