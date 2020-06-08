#include "RegressionModel.hpp"
#include "stats.hpp"

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

    double Fi(const MatrixXd& alpha, const MatrixXd& X, const double variance, const double statSignif, const int i) {        
        return pow(alpha(i, 0), 2) / (variance*X(i, i));
    }

    bool significant(const MatrixXd& alpha, const MatrixXd& X, const double variance, const double statSignif, const int i) {        
        double F = stats::qf(1 - statSignif / 2, 1, X.rows() - X.cols());
        return Fi(alpha,X,variance,statSignif,i) >= F;
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

    int min(const vector<double>& v) {
        int ans = 0;
        for (int i = 0; i < v.size(); ++i)
            if (v[ans] > v[i])
                ans = i;
        return ans;
    }
}

vector<std::pair<double, double>> RegressionModel::confidenceIntervals(const double statSignif) {
    MatrixXd xa = _s->X()*_alpha;
    double sigma = sqrt(varianceEstimator(_s->Y(), xa, _s->X().cols()));

    vector<std::pair<double, double>> result;
    for (int i = 0; i < _s->X().cols(); ++i)
        result.push_back(countInterval(_alpha, _s->X(), sigma, statSignif, i));

    return result;
}

void RegressionModel::significance(const double statSignif) {
    double variance = varianceEstimator(_s->Y(), _s->X()*_alpha, _s->X().cols());
    for (int i = 0; i < _s->X().cols(); ++i)
        if (!significant(_alpha, _s->X(), variance, statSignif, i))
            _alpha(i, 0) = 0.;
}

vector<int> RegressionModel::defineModel(const double statSignif) const {
    vector<int> I(_s->X().cols());
    double variance = varianceEstimator(_s->Y(), _s->X()*_alpha, _s->X().cols());
    for (int i = 0; i < I.size(); ++i)
        I[i] = i;
    int p = I.size();

    while (true) {
        vector<double> vFi(p);
        for (int i = 0; i < p; ++i)
            vFi[i] = Fi(_alpha, _s->X(), variance, statSignif, I[i]);

        int candidate = min(vFi);
        ///Check H0: Fi<F(1,N-p)
        if (vFi[candidate] < stats::qf(1 - statSignif / 2, 1, _s->X().rows() - p)) {
            vFi.erase(vFi.begin() + candidate);
            I.erase(I.begin() + candidate);
            --p;
            if (!p)
                break;
        }
        else
            break;
    }

    return I;
}


void RegressionModel::_countAlpha() {
    MatrixXd tmp = _s->X().transpose()*_s->X();
    _alpha = tmp.inverse()*_s->X().transpose()*_s->Y();
}