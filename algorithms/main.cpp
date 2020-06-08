#include "Sample.hpp"
#include "RegressionModel.hpp"
#include "Eigen"
#include "stats.hpp"
#include <iostream>
#include <vector>
using std::vector;
using std::cout, std::endl;

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    char sep = ' ';
    for (const T& obj : vec) {
        os << sep << obj;
        sep = ',';
    }
    os << " ]";
    return os;
}

template<typename T, typename S>
std::ostream& operator<<(std::ostream& os, const std::pair<T,S>& p) {
    os << "{ " << p.first << ", " << p.second << "}";
    return os;
}

int main() {
    const int REGRESSORS = 10, N = 1e3;
    MatrixXd m2(N,1);
    MatrixXd m1(N,REGRESSORS);

    for (int i = 0; i < N; ++i) {       
        for (int j = 0; j < REGRESSORS; ++j)
            m1(i,j)=stats::rnorm(6, 2);

        m2(i,0) = m1(i,1)+3*m1(i,2)-6.7*m1(i,9);
    }

    Sample sample(m1, m2);
    RegressionModel model;
    
    model.setSample(sample);

    cout<< model.confidenceIntervals(0.95)<<endl;

    model.significance(0.95);
    cout << model.getAlpha() << endl;

    cout << "Done that" << endl;

    system("pause");
    return 0;
}
