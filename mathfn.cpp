//
// Created by peterglenn on 7/11/24.
//


#include <cmath>
#include <numbers>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "mathfn.h"

#include "math.h"

inline int nchoosek( const int n, const int k) {
    if (k == 0)
        return 1;
    return (n* nchoosek(n-1,k-1))/k;
}

// Function to compute Stirling numbers of
// the second kind S(n, k) with memoization
inline int stirling(int n, int k) {

    // Base cases
    if (n == 0 && k == 0) return 1;
    if (k == 0 || n == 0) return 0;
    if (n == k) return 1;
    if (k == 1) return 1;


    // Recursive formula
    return k * stirling(n - 1, k) + stirling(n - 1, k - 1);
}

// Function to calculate the total number of
// ways to partition a set of `n` elements
inline int bellNumber(int n) {

    int result = 0;

    // Sum up Stirling numbers S(n, k) for all
    // k from 1 to n
    for (int k = 1; k <= n; ++k) {
        result += stirling(n, k);
    }
    return result;
}

inline int totient( int n)
{
    int phi = n > 1 ? n : 1;
    int max = sqrt( n ) + 1;
    for (int p = 2; p <= max; p++)
    {
        if (n % p == 0)
        {
            phi -= phi / p;
            while (n % p == 0)
                n /= p;
        }
    }
    if (n > 1)
        phi -= phi / n;
    return phi;
}

inline double isinffn( std::vector<double>& din ) {return (double)(din[0] == std::numeric_limits<double>::infinity());}
inline double floorfn( std::vector<double>& din ) {return floor(din[0]);}
inline double ceilfn( std::vector<double>& din ) {return ceil(din[0]);}
inline double logfn( std::vector<double>& din ) {return log(din[0]);}
inline double log10fn( std::vector<double>& din ) {return log10(din[0]);}
inline double log2fn( std::vector<double>& din ) {return log2(din[0]);}
inline double logbfn( std::vector<double>& din ) {return log(din[0])/log(din[1]);}
inline double sinfn( std::vector<double>& din) {return sin(din[0]);}
inline double cosfn( std::vector<double>& din) {return cos(din[0]);}
inline double tanfn( std::vector<double>& din) {return tan(din[0]);}
inline double gammafn( std::vector<double>& din) {return tgamma(din[0]);}
inline double nchoosekfn( std::vector<double>& din ) {return (double)nchoosek((int)din[0],(int)din[1]);}
inline double expfn(std::vector<double>& din) {return exp(din[0]);}
inline double absfn(std::vector<double>& din) {return abs(din[0]);}
inline double modfn(std::vector<double>& din) {return (int)(din[0]) % (int)(din[1]);}
inline double stirlingfn(std::vector<double>& din) {return stirling((int)din[0], (int)din[1]);};
inline double bellfn(std::vector<double>& din) {return bellNumber((int)din[0]);}
inline double sqrtfn( std::vector<double>& din) {return sqrt(din[0]);}
inline double phifn( std::vector<double>& din) {return totient(din[0]);}

inline double asinfn( std::vector<double>& din) {return asin(din[0]);}
inline double acosfn( std::vector<double>& din) {return acos(din[0]);}
inline double atanfn( std::vector<double>& din) {return atan(din[0]);}

inline double atan2fn(std::vector<double> &din) {return atan2(din[0],din[1]);}
inline double pifn(std::vector<double> &din) {return std::numbers::pi;}