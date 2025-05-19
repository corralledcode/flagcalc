//
// Created by peterglenn on 7/11/24.
//


#include <cmath>
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

inline double isinffn( std::vector<double>& din )
{
/*    if (din.empty())
    {
        std::cout << "No arguments passed to isinfn\n";
        return 0;
    } */
    return (double)(din[0] == std::numeric_limits<double>::infinity());
}


inline double floorfn( std::vector<double>& din )
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to floorfn\n";
        return 0;
    } */
    return floor(din[0]);
}

inline double ceilfn( std::vector<double>& din )
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to ceilfn\n";
        return 0;
    } */
    return ceil(din[0]);
}

inline double logfn( std::vector<double>& din )
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to logfn\n";
        return 0;
    } */
    return log(din[0]);
}

inline double sinfn( std::vector<double>& din)
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to sinfn\n";
        return 0;
    } */
    return sin(din[0]);
}

inline double cosfn( std::vector<double>& din)
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to cosfn\n";
        return 0;
    } */
    return cos(din[0]);
}

inline double tanfn( std::vector<double>& din)
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to tanfn\n";
        return 0;
    } */
    return tan(din[0]);
}

inline double gammafn( std::vector<double>& din)
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to gammafn\n";
        return 0;
    } */
    return tgamma(din[0]);
}

inline double nchoosekfn( std::vector<double>& din )
{
    /* if (din.size() != 2)
    {
        std::cout << "Wrong number of arguments passed to nchoosekfn\n";
        return 1;
    } */
    return (double)nchoosek((int)din[0],(int)din[1]);
}

inline double expfn(std::vector<double>& din)
{
    /* if (din.size() != 1)
    {
        std::cout << "Wrong number of arguments passed to expfn\n";
        return 0;
    } */
    return exp(din[0]);
}

inline double absfn(std::vector<double>& din)
{
    /* if (din.empty())
    {
        std::cout << "No arguments passed to absfn\n";
        return 0;
    } */
    return abs(din[0]);
}

inline double modfn(std::vector<double>& din) {
    /* if (din.size() != 2)
    {
        std::cout << "Wrong number of arguments passed to modfn\n";
        return 0;
    } */
    return (int)(din[0]) % (int)(din[1]);
}

inline double stirlingfn(std::vector<double>& din)
{
    /* if (din.size() != 2)
    {
        std::cout << "Wrong number of arguments passed to stirlingfn\n";
        return 0;
    } */
    return stirling((int)din[0], (int)din[1]);
};

inline double bellfn(std::vector<double>& din)
{
    /* if (din.size() != 1)
    {
        std::cout << "Wrong number of arguments passed to bellfn\n";
        return 0;
    } */
    return bellNumber((int)din[0]);

}

inline double sqrtfn( std::vector<double>& din)
{
    /* if (din.size() != 1)
    {
        std::cout << "Wrong number of arguments passed to squareroot\n";
        return 0;
    } */
    return sqrt(din[0]);

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


inline double phifn( std::vector<double>& din)
{
    return totient(din[0]);
}

