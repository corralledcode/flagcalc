//
// Created by peterglenn on 7/11/24.
//


#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "mathfn.h"

inline int nchoosek( const int n, const int k) {
    if (k == 0)
        return 1;
    return (n* nchoosek(n-1,k-1))/k;
}


inline double floorfn( std::vector<double>& din )
{
    if (din.empty())
    {
        std::cout << "No arguments passed to floorfn\n";
        return 0;
    }
    return floor(din[0]);
}

inline double ceilfn( std::vector<double>& din )
{
    if (din.empty())
    {
        std::cout << "No arguments passed to ceilfn\n";
        return 0;
    }
    return ceil(din[0]);
}

inline double logfn( std::vector<double>& din )
{
    if (din.empty())
    {
        std::cout << "No arguments passed to logfn\n";
        return 0;
    }
    return log(din[0]);
}

inline double sinfn( std::vector<double>& din)
{
    if (din.empty())
    {
        std::cout << "No arguments passed to sinfn\n";
        return 0;
    }
    return sin(din[0]);
}

inline double cosfn( std::vector<double>& din)
{
    if (din.empty())
    {
        std::cout << "No arguments passed to cosfn\n";
        return 0;
    }
    return cos(din[0]);
}

inline double tanfn( std::vector<double>& din)
{
    if (din.empty())
    {
        std::cout << "No arguments passed to tanfn\n";
        return 0;
    }
    return tan(din[0]);
}

inline double gammafn( std::vector<double>& din)
{
    if (din.empty())
    {
        std::cout << "No arguments passed to gammafn\n";
        return 0;
    }
    return tgamma(din[0]);
}

inline double nchoosekfn( std::vector<double>& din )
{
    if (din.size() != 2)
    {
        std::cout << "Wrong number of arguments passed to nchoosekfn\n";
        return 1;
    }
    return nchoosek(din[0],din[1]);
}

inline double expfn(std::vector<double>& din)
{
    if (din.empty())
    {
        std::cout << "No arguments passed to expfn\n";
        return 0;
    }
    return exp(din[0]);
}
