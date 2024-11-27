//
// Created by peterglenn on 7/15/24.
//

#ifndef MATHFN_H
#define MATHFN_H

#include <vector>


inline int nchoosek( const int n, const int k);

inline double isinffn( std::vector<double>& din );

inline double floorfn( std::vector<double>& din );

inline double ceilfn( std::vector<double>& din );

inline double logfn( std::vector<double>& din );

inline double sinfn( std::vector<double>& din);

inline double cosfn( std::vector<double>& din);

inline double tanfn( std::vector<double>& din);

inline double gammafn( std::vector<double>& din);

inline double nchoosekfn( std::vector<double>& din );

inline double expfn(std::vector<double>& din);

inline double absfn(std::vector<double>& din);

inline double modfn(std::vector<double>& din);

inline double powfn(std::vector<double>& din);


inline std::map<std::string,std::pair<double (*)(std::vector<double>&),int>> global_fnptrs
    {{"log", {&logfn,1}},
     {"sin", {&sinfn,1}},
     {"cos", {&cosfn,1}},
     {"tan", {&tanfn,1}},
     {"floor", {&floorfn,1}},
     {"ceil", {&ceilfn,1}},
     {"gamma", {&gammafn,1}},
     {"nchoosek", {&nchoosekfn,2}},
     {"exp",{&expfn,1}},
     {"isinf",{&isinffn,1}},
     {"abs",{&absfn,1}},
     {"mod",{modfn,2}},
     {"pow",{powfn,2}}};

#endif //MATHFN_H
