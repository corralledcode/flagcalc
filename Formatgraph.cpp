//
// Created by peterglenn on 3/13/24.
//

#ifndef HELLYTOOLCPP_FORMATGRAPH_CPP
#define HELLYTOOLCPP_FORMATGRAPH_CPP

#include <string>
#include <iostream>
#include <ostream>
#include <vector>
#include <regex>
#include "Graph.cpp"
#include "Formatdata.cpp"

//{"([a-zA-Z]{1}[\\d_]*)"}

class Formatvertices : public Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>> {
public:
    Formatvertices( Vertices* iin, Batchprocesseddata<strtype>* ein)
        : Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>(*iin,*ein) {
        //
    };
    Formatvertices( Vertices* iin ) : Formatdata<vertextype, strtype, Vertices, Batchprocesseddata<strtype>>(*iin) {
        //
    }
    explicit Formatvertices(std::istream &is) : Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>(is) {
        //
    }
    Formatvertices(int s) : Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>(s) {
        //
    }
    Formatvertices() : Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>() {
        //
    }
    ~Formatvertices() {
        //Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>::~Formatdata();
    }


protected:

    void parseexternal(std::string* str, std::vector<strtype>* out) override {
        out->push_back(*str);
    }

    void parseinternal(std::string* str, std::vector<vertextype>* out) override {
        out->push_back(Formatdata<vertextype,strtype,Vertices,Batchprocesseddata<strtype>>::lookup(*str));
    }
};

class Edgestr : public std::pair<strtype,strtype> {
public:
    bool operator<(const Edgestr& other) const
    {
        return (first < other.first)
               || ((first == other.first) && (second < other.second));
    }
    bool operator>(const Edgestr& other) const
    {
        return (first > other.first)
               || ((first == other.first) && (second > other.second));
    }
    bool operator==(const Edgestr& other) const
    {
        return ((first == other.first && second == other.second)
                || (first == other.second && second == other.first));
    }
};

inline std::ostream& operator<<(std::ostream& os, const Edgestr& e) {
    return os << "[" << e.first << ", " << e.second << "]";
}

inline std::ostream& operator<<(std::ostream& os, const Edge& e) {
    return os << "[" << e.first << ", " << e.second << "]";
}

inline std::ostream& operator<<(std::ostream& os, const Edges& e) {
    int sz = e.size();
    os << "[";
    for (int n = 0; n < sz; ++n) {
        os << "[" << e.getdata(n).first << "," << e.getdata(n).second << "]";
        if (n < sz-1)
            os << ",";
    }
    return os;
}


/*
inline std::istream& operator>>(std::istream& is, Edgestr& e) {

}
*/


class Formatedges : public Formatdata<Edge,Edgestr,Edges,Batchprocesseddata<Edgestr>> {
public:
    Formatedges( Edges* iin, Batchprocesseddata<Edgestr>* ein )
        : Formatdata<Edge,Edgestr,Edges,Batchprocesseddata<Edgestr>>( *iin, *ein ) {
        //
    }
    Formatedges( Edges* iin ) : Formatdata<Edge,Edgestr,Edges,Batchprocesseddata<Edgestr>>(*iin) {
        //
    }
    Formatedges(std::istream &is) : Formatdata<Edge,Edgestr,Edges, Batchprocesseddata<Edgestr>>(is) {
        //
    }
    Formatedges(int s) : Formatdata<Edge,Edgestr,Edges, Batchprocesseddata<Edgestr>>(s) {
        //
    }
    Formatedges() : Formatdata<Edge,Edgestr,Edges,Batchprocesseddata<Edgestr>>() {
        //
    }
    ~Formatedges() {
    }
    Formatvertices* FV {};  // Formatedges is not responsible for memory management of this pointer
    void setvertices( Formatvertices* FVin ) {
        bool b = paused();
        pause();
        int sz = size();
        FV = FVin;
        for (int n = 0; n < sz; ++n) {
            Edge e;
            Edgestr es = edata->getdata(n);
            //std::cout << "n: " << n << ","<< es.first << " "<<es.second<<"\n";
            e.first = FV->lookup(es.first);
            e.second = FV->lookup(es.second);
            //std::cout << "e: " << e.first << " " << e.second << "\n";
            idata->setdata(e,n);
            //std::cout << "idata "<< idata->getdata(n).first << " " << idata->getdata(n).second << "\n";
        }
        if (!b)
            resume();
    }


protected:
    void parseexternal(std::string* str, std::vector<Edgestr>* edge) override {
        if (str->size()==0)
            return;
        std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
        int n = 0;
        std::vector<strtype> v;
        for (std::sregex_iterator p(str->begin(),str->end(),pat); p != std::sregex_iterator{};++p)
            v.push_back((*p)[1]);
        std::sort(v.begin(), v.end());
        int sz = v.size();
        Edgestr e;
        int j = 0;
        for (int m = 0; m < sz; ++m) {
            for (int n = m+1; n < sz; ++n) {
                e.first = v[m];
                e.second = v[n];
                edge->push_back(e);
                ++j;
                //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
            }
        }
    }

    void parseinternal(std::string* str, std::vector<Edge>* edge) override {
        if (str->size()==0)
            return;
        Edgestr es;
        std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
        int n = 0;
        std::vector<strtype> v;
        for (std::sregex_iterator p(str->begin(),str->end(),pat); p != std::sregex_iterator{};++p)
            v.push_back((*p)[1]);
        std::sort(v.begin(), v.end());
        for (int m = 0; m < v.size(); ++m) {
            for (int n = m + 1; n < v.size(); ++n) {
                es.first = v[m];
                es.second = v[n];
                //std::cout << ":: v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
                edge->push_back(lookup(es));
            }
        }
    }
};

class Formatcover : public Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>> {
public:
    Formatcover( Cover* iin, Batchprocesseddata<std::vector<Edgestr>>* ein )
            : Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>( *iin, *ein ) {
        //
    }
    Formatcover( Cover* iin ) : Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>( *iin ) {
        //
    }
    Formatcover(std::istream &is) : Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>(is) {
        //
    }
    Formatcover(int s) : Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>(s) {
        //
    }
    Formatcover() : Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>() {
        //
    }
    ~Formatcover() {
    }
    Formatvertices* FV;  // Formatedges is not responsible for memory management of this pointer
    void setvertices( Formatvertices* FVin ) {

        bool p = paused();
        pause();
        int sz = size();
        FV = FVin;
        for (int n = 0; n < sz; ++n) {
            Edges E = idata->getdata(n);
            std::vector<Edgestr> ES = edata->getdata(n);
            int sz2 = E.size();
            for (int i = 0; i < sz2; ++i) {
                Edge e = E.getdata(i);
                Edgestr es = ES[i];
                e.first = FV->lookup(es.first);
                e.second = FV->lookup(es.second);
                E.setdata(e, i);
            }
        }
        if (!p)
            resume();
    }


    void simplifycover() {
        bool p = paused();
        pause();
        idata->simplifycover();
        Formatdata<Edges,std::vector<Edgestr>,Cover,Batchprocesseddata<std::vector<Edgestr>>>::matchiedata();
        if (!p)
            resume();
    }

protected:
    void parseexternal(std::string* str, std::vector<std::vector<Edgestr>>* edges ) override {
        if (str->size()==0)
            return;

        Edgestr e;
        std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
        int n = 0;
        std::vector<strtype> v;
        for (std::sregex_iterator p(str->begin(),str->end(),pat); p != std::sregex_iterator{};++p)
            v.push_back((*p)[1]);
        std::sort(v.begin(), v.end());

        int sz = v.size();
        std::vector<Edgestr> tempedge {};
        for (int m = 0; m < v.size(); ++m) {
            for (int n = m+1; n < v.size(); ++n) {
                e.first = v[m];
                e.second = v[n];
                tempedge.push_back(e);
                //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
            }
        }
        edges->push_back(tempedge);
    }


    void parseinternal(std::string* str, std::vector<Edges>* edges) override {
        if (str->size()==0)
            return;

        Edgestr e;
        std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
        int n = 0;
        std::vector<strtype> v;
        for (std::sregex_iterator p(str->begin(),str->end(),pat); p != std::sregex_iterator{};++p)
            v.push_back((*p)[1]);
        std::sort(v.begin(), v.end());
        std::vector<Edgestr> tempedge {};
        for (int m = 0; m < v.size(); ++m) {
            for (int n = m+1; n < v.size(); ++n) {
                e.first = v[m];
                e.second = v[n];
                tempedge.push_back(e);
                //std::cout << ":: v[m]: " << v[m] << " v[n]: " << v[n] << "\n";

            }
        }

        Edges etmp = lookup(tempedge);
        etmp.setsize(tempedge.size());
        int sz = etmp.size();
/*        for (int n = 0; n < sz; ++n) {
            std::cout << "[" << etmp.getdata(n).first << "," << etmp.getdata(n).second << "]";
            if (n < sz-1)
                std::cout << ",";
        }
        std::cout << etmp << " (lookup) \n";*/
        edges->push_back(etmp);
    }

};





#endif //HELLYTOOLCPP_FORMATGRAPH_CPP
