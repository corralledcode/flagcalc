//
// Created by peterglenn on 6/10/24.
//

#ifndef GRAPHIO_H
#define GRAPHIO_H
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>
//#include <bits/regex.h>

#include "graphs.h"









class graphstyle {
public:
    virtual bool applies( std::string sin, std::string& sout ) {return false;}
    virtual void applytograph( std::string sin, std::vector<std::string> vertices, std::vector<std::pair<bool,std::pair<std::string,std::string>>>* moves ) {}
};

class tokenizedgraphstyle : public graphstyle
{
public:
    const char token;
    bool applies(std::string sin, std::string& sout ) override
    {
        sout = sin;
        if (sin.size() > 0)
            if (sin[0] == token)
            {
                sout.erase(0,1);
                return true;
            }
        return false;
    }
    tokenizedgraphstyle(const char tokenin) :token{tokenin} {}
};

class regexgraphstyle : public graphstyle {
public:
    const std::regex r;
    bool applies( std::string sin, std::string& sout ) override {

        std::smatch m;
        std::regex_search(sin, m, r);
        //for (auto s : m)
        //    std::cout << s << ", ";
        //std::cout << "\n";
        if (m.size()>1)
            sout = m[1];
        else
            sout = sin;
        return m.size()>1;
    }
    regexgraphstyle(const std::string rin) :r{rin} {}
};

class verticesforgraphstyle {
protected:
    const std::regex r {"(([[:alpha:]][\\d|_|\\:]*)|\\{([^}]+)\\})"};
public:
    std::vector<std::string> getvertices( std::string sin )
    {
        std::vector<std::string> out {};
        for (std::sregex_iterator p(sin.begin(),sin.end(),r); p!=std::sregex_iterator{}; ++p)
        {
            out.push_back((*p)[1]);
        }
        return out;
    }
};

class negationgraphstyle : public tokenizedgraphstyle {
public:
    negationgraphstyle() : tokenizedgraphstyle('!') {};
    void applytograph( std::string sin,  std::vector<std::string> vertices, std::vector<std::pair<bool,std::pair<std::string,std::string>>>* moves ) override
    {
        for (int i = 0; i < moves->size(); ++i)
            (*moves)[i].first = !(*moves)[i].first;
    }
};

class completegraphstyle : public tokenizedgraphstyle {
public:
    completegraphstyle() : tokenizedgraphstyle('*') {}

    void applytograph( std::string sin,  std::vector<std::string> vertices, std::vector<std::pair<bool,std::pair<std::string,std::string>>>* moves ) override
    {
        moves->clear();
        for (int i = 0; i < vertices.size(); ++i)
            for (int j = i+1; j < vertices.size(); ++j)
                if (vertices[i] != vertices[j])
                    moves->push_back({true,{vertices[i],vertices[j]}});
    }
};

class pathgraphstyle : public tokenizedgraphstyle
{
public:
    pathgraphstyle() : tokenizedgraphstyle('-') {}

    void applytograph(std::string sin, std::vector<std::string> vertices,  std::vector<std::pair<bool,std::pair<std::string,std::string>>>* moves) override
    {
        for (int i = 0; i < vertices.size()-1; ++i)
        {
            moves->push_back({true,{vertices[i],vertices[i+1]}});
        }
    }
};

class radialgraphstyle : public graphstyle {
public:
    bool applies(std::string sin, std::string& sout ) override
    {
        int pos = sin.find("+");
        if (pos != std::string::npos)
        {
            sout = sin.substr(0,pos) + sin.substr(pos+1,sin.size()-pos-1);
            return true;
        } else
            return false;
    }

    void applytograph(std::string sin, std::vector<std::string> vertices,
        std::vector<std::pair<bool, std::pair<std::string, std::string>>>* moves) override {

        int pos = sin.find("+");

        if (pos == std::string::npos)
            return;

        auto vgs = new verticesforgraphstyle();
        std::vector<std::string> v0 = vgs->getvertices(sin);
        std::vector<std::string> v1 = vgs->getvertices(sin.substr(0,pos));
        std::vector<std::string> v2 = vgs->getvertices(sin.substr(pos,sin.size()-pos));
        auto cgs = new completegraphstyle();
        cgs->applytograph(sin,v0,moves);
        std::vector<std::pair<bool, std::pair<std::string, std::string>>> movesinvert {};
        cgs->applytograph( sin,v2,&movesinvert );
        auto ngs = new negationgraphstyle();
        ngs->applytograph(sin.substr(pos,sin.size()-pos),v2,&movesinvert);

        for (auto m : movesinvert)
        {
            moves->push_back(m);
        }

        delete vgs;
        delete cgs;
        delete ngs;
    }
};

class bipartitegraphstyle : public graphstyle
{
public:
    bool applies(std::string sin, std::string& sout ) override
    {
        int pos = sin.find("=");
        if (pos != std::string::npos)
        {
            sout = sin.substr(0,pos) + sin.substr(pos+1,sin.size()-pos-1);
            return true;
        } else
            return false;
    }

    void applytograph(std::string sin, std::vector<std::string> vertices,
        std::vector<std::pair<bool, std::pair<std::string, std::string>>>* moves) override {

        int pos = sin.find("=");

        if (pos == std::string::npos)
            return;

        auto vgs = new verticesforgraphstyle();
        std::vector<std::string> v0 = vgs->getvertices(sin);
        std::vector<std::string> v1 = vgs->getvertices(sin.substr(0,pos));
        std::vector<std::string> v2 = vgs->getvertices(sin.substr(pos,sin.size()-pos));
        auto cgs = new completegraphstyle();
        cgs->applytograph(sin,v0,moves);
        std::vector<std::pair<bool, std::pair<std::string, std::string>>> movesinvertleft {};
        std::vector<std::pair<bool, std::pair<std::string, std::string>>> movesinvertright {};
        cgs->applytograph( sin,v1,&movesinvertleft );
        cgs->applytograph( sin, v2, &movesinvertright );
        auto ngs = new negationgraphstyle();
        ngs->applytograph(sin.substr(pos,sin.size()-pos),v2,&movesinvertleft);
        ngs->applytograph(sin.substr(pos,sin.size()-pos),v2,&movesinvertright);

        for (auto m : movesinvertleft)
        {
            moves->push_back(m);
        }
        for (auto m : movesinvertright)
        {
            moves->push_back(m);
        }

        delete vgs;
        delete cgs;
        delete ngs;
    }
};


inline std::vector<std::string> enumvertices( std::vector<std::string> vsin )
{
    auto vgs = new verticesforgraphstyle;
    std::vector<std::string> v {};
    for (auto s : vsin)
    {
        auto tempv = vgs->getvertices(s);
        for (auto av : tempv)
        {
            if (av == "END" || av == "###")
                break;
            bool found = false;
            for (auto bv : v) {
                found |= (bv == av);
                if (found)
                    break;
            }
            if (!found)
                v.push_back(av);
        }
    }
    delete vgs;
    std::sort(v.begin(),v.end());
    return v;
}

inline void graphstylerecurse( std::vector<std::pair<bool,std::pair<std::string,std::string>>>* moves,
    std::vector<std::string> vertices, std::string sin, std::vector<graphstyle*>* gsv, const int defaultgsidx = 0, const bool repeated = false )
{
    std::string sout {};
    for (int i = 0; (i < gsv->size()); ++i)
    {
        if ((*gsv)[i]->applies( sin, sout ))
        {
            //std::cout << sin << " sin sout " << sout << "\n";
            graphstylerecurse( moves, vertices,sout, gsv, true );
            (*gsv)[i]->applytograph(sin,vertices,moves);
            return;
        }
        //if (sin == sout)
        //    return;

    }
    if (!repeated) // && !sin.empty())
        (*gsv)[defaultgsidx]->applytograph( sin, vertices, moves );
}

inline graphtype* igraphstyle( std::vector<std::string> vsin)
{
    std::vector<graphstyle*> styles {};
    styles.push_back(new completegraphstyle());
    styles.push_back(new negationgraphstyle());
    styles.push_back(new pathgraphstyle());
    styles.push_back(new radialgraphstyle());
    styles.push_back(new bipartitegraphstyle());

    auto v = enumvertices(vsin);
    auto outg = new graphtype(v.size());

    outg->vertexlabels = v;

    zerograph(outg);

    auto vgs = new verticesforgraphstyle();

    for (auto s : vsin)
    {
        auto workingv = vgs->getvertices(s); // allow unsorted and allow repeats
        //auto workingg = new graphtype(workingv.size());
        //zerograph(workingg);
        std::vector<std::pair<bool,std::pair<std::string,std::string>>> moves {};
        graphstylerecurse(&moves,workingv,s,&styles);
        for (auto m : moves)
        {
            for (int k = 0; k < v.size(); ++k)
                for (int l = 0; l < v.size(); ++l)
                    if (m.second.first == v[k] && m.second.second == v[l])
                    {
                        outg->adjacencymatrix[k*outg->dim + l] = m.first;
                        outg->adjacencymatrix[l*outg->dim + k] = m.first;
                    }
        }
    }
    for (auto s : styles)
        delete s;
    return outg;

}









class graphio {
    int id = -1;
    virtual graph readgraph();
    virtual int writegraph(graph g);
    virtual int newgraphid();
};

class graphsqlio : graphio {
    graph readgraph();
    int writegraph(graph g);
    int newgraphid();
};

class graphstreamio : graphio {
    std::ifstream ifs;
    std::istream* s = &std::cin;
    graph readgraph();
    int writegraph(graph g);
};

inline graph graphio::readgraph() {

}

inline int graphio::writegraph(graph g) {

}

inline int graphio::newgraphid() {

}


inline graph graphsqlio::readgraph() {

}

inline int graphsqlio::newgraphid() {

}

inline int graphsqlio::writegraph(graph g) {

}


#endif //GRAPHIO_H
