//
// Created by peterglenn on 6/25/24.
//

#ifndef GRAPHOUTCOMEITEM_H
#define GRAPHOUTCOMEITEM_H

#include "workspace.h"
#include "asymp.h"

template<typename T>
class graphoutcome;



class graphitem : public abstractgraphitem {
public:
    std::vector<graphoutcome<int>*> intitems {};
    std::vector<graphoutcome<bool>*> boolitems {};
    std::vector<graphoutcome<double>*> doubleitems {};

    graphitem() : abstractgraphitem() {
        classname = "GRAPH";
        g = nullptr;
        ns= nullptr;
        verbositylevel = VERBOSE_LISTGRAPHS;
    }


    void osmachinereadablegraph(std::ostream &os);
    ~graphitem() {
        for (auto io : intitems) {
            delete &io;
        }
        for (auto bo : boolitems) {
            delete &bo;
        }
        for (auto fo : doubleitems) {
            delete &fo;
        }
    }

};


template <typename T>
class graphoutcome {
public:
    const T value;
    const graphitem* gi;
    graphoutcome(const graphitem* giin, const T newvalue) :value{newvalue}, gi{giin} {}
    virtual std::string name() {return "_abstract";}
    virtual std::string longname() {return "_abstractlongname";}
    virtual void osdata(std::ostream& os) {
        os << "#" << name() << "=" << value << "\n";
    }
    virtual ~graphoutcome() {}
};

template<typename T>
class genericgraphoutcome: public graphoutcome<T> {
protected:
    const std::string _name;
    const std::string _longname;
public:
    genericgraphoutcome(const std::string namein, const std::string longnamein, const graphitem* giin, const int newvalue)
        : graphoutcome<T>(giin,newvalue),_name{namein},_longname{longnamein} {}
    std::string name() override {return _name;}
    std::string longname() override {return _longname;}


};

class commentoutcome : public graphoutcome<std::string> {
public:
    std::string name() override {return "Comment";}
    std::string longname() override {return "User comment";}
};

class fpoutcome : public graphoutcome<int> {
public:
    const FP* fp;
    fpoutcome(const FP* infp, const graphitem* giin, const int newvalue) : graphoutcome<int>(giin,newvalue),fp{infp} {}
    std::string name() override {return "FP";}
    std::string longname() override {return "Fingerprint";}
};

class automorphismsoutcome : public graphoutcome<int> {
public:
    std::string name() override {return "automorphisms";}
    std::string longname() override {return "Number of automorphisms";}
    automorphismsoutcome(const graphitem* giin, const int newvalue) : graphoutcome<int>(giin,newvalue) {}
};

class isomorphismsoutcome : public graphoutcome<int> {
public:
    const graphitem* gi2;
    std::string name() override {return "isomorphisms";}
    std::string longname() override {return "Number of automorphisms with adjacent graph";}
    isomorphismsoutcome(const graphitem* gi2in, const graphitem* giin, const int newvalue) : graphoutcome<int>(giin,newvalue), gi2{gi2in} {}
};

/*
template<typename Tc>
class abstractcriterionoutcome : public graphoutcome<Tc> {
public:
    abstractcriterion<Tc>* cr;
    abstractcriterionoutcome(abstractcriterion<Tc>* crin, const graphitem* giin, Tc newcvalue)
        : graphoutcome<Tc>(giin,newcvalue),cr{crin} {}
    std::string name() override {return cr->shortname(); // "_abstractcriterion";
    std::string longname() override {return cr->name;}
};*/

template<typename Tm>
class abstractmeasureoutcome : public graphoutcome<Tm> {
    std::string _shortname;
    std::string _name;
    public:
    abstractmeasureoutcome(abstractmeasure<Tm>* msin, const graphitem* giin, Tm newmvalue)
        : graphoutcome<Tm>(giin,newmvalue),_shortname{msin->shortname()}, _name{msin->name} {}
    std::string name() override {return _shortname;}
    std::string longname() override {return _name;}
};

template<typename Tm>
class ameasoutcome : public graphoutcome<Tm> {
    std::string _shortname;
    std::string _name;
public:
    ameasoutcome(pameas<Tm>* pamin, const graphitem* giin, Tm newmvalue)
        : graphoutcome<Tm>(giin,newmvalue),_shortname{pamin->shortname}, _name{pamin->name} {}
    std::string name() override {return _shortname;}
    std::string longname() override {return _name;}
};


/*
class embedscriterionoutcome : public abstractcriterionoutcome<bool> {
public:
    const graphitem* flaggi;
    embedscriterionoutcome( const graphitem* flaggiin, const embedscriterion* crin, const graphitem* giin, bool newvalue )
        : abstractcriterionoutcome<bool>(crin, giin, newvalue), flaggi{flaggiin} {}
    std::string flagname() {
        if (flaggi != nullptr)
            return flaggi->name;
        else
            return "<unnamed>"; // should never occur
    }
    std::string name() override {
        return "embeds"+flagname();
    }
    std::string longname() override {
        return "embeds " + flagname() + " criterion";
    }
};
*/


inline void graphitem::osmachinereadablegraph( std::ostream &os ) {
    if (!intitems.empty() || !boolitems.empty() || !doubleitems.empty() || name != "") {
        os << "/* #name=" << name << "\n";
        for (int i = 0; i < intitems.size(); ++i) {
            os << " * ";
            intitems[i]->osdata(os);
        }
        for (int i = 0; i < boolitems.size(); ++i) {
            os << " * ";
            boolitems[i]->osdata(os);
        }
        for (int i = 0; i < doubleitems.size(); ++i) {
            os << " * ";
            doubleitems[i]->osdata(os);
        }
        os << " */\n";
    }
    if (g->vertexlabels.size() != g->dim) {
        char ch = 'a';
        int idx = 1;
        for (int i = 0; i < g->dim; ++i) {
            if (idx == 1)
                g->vertexlabels.push_back( std::string{ ch++ });
            else
                g->vertexlabels.push_back(std::string{ch++} + std::to_string(idx));
            if (ch == 'z') {
                idx++;
                ch = 'a';
            }
        }
    }
    for (auto l : g->vertexlabels) {
        os << l << " ";
    }
    os << "END\n";
    for (int i = 0; i < g->dim-1; ++i) {
        bool emptyline = true;
        for (int j = i+1; j < g->dim; ++j) {
            if (g->adjacencymatrix[i*g->dim + j]) {
                os << g->vertexlabels[i] << g->vertexlabels[j] << " ";
                emptyline = false;
            }
        }
        if (!emptyline)
            os << "\n";
    }
    os << "END\n";
}

class abstractsubobjectitem : public graphitem
{
public:
    graphitem* parentgi;

    std::string shortname;

    std::string str; // used to pass vertex labels, edges, etc.

    std::vector<int> intvertices {};

    // graphtype* g;
    // neighborstype* ns;

    virtual bool ositem( std::ostream& os, std::string verbositylevel ) {
        os << classname << " " << name << ":\n";
        return true;
    }
    virtual bool isitem( std::istream& is ) {return true;}
    virtual void freemem() {}


    abstractsubobjectitem( graphitem* parentgiin, std::string shortnamein, std::string strin )
        : graphitem(), parentgi{parentgiin}, shortname{shortnamein}, str{strin}
    {
        classname = "SUBOBJECT";
        verbositylevel = VERBOSE_SUBOBJECT;
    }

};


class inducedsubgraphitem : public abstractsubobjectitem
{
public:

    graphtype* typeg;
    neighborstype* typens;

    virtual bool ositem( std::ostream& os, std::string verbositylevel ) {
        os << classname << " " << name << ":\n";
        os << "Subgraph of parent " << parentgi->name << ":\n";
        osadjacencymatrix(os,g);
        osneighbors(os,ns);
        return true;
    }


    inducedsubgraphitem( graphitem* parentgraphin, std::string strin )
        : abstractsubobjectitem(parentgraphin, "n", strin)
    {
        auto vgs = new verticesforgraphstyle;
        auto workingv = vgs->getvertices(str); // allow unsorted and allow repeats
        getintvertices(workingv);
        induce();
        delete vgs;
    }

    void getintvertices( std::vector<std::string> strvertices )
    {
        intvertices.clear();
        for (auto s : strvertices)
        {
            bool found = false;
            for (auto i = 0; !found && (i < parentgi->g->vertexlabels.size()); ++i)
            {
                if (s==parentgi->g->vertexlabels[i])
                {
                    found = true;
                    intvertices.push_back(i);
                }
            }
        }
    }

    void induce()
    {
        if (!parentgi)
            return;
        auto pg = parentgi->g;
        //auto pns = parentgraph->ns;
        auto pdim = pg->dim;
        typeg = new graphtype(intvertices.size());
        int dim = typeg->dim;
        for (int i = 0; i < dim; ++i)
        {
            typeg->adjacencymatrix[i*dim + i] = false;
            for (int j = i+1; j < dim; ++j)
            {
                bool b = pg->adjacencymatrix[intvertices[i]*pdim + intvertices[j]];
                typeg->adjacencymatrix[i*dim + j] = b;
                typeg->adjacencymatrix[j*dim + i] = b;
            }
        }
        typens = new neighborstype(typeg);
    }

};

class vertexsetsubobjectitem : public abstractsubobjectitem
{
public:

    vertexsetsubobjectitem( graphitem* parentgraphin, std::string strin )
        : abstractsubobjectitem(parentgraphin, "v", strin) {}

};

class edgesetsubobjectitem : public abstractsubobjectitem
{
public:

    edgesetsubobjectitem( graphitem* parentgraphin, std::string strin )
        : abstractsubobjectitem(parentgraphin, "e", strin) {}

};



template<typename T> abstractsubobjectitem* abstractsubobjectitemfactory(graphitem* parentgi, std::string strin)
{
    return new T(parentgi, strin);
}





#endif //GRAPHOUTCOMEITEM_H
