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
    std::vector<graphoutcome<float>*> floatitems {};

    graphitem() : abstractgraphitem() {
        classname = "GRAPH";
        g = nullptr;
        ns= nullptr;
        verbositylevel = VERBOSE_LISTGRAPHS;
    }

    bool isitem( std::istream& is) {
        int s = 0;

        std::vector<std::string> eresa{}; // not used
        std::string tmp1a = "";
        std::string tmp2a = "";
        while ((is >> tmp1a) && (tmp1a != "END") && (tmp1a != "###")) {
            if (tmp1a == "/*") {
                bool res = bool(is >> tmp1a);
                while (res && (tmp1a != "*/")) {
                    if (tmp1a.substr(0,6) == "#name=") {
                        std::string tmpname = tmp1a.substr(6,tmp1a.length()-6);
                        //std::cout << tmpname << "\n";
                        if (tmpname != "")
                            name = tmpname;
                    }
                    res = bool(is >> tmp1a);
                }
                continue;
            }
            eresa.push_back(tmp1a);
            tmp2a += tmp1a + " ";
            tmp1a = "";
            s++;
        }
        s = 0;

        std::string tmp1b = "";
        std::string tmp2b = "";

        std::vector<std::string> eresb {}; // not used
        while ((is >> tmp1b) && (tmp1b != "END") && (tmp1b != "###")) {
            if (tmp1b == "/*") {
                bool res = bool(is >> tmp1b);
                while (res && (tmp1b != "*/"))
                    res = bool(is >> tmp1b);
                continue;
            }
            eresb.push_back(tmp1b);
            tmp2b += tmp1b + " ";
            tmp1b = "";
            s++;
        }
        return isiteminternal(tmp1a,tmp2a, tmp1b, tmp2b );
    }

    void osmachinereadablegraph(std::ostream &os);
    ~graphitem() {
        for (auto io : intitems) {
            delete io;
        }
        for (auto bo : boolitems) {
            delete bo;
        }
        for (auto fo : floatitems) {
            delete fo;
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

template<typename Tc>
class abstractcriterionoutcome : public graphoutcome<Tc> {
public:
    abstractcriterion<Tc>* cr;
    abstractcriterionoutcome(abstractcriterion<Tc>* crin, const graphitem* giin, Tc newcvalue)
        : graphoutcome<Tc>(giin,newcvalue),cr{crin} {}
    std::string name() override {return cr->shortname(); /*"_abstractcriterion";*/}
    std::string longname() override {return cr->name;}
};

template<typename Tm>
class abstractmeasureoutcome : public graphoutcome<Tm> {
public:
    abstractmeasure<Tm>* ms;
    abstractmeasureoutcome(abstractmeasure<Tm>* msin, const graphitem* giin, Tm newmvalue)
        : graphoutcome<Tm>(giin,newmvalue),ms{msin} {}
    std::string name() override {return ms->shortname(); /*"_abstractcriterion"*/}
    std::string longname() override {return ms->name;}
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
    if (!intitems.empty() || !boolitems.empty() || !floatitems.empty() || name != "") {
        os << "/* #name=" << name << "\n";
        for (int i = 0; i < intitems.size(); ++i) {
            os << " * ";
            intitems[i]->osdata(os);
        }
        for (int i = 0; i < boolitems.size(); ++i) {
            os << " * ";
            boolitems[i]->osdata(os);
        }
        for (int i = 0; i < floatitems.size(); ++i) {
            os << " * ";
            floatitems[i]->osdata(os);
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



#endif //GRAPHOUTCOMEITEM_H
