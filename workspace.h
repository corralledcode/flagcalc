//
// Created by peterglenn on 6/12/24.
//

#ifndef WORKSPACE_H
#define WORKSPACE_H
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
//#include <bits/regex.h>
#include <boost/regex.hpp>

#include "asymp.h"
#include "graphio.h"
#include "graphs.h"
#include "ameas.h"

#define VERBOSE_CMPFINGERPRINTLEVEL "cmp"
#define VERBOSE_ENUMISOMORPHISMSLEVEL "enum"

#define VERBOSE_DONTLISTISOS "Noiso"
#define VERBOSE_LISTGRAPHS "graphs"
#define VERBOSE_LISTFINGERPRINTS "fp"
#define VERBOSE_ISOS "Iso"
#define VERBOSE_RUNTIMES "rt"
#define VERBOSE_VERBOSITYRUNTIME "vrunt"
#define VERBOSE_VERBOSITYFILEAPPEND "vappend"
#define VERBOSE_MINIMAL "min"
#define VERBOSE_MANTELSTHEOREM "Mantel"
#define VERBOSE_FINGERPRINT "Fp"
#define VERBOSE_SAMPLERANDOMMATCHING "srm"
#define VERBOSE_FPMINIMAL "FpMin"
#define VERBOSE_FPNONE "fpnone"
#define VERBOSE_APPLYCRITERION "crit"
#define VERBOSE_SUBOBJECT "subobj"
#define VERBOSE_PAIRWISEDISJOINT "pd"
#define VERBOSE_APPLYSET "set"
#define VERBOSE_SETVERBOSE "allsets"
#define VERBOSE_CRITVERBOSE "allcrit"
#define VERBOSE_MEASVERBOSE "allmeas"
#define VERBOSE_TALLYVERBOSE "alltally"

#define VERBOSE_APPLYSTRING "strmeas"
#define VERBOSE_APPLYGRAPH "measg"
#define VERBOSE_RANDOMSUMMARY "randomizer"

#define VERBOSE_ALL "Noiso graphs fp Iso rt vrunt vappend min Mantel Fp srm FpMin subobj pd set allsets randomizer"
#define VERBOSE_DEFAULT "Noiso graphs fp Iso rt vrunt vappend min Mantel Fp srm FpMin crit rm fpnone vrunt subobj pd set allsets randomizer"

#define VERBOSE_FORDB "db"

#define CMDLINE_ALL "all"
#define CMDLINE_ENUMISOSSORTED "sorted"
#define CMDLINE_ENUMISOSSORTEDVERIFY "sortedverify"
#define CMDLINE_SUBOBJECTS "sub"




inline bool verbositycmdlineincludes( const std::string str, const std::string s2 ) {
    std::string tmp2 = " " + s2 + " ";
    std::string tmp1 = " " + str + " ";
    return (tmp1.find(tmp2) != std::string::npos);
}


inline std::vector<std::pair<std::string,std::string>>  cmdlineparseiterationtwo( const std::vector<std::string> args ) {
    std::vector<std::pair<std::string,std::string>> res {};
    for (int i = 1; i < args.size(); ++i) {

        boost::regex r("([[:alnum:]]+)=((\\w|[[:punct:]]|\\s)*)"); // entire match will be 2 numbers

        boost::smatch m;
        boost::regex_search(args[i], m, r);

        std::vector<std::string> m2 {}; // boost as opposed to std::regex_search, seems to return blank strings...
        for (auto me : m)
            if (me.length() != 0)
                m2.push_back(me);
        if (m2.size() > 2) {
            res.push_back({m2[1],m2[2]});
        } else {
            if (m2.size() > 0) {
                res.push_back( {"default",m2[0]});
            } else {
                res.push_back( {"default",args[i]});
            }
        }
        //for(auto v: m) {
        //    std::cout << v << std::endl;
        //    res.push_back({std::to_string(i),v});
    }
    return res;

}


inline std::vector<std::pair<std::string,std::vector<std::string>>> cmdlineparseiterationthree( const std::string arg ) {
    std::vector<std::string> res;
    std::vector<std::pair<std::string,std::vector<std::string>>> overallres {};
    boost::regex r( "(-|\\w|\\(|\\)|,)+(;|\\s|:)?");

    for (boost::sregex_iterator p(arg.begin(),arg.end(),r); p!=boost::sregex_iterator{}; ++p) {
        boost::regex r2( "(-|\\w)*");
        boost::smatch m2;
        std::string tmp2 = (*p)[0];
        boost::regex_search(tmp2,m2,r2);
        std::string s2 = "default";

        std::vector<std::string> m2d {}; // boost as opposed to std::regex_search, seems to return blank strings...
        for (auto me : m2)
            if (me.length() != 0)
                m2d.push_back(me);

        if (m2d.size() > 0)
            s2 = m2d[0];

        //std::regex r3( "\\((\\w)+,(\\w)+\\)");
        boost::regex r3( "\\(([^\\)]+)\\)" );

        std::vector<std::string> parametersall {};

        for (boost::sregex_iterator p3( tmp2.begin(), tmp2.end(),r3); p3 != boost::sregex_iterator{}; ++p3) {
            parametersall.push_back((*p3)[1]);
        }

        std::vector<std::string> parameters {};

        if (!parametersall.empty()) {

            boost::regex r4( "(.+?)(?:,|$)" );

            for (boost::sregex_iterator p4( parametersall[0].begin(),parametersall[0].end(),r4); p4 != boost::sregex_iterator{}; ++p4) {
                parameters.push_back((*p4)[1]);
            }

        }


        overallres.push_back({s2,parameters});
    }

    return overallres;
}




class workitems {
public:
    std::string classname;
    std::string name;
    std::string verbositylevel;
    virtual bool ositem( std::ostream& os, std::string verbositylevel ) {
        os << classname << " " << name << ":\n";
        return true;
    }
    virtual bool isitem( std::istream& is ) {return true;}
    virtual void freemem() {}
    workitems() {
        classname = "_Unnamed class";
    }
};

class workspace {
public:
    int namesused = 0;
    std::vector<workitems*> items {};
    std::string getuniquename(std::string name) {
        return name + std::to_string(namesused++);

        /*bool unique = true;
        std::string tmpname;
        int namesused = 0;
        if (name == "")
            name = "WORKITEM";
        do {
            tmpname = name + std::to_string(namesused);
            unique = true;
            for (int n = 0; unique && (n < items.size()); ++n) {
                unique = unique && (items[n]->name != tmpname);
            }
            namesused++;
        } while (!unique);
        return tmpname;*/
    }
    workspace() {
    }
};

class abstractgraphitem : public workitems {
public:
    graphtype* g;
    neighbors* ns;
    abstractgraphitem() : workitems() {
        g = nullptr;
        ns = nullptr;
        verbositylevel = VERBOSE_LISTGRAPHS;
        classname = "ABSTRACTGRAPH";
    }
    void freemem() override {
        workitems::freemem();
        delete ns;
        //free (g->adjacencymatrix);
        delete g;
    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );

        if (verbositycmdlineincludes(verbositylevel, VERBOSE_MINIMAL)) {
            os << name << ", dim==" << g->dim << ", edgecount==" << edgecnt(g) << "\n";
        } else {
            if (g != nullptr)
                osadjacencymatrix(os,g);
            osedges(os,g);  // duplicates osneighbors
            if (ns != nullptr)
                osneighbors(os,ns);
        }
        return true;
    }

    bool isitemstr( std::vector<std::string> streamstr) {
        std::vector<std::string> input {};
        //std::string item {};
        bool comment = false;
        int delimetercount = 0;

        for (auto item : streamstr)
        {
            if (item == "END" || item == "###") {
                ++delimetercount;
                if (delimetercount >= 2)
                    break;
                else
                    continue;;
            }
            if (!comment)
            {
                int pos = item.find("/*");
                if (pos != std::string::npos) {
                    if (pos > 0)
                        input.push_back(item.substr(0, pos));
                    comment = true;
                    item = item.substr(pos+2,item.size()-pos-2);

                    //continue;
                }
            }
            if (comment)
            {
                int pos = item.find("*/");
                if (pos != std::string::npos) {
                    if (pos+2 < item.size())
                        input.push_back(item.substr(pos+2,item.size()-pos-2));
                    comment = false;
                }
                continue;
            }
            input.push_back(item);
        }

        this->g = igraphstyle(input);
        ns = new neighbors(this->g);
        return (g->dim > 0);
    }

    bool isitem( std::istream& is) {
        std::vector<std::string> input {};
        std::string item {};
        bool comment = false;
        int delimetercount = 0;
        std::vector<std::string> streamstr {};
        while ((is >> item)) {
            if (item == "END" || item == "###")
            {
                ++delimetercount;
                if (delimetercount >= 2)
                    break;
                else
                    continue;;
            }
            streamstr.push_back(item);
        }
        return isitemstr(streamstr);
    }

};

class randomgraphsitem : public workitems {
public:
    abstractrandomgraph* rg;
    std::vector<std::string> ps;
    virtual bool ositem( std::ostream& os, std::string verbositylevel ) {
        os << classname << " " << name << ": ";
        os << rg->shortname() << " " << rg->name << ": ";
        for (auto item : ps)
            os << item << ", ";
        os << "\b\b  \n";
        return true;
    }
    virtual bool isitem( std::istream& is ) {return true;}
    virtual void freemem() {}
    randomgraphsitem( abstractrandomgraph* rg ) : rg{rg}, workitems() {
        classname = "RANDOMGRAPHS";
        verbositylevel = VERBOSE_RANDOMSUMMARY;
    }
};




class enumisomorphismsitem : public workitems {
public:
    graphtype* g1;
    graphtype* g2;
    std::vector<graphmorphism>* gm;
    enumisomorphismsitem() : workitems() {
        classname = "GRAPHISOS";
        verbositylevel = VERBOSE_ISOS; // use primes
    }
    void freemem() override {
/* already freed by graphitem */
        delete gm;
    }

    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        os << "Total number of isomorphisms == " << gm->size() << "\n";
        if (verbositycmdlineincludes(verbositylevel, VERBOSE_DONTLISTISOS)) {
        } else {
            osgraphmorphisms(os, g1,g2, gm);
        }
        return true;
    }
};

class cmpfingerprintsitem : public workitems {
public:
    std::vector<graphtype*> glist;
    std::vector<neighborstype*> nslist;
    std::vector<FP*> fpslist;
    std::vector<std::string> gnames;

    bool fingerprintsmatch;  // all
    bool fingerprintsdontmatch; // all
    int overallmatchcount = 0;
    std::vector<int> res;
    std::vector<int> sorted {};
    cmpfingerprintsitem() : workitems() {
        classname = "CMPFINGERPRINTS";
        verbositylevel = VERBOSE_LISTFINGERPRINTS;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n]->nscnt > 0) {
                freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                free(fpslist[n]->ns);
            }
        } // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
        {
            if (!verbositycmdlineincludes(verbositylevel, VERBOSE_FPNONE)) {
                for (int n = 0; n < sorted.size(); ++n) {
                    os << "fingerprint of graph "<<gnames[sorted[n]]<<", ordered number " << n+1 << " out of " << sorted.size();
                    if (n < sorted.size()-1) {
                        if (res[n] == 1) {
                            os << " (<)";
                        } else
                            if (res[n] == -1) {
                                os << "(>) (error) ";
                            } else
                                os << "(==)";
                    }
                    os << ":\n";
                    if (verbositycmdlineincludes(verbositylevel, VERBOSE_FPMINIMAL)) {
                        osfingerprintminimal(os,nslist[sorted[n]],fpslist[sorted[n]]->ns, fpslist[sorted[n]]->nscnt);
                    } else
                        osfingerprint(os,nslist[sorted[n]],fpslist[sorted[n]]->ns, fpslist[sorted[n]]->nscnt);
                }

            }
        }

        os << "Ordered, with pairwise results: ";
        for (int n = 0; n < sorted.size(); ++n) {
            os << gnames[sorted[n]] << " ";
            if (n < sorted.size()-1) {
                if (res[n] == 1) {
                    os << "< ";
                } else
                    if (res[n] == -1) {
                        os << "> (error) ";
                    } else
                        os << "== ";
            }
        }
        if (sorted.size()>0)
            os << "\b\n";
        else
            os << "<none>\n";
        if (fingerprintsmatch) {
            os << "Fingerprints MATCH: ";
        } else {
            if (fingerprintsdontmatch) {
                os << "NO fingerprints match: ";
            } else {
                os << "Some fingerprints MATCH and some DON'T MATCH: ";
            }
        }
        os << overallmatchcount << " adjacent pairs out of " << sorted.size()-1 << " match\n";
        return true;
    }
};

template<typename T>
class abstractmeasure;

template<typename Tc, typename Tm>
class checkcriterionmeasureitem : public workitems {
public:
    std::vector<graphtype*> glist;
    std::vector<neighborstype*> nslist;
    std::vector<FP*> fpslist;
    std::vector<std::string> gnames;
    std::vector<Tc> res;
    std::vector<Tm> meas;
    abstractmeasure<Tm> am;
    abstractmeasure<Tc> ac;
    std::vector<int> sorted {};
    checkcriterionmeasureitem(abstractmeasure<Tc> acin, abstractmeasure<Tm> amin ) : workitems(), ac{acin}, am{amin} {
        classname = "APPLYCRITERION";
        verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

/*        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n]->nscnt > 0) {
                freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                free(fpslist[n]->ns);
            }
        }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);

        if (verbositycmdlineincludes(verbositylevel,VERBOSE_CRITVERBOSE))
        {
            os << "Criterion " << ac.name << " results graph-by-graph:\n";
            for (int i = 0; i < this->res.size(); ++i )
            {
                os << this->gnames[i] << ": " << this->meas[i] << "\n";
                // min = this->meas[i] < min ? this->meas[i] : min;
                // sum += this->meas[i];
                // max = this->meas[i] > max ? this->meas[i] : max;
                // cnt++;
            }
        }


        std::vector<std::pair<Tc,int>> count = {};
        count.clear();
        count.resize(0);
        //if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
            os << "Criterion "<< ac.name << " results of graphs:\n";
        for (int n = 0; n < res.size(); ++n) {
            if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL)) {
                os << gnames[n]<<", number " << n+1 << " out of " << sorted.size();
                os << ": " << res[n] << "\n";
            }
            bool found = false;
            for (int i = 0; !found && (i < count.size()); ++i)
                if (count[i].first == res[n]) {
                    count[i].second += 1;
                    found = true;
                }
            if (!found)
                count.push_back({res[n],1});
        }

        for (int i = 0; i < count.size(); ++i)
            os << "result == " << count[i].first << ": " << count[i].second << " out of " << sorted.size() << ", " << (double)count[i].second / (double)sorted.size() << "\n";

        Tm sum = 0;
        int cnt = 0;
        double max = - std::numeric_limits<double>::infinity();
        double min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < res.size(); ++i ) {
            if (res[i]) {
                min = meas[i] < min ? meas[i] : min;
                sum += meas[i];
                max = meas[i] > max ? meas[i] : max;
                cnt++;
            }
        }
        os << "Average, min, max of measure " << am.name << ": " << (double)sum/(double)cnt << ", " << min << ", " << max << "\n";

        return true;
    }
};








template<typename T>
class chkmeasaitem : public workitems
{
public:
    std::vector<graphtype*> glist;
    std::vector<neighborstype*> nslist;
    std::vector<FP*> fpslist;
    std::vector<std::string> gnames;
    std::vector<T> res;
    std::vector<T> meas;
    std::vector<bool> parentbool;
    int parentboolcnt;
    pameas<T>& pam;
    std::vector<int> sorted {};

    chkmeasaitem( pameas<T>& pamin) : workitems(), pam{pamin} {}

};



template<typename Tm>
class checkcontinuousitem : public chkmeasaitem<Tm> {
public:
    checkcontinuousitem( pameas<Tm>& pamin ) : chkmeasaitem<Tm>(pamin) {
        this->classname = "APPLYCRITERION";
        this->verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

/*        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n]->nscnt > 0) {
                freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                free(fpslist[n]->ns);
            }
        }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);

        if (verbositycmdlineincludes(verbositylevel,VERBOSE_MEASVERBOSE))
        {
            os << "Measure " << this->pam.name << " results graph-by-graph:\n";
            for (int i = 0; i < this->res.size(); ++i )
            {
                if (this->parentbool[i])
                {
                    os << this->gnames[i] << ": " << this->meas[i] << "\n";
                    // min = this->meas[i] < min ? this->meas[i] : min;
                    // sum += this->meas[i];
                    // max = this->meas[i] > max ? this->meas[i] : max;
                    // cnt++;
                }
            }
        }


        Tm sum = 0;
        int cnt = 0;
        double max = - std::numeric_limits<double>::infinity();
        double min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < this->parentbool.size(); ++i ) {
            if (this->parentbool[i]) {
                min = this->meas[i] < min ? this->meas[i] : min;
                sum += this->meas[i];
                max = this->meas[i] > max ? this->meas[i] : max;
                cnt++;
            }
        }
        os << "Average, min, max of measure " << this->pam.name << ": " << (double)sum/(double)cnt << ", " << min << ", " << max << "\n";

        return true;
    }
};

template<typename Tc>
class checkbooleanitem : public chkmeasaitem<Tc> {
public:
    checkbooleanitem(pameas<Tc>& pamin ) : chkmeasaitem<Tc>(pamin) {
        this->classname = "APPLYBOOLEANCRITERION";
        this->verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

/*        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n]->nscnt > 0) {
                freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                free(fpslist[n]->ns);
            }
        }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);


        if (verbositycmdlineincludes(verbositylevel,VERBOSE_CRITVERBOSE))
        {
            os << "Criterion " << this->pam.getname() << " results graph-by-graph:\n";
            for (int i = 0; i < this->res.size(); ++i )
            {
                if (this->parentbool[i])
                {
                     os << this->gnames[i] << ": " << this->meas[i] << "\n";
                   // min = this->meas[i] < min ? this->meas[i] : min;
                    // sum += this->meas[i];
                    // max = this->meas[i] > max ? this->meas[i] : max;
                    // cnt++;
                }
            }
        }



        std::vector<std::pair<Tc,int>> count = {};
        count.clear();
        count.resize(0);
        //if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
            os << "Criterion "<< this->pam.getname() << " results of graphs:\n";
        for (int n = 0; n < this->res.size(); ++n) {
            if (!this->parentbool[n])
                continue;
            if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL)) {
                os << this->gnames[n]<<", number " << n+1 << " out of " << this->parentboolcnt;
                os << ": " << this->res[n] << "\n";
            }
            bool found = false;
            for (int i = 0; !found && (i < count.size()); ++i)
            {
                if (count[i].first == this->res[n]) {
                    count[i].second += 1;
                    found = true;
                }
            }
            if (!found)
                count.push_back({this->res[n],1});
        }

        for (int i = 0; i < count.size(); ++i)
            os << "result == " << count[i].first << ": " << count[i].second << " out of " << this->parentboolcnt << ", " << (double)count[i].second / (double)this->parentboolcnt << "\n";


        return true;
    }
};









template<typename Tc>
class checkdiscreteitem : public chkmeasaitem<Tc> {
public:
    checkdiscreteitem(pameas<Tc>& pamin ) : chkmeasaitem<Tc>(pamin) {
        this->classname = "APPLYDISCRETECRITERION";
        this->verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

/*        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n]->nscnt > 0) {
                freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                free(fpslist[n]->ns);
            }
        }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);


        if (verbositycmdlineincludes(verbositylevel,VERBOSE_TALLYVERBOSE))
        {
            os << "Tally " << this->pam.getname() << " results graph-by-graph:\n";
            for (int i = 0; i < this->res.size(); ++i )
            {
                if (this->parentbool[i])
                {
                     os << this->gnames[i] << ": " << this->meas[i] << "\n";
                   // min = this->meas[i] < min ? this->meas[i] : min;
                    // sum += this->meas[i];
                    // max = this->meas[i] > max ? this->meas[i] : max;
                    // cnt++;
                }
            }
        }



        std::vector<std::pair<Tc,int>> count = {};
        count.clear();
        count.resize(0);
        //if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
            os << "Tally "<< this->pam.getname() << " results of graphs:\n";
        for (int n = 0; n < this->res.size(); ++n) {
            if (!this->parentbool[n])
                continue;
            if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL)) {
                os << this->gnames[n]<<", number " << n+1 << " out of " << this->parentboolcnt;
                os << ": " << this->res[n] << "\n";
            }
            bool found = false;
            for (int i = 0; !found && (i < count.size()); ++i)
            {
                if (count[i].first == this->res[n]) {
                    count[i].second += 1;
                    found = true;
                }
            }
            if (!found)
                count.push_back({this->res[n],1});
        }

        for (int i = 0; i < count.size(); ++i)
            os << "result == " << count[i].first << ": " << count[i].second << " out of " << this->parentboolcnt << ", " << (double)count[i].second / (double)this->parentboolcnt << "\n";


        return true;
    }
};


inline void osset( std::ostream& os, itrpos* itr, std::string pre, measuretype t, bool alreadyindented = false )
{
    if (alreadyindented)
    {
        os << "\t" << (t == mtset ? "Set" : "Tuple") << " type output, size == " << itr->getsize() << "\n";
    } else
        os << pre << (t == mtset ? "Set" : "Tuple") << " type output, size == " << itr->getsize() << "\n";


    os << pre << (t == mtset ? "{" : "<");
    alreadyindented = true;

    bool e = itr->ended();
    if (e)
    {
        std::cout << (t == mtset ? "}" : ">");
        return;
    }
    bool newline;
    while (!e)
    {
        valms v = itr->getnext();
        e = itr->ended();
        newline = false;
        switch (v.t)
        {
        case mtbool:
            os << v.v.bv << ", ";
            break;
        case mtdiscrete:
            os << v.v.iv << ", ";
            break;
        case mtcontinuous:
            os << v.v.dv << ", ";
            break;
        case mtset:
        case mttuple:
            {
                std::string pre2 = pre + "\t";
                auto itr2 = v.seti->getitrpos();
                newline = true;
                osset( os, itr2, pre2, v.t, alreadyindented );
                os << (e ? "\n" : ",\n ");
                alreadyindented = false;
                break;
            }
        case mtstring:
            os << v.v.rv << ", ";
            break;
        case mtgraph:
            osadjacencymatrix(os, v.v.nsv->g);
            os << "\n";
            break;
        case mtuncast:
            os << "<uncast> \n";
            break;
        }
    }
    if (!newline)
        os << "\b\b" << (t == mtset ? "}" : ">");
    else
        os << pre << (t == mtset ? "}" : ">");
}

template<typename Tm>
class checksetitem : public chkmeasaitem<Tm> {
public:
    checksetitem( pameas<Tm>& pamin ) : chkmeasaitem<Tm>(pamin) {
        this->classname = "APPLYSETCRITERION";
        // this->verbositylevel = VERBOSE_APPLYSET;
        this->verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        /*        for (int n = 0; n < fpslist.size(); ++n) {
                    if (fpslist[n]->nscnt > 0) {
                        freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                        free(fpslist[n]->ns);
                    }
                }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);


        int cnt = 0;
        int max = 0;
        int min = -1;
        int sizesum = 0;

        if (verbositycmdlineincludes(verbositylevel,VERBOSE_SETVERBOSE))
        {
            for (int i = 0; i < this->res.size(); ++i )
            {
                if (this->parentbool[i])
                {
                    auto itr = this->meas[i];
                    auto pos = itr->getitrpos();
                    std::string pre = "";
                    osset( os, pos, pre, mtset );
                    os << "\n";
                    // min = this->meas[i] < min ? this->meas[i] : min;
                    // sum += this->meas[i];
                    // max = this->meas[i] > max ? this->meas[i] : max;
                    // cnt++;
                    delete pos;
                }
            }
        }


        for (int i = 0; i < this->res.size(); ++i ) {
            if (this->parentbool[i]) {
                auto itr = this->meas[i];
                int size = itr->getsize();
                sizesum += size;
                min = (min == -1 ? size : (size < min ? size : min));
                max = size > max ? size : max;
                cnt++;
            }
        }
        if (cnt > 0)
            os << "Count, average, min, max of set size " << this->pam.name << ": " << cnt << ", " << (double)sizesum/(double)cnt << ", " << min << ", " << max << "\n";
        else
            os << "Count, average, min, max of set size " << this->pam.name << ": 0, undef, undef, undef\n";

        return true;
    }
};

template<typename Tm>
class checktupleitem : public chkmeasaitem<Tm> {
public:
    checktupleitem( pameas<Tm>& pamin ) : chkmeasaitem<Tm>(pamin) {
        this->classname = "APPLYTUPLECRITERION";
        // this->verbositylevel = VERBOSE_APPLYSET;
        this->verbositylevel = VERBOSE_APPLYCRITERION;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        /*        for (int n = 0; n < fpslist.size(); ++n) {
                    if (fpslist[n]->nscnt > 0) {
                        freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                        free(fpslist[n]->ns);
                    }
                }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        // Tm sum = 0;
        // int cnt = 0;
        // double max = 0;
        // double min = std::numeric_limits<double>::infinity();

        int cnt = 0;
        int max = 0;
        int min = -1;
        int sizesum = 0;

        if (verbositycmdlineincludes(verbositylevel,VERBOSE_SETVERBOSE))
        {
            for (int i = 0; i < this->res.size(); ++i ) {
                if (this->parentbool[i])
                {
                    auto itr = this->meas[i];
                    auto pos = itr->getitrpos();
                    std::string pre = "";
                    osset( os, pos, pre, mttuple );
                    os << "\n";
                    // min = this->meas[i] < min ? this->meas[i] : min;
                    // sum += this->meas[i];
                    // max = this->meas[i] > max ? this->meas[i] : max;
                    // cnt++;
                    delete pos;
                }
            }
        }


        for (int i = 0; i < this->res.size(); ++i ) {
            if (this->parentbool[i]) {
                auto itr = this->meas[i];
                int size = itr->getsize();
                sizesum += size;
                min = (min == -1 ? size : (size < min ? size : min));
                max = size > max ? size : max;
                cnt++;
            }
        }
        if (cnt > 0)
            os << "Count, average, min, max of tuple size " << this->pam.name << ": " << cnt << ", " << (double)sizesum/(double)cnt << ", " << min << ", " << max << "\n";
        else
            os << "Count, average, min, max of tuple size " << this->pam.name << ": 0, undef, undef, undef\n";

        // recode this to do coordinate-wise avg, min, max


        return true;
    }
};


template<typename Tc>
class checkstringitem : public chkmeasaitem<Tc> {
public:
    checkstringitem(pameas<Tc>& pamin ) : chkmeasaitem<Tc>(pamin) {
        this->classname = "APPLYSTRINGCRITERION";
        this->verbositylevel = VERBOSE_APPLYSTRING;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        /*        for (int n = 0; n < fpslist.size(); ++n) {
                    if (fpslist[n]->nscnt > 0) {
                        freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                        free(fpslist[n]->ns);
                    }
                }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        std::vector<std::pair<Tc,int>> count = {};
        count.clear();
        count.resize(0);
        //if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
        os << "String "<< this->pam.getname() << " results of graphs:\n";
        for (int n = 0; n < this->res.size(); ++n) {
            if (!this->parentbool[n])
                continue;
            if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL)) {
                os << this->gnames[n]<<", number " << n+1 << " out of " << this->parentboolcnt;
                os << ": " << this->res[n] << "\n";
            }
            bool found = false;
            for (int i = 0; !found && (i < count.size()); ++i)
            {
                if (count[i].first == this->res[n]) {
                    count[i].second += 1;
                    found = true;
                }
            }
            if (!found)
                count.push_back({this->res[n],1});
        }

        for (int i = 0; i < count.size(); ++i)
            os << "result == " << count[i].first << ": " << count[i].second << " out of " << this->parentboolcnt << ", " << (double)count[i].second / (double)this->parentboolcnt << "\n";


        return true;
    }
};

template<typename Tc>
class checkgraphitem : public chkmeasaitem<Tc> {
public:
    checkgraphitem(pameas<Tc>& pamin ) : chkmeasaitem<Tc>(pamin) {
        this->classname = "APPLYGRAPHCRITERION";
        this->verbositylevel = VERBOSE_APPLYGRAPH;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        /*        for (int n = 0; n < fpslist.size(); ++n) {
                    if (fpslist[n]->nscnt > 0) {
                        freefps(fpslist[n]->ns,fpslist[n]->nscnt);
                        free(fpslist[n]->ns);
                    }
                }*/ // no: the format has changed to a vector of pointers

    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        for (int n = 0; n < this->res.size(); ++n)
        {

            osadjacencymatrix(os, this->res[n]->g);
            os << "\n";
        }
        return true;
    }
};






class timedrunitem : public workitems {
public:
    long duration;

    timedrunitem() : workitems() {
        classname = "TIMEDRUN";
        verbositylevel = VERBOSE_RUNTIMES;
    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << ((double)duration)/1000000<< "\n";
        return true;
    }

};

class mantelstheoremitem : public workitems {
public:
    double max;
    int limitdim;
    int outof;

    mantelstheoremitem() : workitems() {
        classname = "MANTELSTHM";
        verbositylevel = VERBOSE_MANTELSTHEOREM;
    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << "Asymptotic approximation at limitdim == " << limitdim << ", outof == " << outof << ": " << max << "\n";
        os << "(n^2/4) == " << limitdim * limitdim / 4.0 << "\n";
        return true;
    }

};


class legacysamplerandommatchinggraphsitem : public workitems {
public:
    double percent = -1;
    int cnt = 0;
    int outof = 0;
    int dim = 0;
    std::string rgname {};
    legacysamplerandommatchinggraphsitem() : workitems() {
        classname = "LEGACYSAMPLEGRAPHS";
        verbositylevel = VERBOSE_SAMPLERANDOMMATCHING;
    }

    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        percent = double(cnt)/double(outof);
        os << "Probability amongst \""<< rgname << "\", dimension "<<dim<<"\n";
        os << "of fingerprints matching is " << cnt << " out of " << outof << " == " << percent << "\n";
        return true;
    }


};

class samplerandommatchinggraphsitem : public workitems {
public:
    double percent = -1;
    int cnt = 0;
    int outof = 0;
    //abstractsubobjectitem* asoi {};
    samplerandommatchinggraphsitem() : workitems() {
        classname = "SAMPLEGRAPHS";
        verbositylevel = VERBOSE_SAMPLERANDOMMATCHING;
    }

    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        percent = double(cnt)/double(outof);
        os << "Probability ";
        os << "of fingerprints matching is " << cnt << " out of " << outof << " == " << percent << "\n";
        return true;
    }


};




#endif //WORKSPACE_H
