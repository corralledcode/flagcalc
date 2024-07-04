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

#include "asymp.h"
#include "graphio.h"
#include "graphs.h"
#include "measure.cpp"

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

#define VERBOSE_ALL "Noiso graphs fp Iso rt vrunt vappend min Mantel Fp srm FpMin"
#define VERBOSE_DEFAULT "Noiso graphs fp Iso rt vrunt vappend min Mantel Fp srm FpMin"

#define VERBOSE_FORDB "db"

#define CMDLINE_ALL "all"
#define CMDLINE_ENUMISOSSORTED "sorted"
#define CMDLINE_ENUMISOSSORTEDVERIFY "sortedverify"

inline bool verbositycmdlineincludes( const std::string str, const std::string s2 ) {
    std::string tmp2 = " " + s2 + " ";
    std::string tmp1 = " " + str + " ";
    return (tmp1.find(tmp2) != std::string::npos);
}

inline std::vector<std::pair<std::string,std::string>>  cmdlineparseiterationtwo( const std::vector<std::string> args ) {
    std::vector<std::pair<std::string,std::string>> res {};
    for (int i = 1; i < args.size(); ++i) {

        std::regex r("([[:alnum:]]+)=((\\w|[[:punct:]]|\\s)*)"); // entire match will be 2 numbers

        std::smatch m;
        std::regex_search(args[i], m, r);

        if (m.size() > 2) {
            res.push_back({m[1],m[2]});
        } else {
            if (m.size() > 0) {
                res.push_back( {"default",m[0]});
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
    std::regex r( "(-|\\w|\\(|\\)|,)+(;|\\s|:)?");

    for (std::sregex_iterator p(arg.begin(),arg.end(),r); p!=std::sregex_iterator{}; ++p) {
        std::regex r2( "(-|\\w)*");
        std::smatch m2;
        std::string tmp2 = (*p)[0];
        std::regex_search(tmp2,m2,r2);
        std::string s2 = "default";
        if (m2.size() > 0)
            s2 = m2[0];

        //std::regex r3( "\\((\\w)+,(\\w)+\\)");
        std::regex r3( "\\(([^\\)]+)\\)" );

        std::vector<std::string> parametersall {};

        for (std::sregex_iterator p3( tmp2.begin(), tmp2.end(),r3); p3 != std::sregex_iterator{}; ++p3) {
            parametersall.push_back((*p3)[1]);
        }

        std::vector<std::string> parameters {};

        if (!parametersall.empty()) {

            std::regex r4( "(.+?)(?:,|$)" );

            for (std::sregex_iterator p4( parametersall[0].begin(),parametersall[0].end(),r4); p4 != std::sregex_iterator{}; ++p4) {
                parameters.push_back((*p4)[1]);
            }


/*            for (auto f4 : parameters)
                std::cout << "parameters " << f4 << " // ";
            std::cout << "\n";
*/
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
                    continue;
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
            if (item == "END" || item == "###") {
                ++delimetercount;
                if (delimetercount >= 2)
                    break;
                else
                    continue;;

                streamstr.push_back(item);
            }
        }
        return isitemstr(streamstr);
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
        std::vector<std::pair<Tc,int>> count = {};
        //if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL))
            os << "Criterion "<< ac.name << " results of graphs:\n";
        for (int n = 0; n < sorted.size(); ++n) {
            if (!verbositycmdlineincludes(verbositylevel,VERBOSE_MINIMAL)) {
                os << gnames[n]<<", number " << n+1 << " out of " << sorted.size();
                os << ": " << res[n] << "\n";
            }
            bool found = false;
            for (int i = 0; !found && (i < count.size()); ++i)
                if (count[i].first == res[n]) {
                    ++(count[i].second);
                    found = true;
                }
            if (!found)
                count.push_back({res[n],1});
        }

        for (int i = 0; i < count.size(); ++i)
            os << "result == " << count[i].first << ": " << count[i].second << " out of " << sorted.size() << ", " << (float)count[i].second / (float)sorted.size() << "\n";

        Tm sum = 0;
        int cnt = 0;
        float max = 0;
        float min = -1;
        bool first = true;
        for (int i = 0; i < sorted.size(); ++i ) {
            if (res[i]) {
                min = first || (meas[i] < min) ? meas[i] : min;
                first = false;
                sum += meas[i];
                max = meas[i] > max ? meas[i] : max;
                cnt++;
            }
        }
        os << "Average, min, max of measure " << am.name << ": " << (float)sum/(float)cnt << ", " << min << ", " << max << "\n";

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
        os << ((float)duration)/1000000<< "\n";
        return true;
    }

};

class mantelstheoremitem : public workitems {
public:
    float max;
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


class samplerandommatchinggraphsitem : public workitems {
public:
    float percent = -1;
    int cnt = 0;
    int outof = 0;
    int dim = 0;
    std::string rgname {};
    samplerandommatchinggraphsitem() : workitems() {
        classname = "SAMPLEGRAPHS";
        verbositylevel = VERBOSE_SAMPLERANDOMMATCHING;
    }

    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        percent = float(cnt)/float(outof);
        os << "Probability amongst \""<< rgname << "\", dimension "<<dim<<"\n";
        os << "of fingerprints matching is " << cnt << " out of " << outof << " == " << percent << "\n";
        return true;
    }


};





#endif //WORKSPACE_H
