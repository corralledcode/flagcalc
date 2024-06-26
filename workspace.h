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

        std::vector<std::string> parameters {};

        for (std::sregex_iterator p3( tmp2.begin(), tmp2.end(),r3); p3 != std::sregex_iterator{}; ++p3) {
            parameters.push_back((*p3)[1]);
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
    }
    bool ositem( std::ostream& os, std::string verbositylevel ) override {
        workitems::ositem( os, verbositylevel );

        //to do: would be nice to have a list of edges
        //and a labelling for the adjacency matrix;
        //add vertexlabels to graph struct type
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

    bool isiteminternal( std::string tmp1a, std::string tmp2a, std::string tmp1b, std::string tmp2b ) {
        std::vector<std::string> vertexlabels {}; // ultimately store this in workitem to use for readouts
        if (tmp2a.size() > 0) {
            std::regex pat{"([\\w]+)"};

            for (std::sregex_iterator p(tmp2a.begin(), tmp2a.end(), pat); p != std::sregex_iterator{}; ++p) {
                std::string tmp3;
                tmp3 = (*p)[1];
                vertexlabels.push_back(tmp3);
            }

            // idata->removeduplicates(); must not remove duplicates yet... wait until setvertices has been called
            this->g = new graphtype(vertexlabels.size());
            int dim = this->g->dim;
            for (int i = 0; i <  dim; ++i)
                for (int n = 0; n< dim; ++n)
                    this->g->adjacencymatrix[n*dim + i] = false;


            g->vertexlabels.resize(vertexlabels.size());
            for (int i = 0; i < vertexlabels.size(); ++i)
                g->vertexlabels[i] = vertexlabels[i];


        } else {
            this->g = new graphtype(0);
            this->ns = new neighbors(this->g);
            return false;   // in case only one graph is given, default to computing automorphisms
        }



        std::vector<std::string> edgecommands {};
        if (tmp2b.size() > 0) {
            //std::regex pat{"([[:punct:]]*[\\w]+)|(\\w+\\+\\w+)"};

            //std::regex pat{"([[:punct:]]*[\\w]+)"};
            std::regex pat{"([[:punct:]|\\w]+)"};

            for (std::sregex_iterator p(tmp2b.begin(), tmp2b.end(), pat); p != std::sregex_iterator{}; ++p) {
                std::string tmp3;
                tmp3 = (*p)[1];
                edgecommands.push_back(tmp3);
                //std::cout << tmp3 << ", ";
            }
            //std::cout << "\n";

            // idata->removeduplicates(); must not remove duplicates yet... wait until setvertices has been called
        }

        if (edgecommands.size() >= 0) {
            std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
            for( int n = 0;n< edgecommands.size(); ++n) {
                std::vector<std::string> v;
                bool cmdomit = false;
                bool cmdcomplete = true;
                bool cmdline = false;
                bool radial = false;
                std::string radialstr1 {};
                std::string radialstr2 {};
                if (edgecommands[n].size() > 0) {
                    if (edgecommands[n][0] == '!') {
                        cmdomit = true;
                        if (edgecommands[n].size()>1) {
                            edgecommands[n] = edgecommands[n].substr(1,edgecommands[n].size()-1);
                        }
                    }
                    if (edgecommands[n][0] == '*')
                        cmdcomplete=true;
                    if (edgecommands[n][0] == '-') {
                        cmdline = true;
                        cmdcomplete = false;
                    }
                }
                int pluspos = edgecommands[n].find("+");
                //std::cout << "edgecommands[n] " << edgecommands[n] << " , " << pluspos << "\n";
                if (pluspos != std::string::npos) {
                    if (cmdline)
                        break;
                    cmdcomplete = false;
                    radial = true;
                    radialstr1 = edgecommands[n].substr(0,pluspos);
                    radialstr2 = edgecommands[n].substr(pluspos+1,edgecommands[n].size()-pluspos-1);
                    //std::cout << "radialstr1,2 " << radialstr1 << ", " << radialstr2 << "\n";
                    std::string radialstrcombined = radialstr1 + radialstr2;
                    for (std::sregex_iterator p(radialstrcombined.begin(),radialstrcombined.end(),pat); p != std::sregex_iterator{};++p)
                        v.push_back((*p)[1]);

                    std::sort(v.begin(), v.end());
                    int sz = v.size();
                    //std::cout<< "v.size == " << v.size() << "\n";
                    // connect all pairs within the sequence of vertices
                    for (int m = 0; m < sz; ++m) {
                        for (int n = m+1; n < sz; ++n) {
                            int i = 0;
                            while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                                ++i;
                            int j = 0;
                            while( j < vertexlabels.size() && vertexlabels[j] != v[n])
                                ++j;
                            if (j < vertexlabels.size() && i < vertexlabels.size()) {
                                if (vertexlabels[j] == v[n] && vertexlabels[i] == v[m]) {
                                    if (j != i) {
                                        g->adjacencymatrix[i*g->dim + j] = !cmdomit;
                                        g->adjacencymatrix[j*g->dim + i] = !cmdomit;
                                        //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
                                    }
                                }
                            }
                        }
                    }

                    v.clear(); // now exclude all pairs within the second sequence of vertices amongst itself
                    for (std::sregex_iterator p(radialstr2.begin(),radialstr2.end(),pat); p != std::sregex_iterator{};++p)
                        v.push_back((*p)[1]);
                    cmdomit = !cmdomit;
                    sz = v.size();
                    std::sort(v.begin(), v.end());
                    //std::cout<< "v.size == " << v.size() << "\n";
                    // connect all pairs within the sequence of vertices
                    for (int m = 0; m < sz; ++m) {
                        for (int n = m+1; n < sz; ++n) {
                            int i = 0;
                            while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                                ++i;
                            int j = 0;
                            while( j < vertexlabels.size() && vertexlabels[j] != v[n])
                                ++j;
                            if (j < vertexlabels.size() && i < vertexlabels.size()) {
                                if (vertexlabels[j] == v[n] && vertexlabels[i] == v[m]) {
                                    if (j != i) {
                                        g->adjacencymatrix[i*g->dim + j] = !cmdomit;
                                        g->adjacencymatrix[j*g->dim + i] = !cmdomit;
                                        //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
                                    }
                                }
                            }
                        }
                    }


                } else {
                    for (std::sregex_iterator p(edgecommands[n].begin(),edgecommands[n].end(),pat); p != std::sregex_iterator{};++p)
                        v.push_back((*p)[1]);
                    if (!cmdline)
                        std::sort(v.begin(), v.end());
                    int sz = v.size();
                    //std::cout<< "v.size == " << v.size() << "\n";
                    if (cmdcomplete) {
                        // connect all pairs within the sequence of vertices
                        for (int m = 0; m < sz; ++m) {
                            for (int n = m+1; n < sz; ++n) {
                                int i = 0;
                                while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                                    ++i;
                                int j = 0;
                                while( j < vertexlabels.size() && vertexlabels[j] != v[n])
                                    ++j;
                                if (j < vertexlabels.size() && i < vertexlabels.size()) {
                                    if (vertexlabels[j] == v[n] && vertexlabels[i] == v[m]) {
                                        if (j != i) {
                                            g->adjacencymatrix[i*g->dim + j] = !cmdomit;
                                            g->adjacencymatrix[j*g->dim + i] = !cmdomit;
                                            //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (cmdline) {
                        for (int m = 0; m < (sz-1); ++m) {
                            int i = 0;
                            while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                                ++i;
                            int j = 0;
                            while( j < vertexlabels.size() && vertexlabels[j] != v[m+1])
                                ++j;
                            if (j < vertexlabels.size() && i < vertexlabels.size()) {
                                if (vertexlabels[i] == v[m] && vertexlabels[j] == v[m+1]) {
                                    if (i != j) {
                                        g->adjacencymatrix[i*g->dim + j] = !cmdomit;
                                        g->adjacencymatrix[j*g->dim + i] = !cmdomit;
                                        //std::cout << "v[m]: " << v[m] << " v[m+1]: " << v[m+1] << "\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ns = new neighbors(this->g);
        //ns->computeneighborslist();
        return (g->dim > 0);   // for now no support for trivial empty graphs
    }


    bool isitem( std::istream& is) {
        //auto p = paused();
        //pause();
        //if (!idata || !edata)
        //    throw std::exception();
        int s = 0;

        std::vector<std::string> eresa{}; // not used
        std::string tmp1a = "";
        std::string tmp2a = "";
        while ((is >> tmp1a) && (tmp1a != "END") && (tmp1a != "###")) {
            if (tmp1a == "/*") {
                bool res = bool(is >> tmp1a);
                while (res && (tmp1a != "*/"))
                    res = bool(is >> tmp1a);
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
        return isiteminternal(tmp1a,tmp2a, tmp1b, tmp2b);
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
        for (int i = 0; i < sorted.size(); ++i ) {
            if (res[i]) {
                sum += meas[i];
                cnt++;
            }
        }
        os << "Average of measure " << am.name << ": " << (float)sum/(float)cnt << "\n";

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
