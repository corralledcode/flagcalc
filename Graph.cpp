//
// Created by peterglenn on 3/13/24.
//

#ifndef GRAPH_H
#define GRAPH_H



#include <utility>
#include <vector>
#include <sys/types.h>
#include "Batchprocesseddata.cpp"
#include <string>

using vertextype = int;
using strtype = std::string;


class Vertices : public Batchprocesseddata<vertextype> {
public:
    vertextype maxvertex = -1;  // should be -1, but by choice vertextype is unsigned
    Vertices(std::vector<vertextype>& verticesin) : Batchprocesseddata<vertextype>(verticesin) {
        bool p = paused();
        pause();
        maxvertex = verticesin.size();
        //for (int n=0;n<maxvertex;++n){
        //    setdata(verticesin[n],n);
        //}
    }
    Vertices(int s) : Batchprocesseddata<vertextype>(s) {
        bool p = paused();
        pause();
        setsize(s);
        if (!p)
            resume();
    }
    Vertices() : Batchprocesseddata<vertextype>() {}
    ~Vertices() {}
    virtual void process() override;
    virtual void setsize(const int s) override {
        bool p = paused();
        pause();
        Batchprocesseddata<vertextype>::setsize(s);
        auto sz = size();
        for (int i = 0; i < sz; ++i)
            setdata( i, i);
        if (!p)
            resume();
    }
};

class Edge : public std::pair<vertextype,vertextype> {
public:
    bool operator<(const Edge& other) const
    {
        return (first < other.first)
               || ((first == other.first) && (second < other.second));
    }
    bool operator>(const Edge& other) const
    {
        return (first > other.first)
               || ((first == other.first) && (second > other.second));
    }
    bool operator==(const Edge& other) const
    {
        return ((first == other.first && second == other.second)
                || (first == other.second && second == other.first));
    }
};


class Edges : public Batchprocesseddata<Edge> {
public:
    vertextype maxvertex=-1;
    void setmaxvertex( int n ) {
        maxvertex = n;
    }
    vertextype* edgematrix; // adjacency matrix
    vertextype* edgeadjacency; // two dimensional array
    bool* vertexadjacency; // two dimensional array
    int maxdegree=0;  // the size of each edge's adjacency list
    Edges(std::vector<Edge> edgesin) : Batchprocesseddata<Edge>(edgesin) {
        maxvertex = computemaxvertex();
        maxdegree = computemaxdegree();
        auto sz = size();
        edgematrix = new vertextype[maxdegree*sz];
        edgeadjacency = new vertextype[sz*sz];
        vertexadjacency = new bool[(maxvertex+1) * (maxvertex+1)];
        computevertexadjacency();
        computeedgematrix();
    }
    Edges(int s) : Batchprocesseddata<Edge>(s) {
        edgematrix = new vertextype[maxdegree*s];
        edgeadjacency = new vertextype[s*s];
    };
    Edges() : Batchprocesseddata<Edge>() {
        edgematrix = new vertextype[0];
        edgeadjacency = new vertextype[0];
    };

    ~Edges() {
        pause();
        //delete[] edgematrix;
        //delete[] edgeadjacency;
    }
    virtual void process() override;
    //virtual void sortdata() override;
    bool operator<(const Edges& other) const
    {
        int n = 0;
        int sz = (size() <= other.size()) ? size() : other.size();
        while ((n<sz) && (getdata(n) == other.getdata(n)))
            n++;
        if (n < sz)
            return getdata(n) < other.getdata(n);
        else
            return false;
    }
    bool operator>(const Edges& other) const
    {
        int n = 0;
        int sz = size() <= other.size() ? size() : other.size();
        while ((n<sz) && (getdata(n) == other.getdata(n)))
            n++;
        if (n < sz)
            return getdata(n) > other.getdata(n);
        else
            return false;
    }
    bool operator==(const Edges& other) const {
        int sz = size();
        if (sz != other.size())
            return false;
        int n = 0;
        bool allfound = true;
        while ((n < sz) && allfound) {
            bool found = false;
            for (int m = 0; (m < sz) && !found; ++m) {
                found = (found || (getdata(n) == other.getdata(m))); // or call sortdata
            }
            allfound = allfound && found;
            ++n;
        }
        return allfound;
    }

    Vertices getvertices();

    inline std::vector<vertextype> vertexneighbors(vertextype v) {
        const int sz = size();
        std::vector<vertextype> adjacent {};
        for (int n = 0; n < sz; ++n) {
            Edge e = getdata(n);
            if (e.first != e.second) {
                if (e.first == v)
                    adjacent.push_back(e.second);
                if (e.second == v)
                    adjacent.push_back(e.first);
            }
        }
        return adjacent;
    }

    inline std::vector<Edge> vertexneighborsasedges(vertextype v) {
        const int sz = size();
        std::vector<Edge> adjacentedges {};
        for (int n = 0; n < sz; ++n) {
            Edge e = getdata(n);
            if (e.first != e.second) {
                if (e.first == v)
                    adjacentedges.push_back(e);
                if (e.second == v)
                    adjacentedges.push_back(e);
            }
        }
        return adjacentedges;
    }

    inline int vertexdegree(vertextype v) {
        const int sz = size();
        std::vector<vertextype> adjacent {};
        for (int n = 0; n < sz; ++n) {
            Edge e = getdata(n);
            if (e.first != e.second) {
                if (e.first == v)
                    adjacent.push_back(e.second);
                if (e.second == v)
                    adjacent.push_back(e.first);
            }
        }
        return adjacent.size();
    }

    inline bool adjacent(vertextype v1, vertextype v2) {
        //if (v1 > maxvertex || v2 > maxvertex)
        //    return false;
        return vertexadjacency[v1*(maxvertex+1) + v2];

    }

    void computevertexadjacency();
private:
    int computemaxvertex() {
        int tempmax = vertextype(0);
        int sz = size();
        for (int n = 0; n < sz; ++n) {
            tempmax = getdata(n).first > tempmax ? getdata(n).first : tempmax;
            tempmax = getdata(n).second > tempmax ? getdata(n).second : tempmax;
        }
        maxvertex = tempmax;
        return maxvertex;
    }

    void computeadjacencymatrix();
    void computeedgematrix();
    int computemaxdegree();
};


class Cover : public Batchprocesseddata<Edges> {
public:
    int maxedgesize = -1;

    Cover(std::vector<Edges> edgesin) : Batchprocesseddata<Edges>(edgesin) {
        //
    }
    Cover(int s) : Batchprocesseddata<Edges>(s) {
        //
    }
    Cover() : Batchprocesseddata<Edges>() {
        //
    }
    ~Cover() {
        //
    }

    void process() override {
        bool p = paused();
        pause();
        Batchprocesseddata<Edges>::process();
        int sz = size();
        for (int n = 0;n < sz;++n) {
            getdata(n).process();
        }
        computemaxedgesize();
        if (!p)
            resume();
    };
    bool coversedges( Edges E ) const {
        int sz = E.size();
        int sz2 = size();
        auto covered = true;
        int n = 0;
        while (covered && n < sz) {
            covered = false;
            int i = 0;
            Edge d = E.getdata(n);
            while (!covered && i < sz2) {
                Edges CE = getdata(i);
                int j = 0;
                int sz3 = E.size();
                while (!covered && (j < sz3)) {
                    covered = (d == CE.getdata(j));
                    ++j;
                }
                ++i;
            }
            ++n;
        }
        return covered;
    }
    void simplifycover();

    int computemaxedgesize() {
        int esz = size();
        maxedgesize = -1;
        for (int k = 0; k < esz; ++k) {
            maxedgesize = ((getdata(k).size() > maxedgesize) ? getdata(k).size() : maxedgesize);
        }
        return maxedgesize;
    }

    bool operator==(const Cover& other) const {
        int sz = size();
        if (sz != other.size())
            return false;
        int n = 0;
        bool allfound = true;
        while ((n < sz) && allfound) {
            bool found = false;
            for (int m = 0; (m < sz) && !found; ++m) {
                found = (found || (getdata(n) == other.getdata(m))); // or call sortdata
            }
            allfound = allfound && found;
            ++n;
        }
        return allfound;
    }


};

class Graph : public Batchprocessed {
public:
    Vertices* vertices;
    Edges* edges;

    Graph(Vertices* verticesin, int vs, Edges* edgesin, int es) : Batchprocessed() {pause(); vertices = verticesin; edges = edgesin; resume();}
    Graph() : Batchprocessed() {}

    void process() override;

};


inline void Vertices::process() {
    //
    Batchprocesseddata<vertextype>::process();
}

inline void Edges::process() {
    //delete[] edgematrix; // getting "double free" message when not commented out
    //delete[] edgeadjacency;
    sortdata();
    auto sz = size();
    for (int n = 0; n < sz; ++n) {
        Edge e = getdata(n);
        maxvertex = ((e.first > maxvertex) ? e.first : maxvertex);
        maxvertex = ((e.second > maxvertex) ? e.second : maxvertex);
    }
    maxdegree = computemaxdegree();
    //std::cout << "maxvertex: " << maxvertex << " maxdegree: " << maxdegree << "size()" << size() << "\n";
    edgeadjacency = new vertextype[sz * maxdegree];
    edgematrix = new vertextype[sz * sz];
    vertexadjacency = new bool[(maxvertex+1) * (maxvertex+1)];
    computeadjacencymatrix();
    computeedgematrix();
    computevertexadjacency();
    Batchprocesseddata<Edge>::process();
}

/* now handled by operator overloading of < and > (see code above)
inline void Edges::sortdata() {
    bool ch = true;
    bool p = paused();
    int sz = size();
    pause();
    while (ch) {
        ch = false;
        for (int i = 0; i < sz - 1; ++i)
            if (getdata(i).first > getdata(i).second) {
                Edge tmp = getdata(i);
                Edge tmp2;
                tmp2.first = tmp.second;
                tmp2.second = tmp.first;
                setdata(tmp2, i);
            }
    }
    if (!p)
        resume(); // implicitly calls inherited sortdata

}
*/

inline int Edges::computemaxdegree() {
    auto szE = size();
    int m = 0;
    auto szV = maxvertex+1;
    auto tally = new int[szV];
    for (int n = 0; n < szV; ++n) {
        tally[n] = 0;
    }
    for (int n = 0; n < szE; ++n) {
        ++tally[getdata(n).first];
        ++tally[getdata(n).second];
    }
    for (int i = 0; i < szV; ++i)
        m = (tally[i] > m) ? tally[i] : m;
    delete[] tally;
    maxdegree = m;
    return maxdegree;
}

inline bool edgesmeet( Edge e1, Edge e2 ) {
    return ((e1.first == e2.first)
            || (e1.first == e2.second)
            || (e1.second == e2.first)
            || (e1.second == e2.second));
}

inline Vertices Edges::getvertices() {
    int sz = size();
    std::vector<vertextype> v {};
    v.resize(sz*2);
    int idx = 0;
    for (int n = 0; n < sz; ++n) {
        v[idx] = getdata(n).first;
        v[idx+1] = getdata(n).second;
        idx += 2;
    }
    v.resize(idx);
    Vertices V {};
    V.readvector(v);
    return V;
}

inline void Edges::computeadjacencymatrix() {
    // this matrix consists of the vertex where two edges meet...
}

inline void Edges::computevertexadjacency() {
    //Vertices v = getvertices();
    //v.removeduplicates();
    vertextype sz = maxvertex+1;
    int Esz = size();
    for (int n = 0; n < sz; ++n) {
        for (int m = 0; m < sz; ++m) {
            bool found = false;
            for (int j = 0; !found && (j < Esz); ++j) {
                Edge e = getdata(j);
                found = (found || (((e.first == n) && (e.second == m)) ||
                        ((e.second == n) && (e.first == m))));
            }
            vertexadjacency[n * sz + m] = found;
            // std::cout << " " << vertexadjacency[n * sz + m];
        }
        // std::cout << "\n";
    }
}


inline void Edges::computeedgematrix() {

/*
    int Esz = size();
    Vertices v = getvertices();
    int sz = v.size();
    for (int j = 0; j < Esz; ++j) {
        int pos = 0;
        for (int k = 0; k < sz; ++k) {
            for (int l = 0; l < Esz; ++l) {
                if (getdata(j).first == getdata(l).first &&)
            }
            if (edgesmeet(getdata(j),getdata(k))) {
                edgematrix[maxdegree*j + pos] = (getdata(k).first == getdata(j).first) ? getdata(k).second : getdata(k).first;
                ++pos;
                std::cout << "Edge " << getdata(j).first << " " << getdata(j).second << " adjacent vertex " << edgematrix[maxdegree*j + pos];
            }
            std::cout << "\n";
        } // now fill each row with zeroes where applicable
    }
*/
}

inline void Cover::simplifycover() { // this should be made instead into a Hellytheory method
    int sz = size();
    std::vector<Edges> Es {};
    Es.clear();
    sortdata();
    for (int n = 0; n < sz; ++n) {
        Edges es = getdata(n);
        int essz = es.size();
        // is every edge e in es covered by some one es2?
        bool allallcovered = false;
        for (int k = 0; (k < sz) && !allallcovered; ++k) {
            if (k != n) {
                Edges es2 = getdata(k);
                // does es2 contain every edge e in es?
                int es2sz = es2.size();
                bool allcovered = true;
                for (int m = 0; (m < essz) && allcovered; ++m) {
                    Edge e = es.getdata(m);
                    bool covered = false;
                    for (int j = 0; (j < es2sz) && !covered; ++j) {
                        Edge e2 = es2.getdata(j);
                        covered = (covered || (e == e2));
                    }
                    allcovered = (allcovered && covered);
                }
                allallcovered = (allallcovered || allcovered);
            }
        }
        if (!allallcovered) {
            Es.push_back(es);
        }
#ifdef VERBOSESIMPLIFYCOVER
        else {
            std::cout << "Removing covered edges ";
            for (int m = 0; m < es.size(); ++m)
                std::cout << "[" << es.getdata(m).first << ", " << es.getdata(m).second << "], ";
            std::cout << "\b\b  \n";
        }
#endif

    }
    readvector(Es);
}

#endif //GRAPH_H
