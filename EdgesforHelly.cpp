//
// Created by peterglenn on 3/16/24.
//

#ifndef HELLYTOOLCPP_EDGESFORHELLY_CPP
#define HELLYTOOLCPP_EDGESFORHELLY_CPP

#include "Graph.cpp"

class EdgesforHelly : public Edges {
public:
    std::vector<std::tuple<vertextype,vertextype,vertextype>> triangles;
    EdgesforHelly( std::vector<Edge> edgein) : Edges(edgein) {
        //
    }
    EdgesforHelly( int s ) : Edges(s) {
        //
    }
    EdgesforHelly() : Edges() {
        //
    }
    ~EdgesforHelly() {
        //
    }
    void process() override;

};

inline void EdgesforHelly::process() {
    Edges::process();
    auto b = paused();
    int sz = size();
    pause();
    triangles.clear();
    for (vertextype v1 = vertextype(0); v1 < maxvertex; ++v1) {
        for (vertextype v2 = v1+1; v2 < maxvertex; ++v2) {
            for (vertextype v3 = v2+1; v3 < maxvertex; ++v3 ) {
                bool b1,b2,b3 {false};
                for (int i = 0; i < sz; ++i) {
                    Edge e = getdata(i);
                    b1 = b1 || (v1 == e.first && v2 == e.second);
                    b2 = b2 || (v1 == e.first && v3 == e.second);
                    b3 = b3 || (v2 == e.first && v3 == e.second);
                }
                if (b1 && b2 && b3)
                    triangles.push_back({v1,v2,v3});
            }
        }
    }
}


#endif //HELLYTOOLCPP_EDGESFORHELLY_CPP
