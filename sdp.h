//
// Created by peter on 2026-09-22.
//

#ifndef FLAGCALC_SDP_H
#define FLAGCALC_SDP_H


#include <iostream>
    #include <scs/glbopts.h> // Essential structure setup macro configuration
    #include <scs/scs.h>     // Contains scs_finish declaration
    #include <scs/util.h>    // Contains memory utilities (scs_malloc, scs_free)
#include <cmath>
#include <vector>



// from Gemini

inline void cleanup_scs(ScsData *data, ScsCone *cone, ScsSolution *sol, ScsWork *work) {
    // 1. Free the internal solver workspace allocated by scs_init()
    if (work) {
        scs_finish(work);
    }

    // 2. Free the problem structures you allocated manually
    if (data) {
        // Free the sparse matrix A structures if you allocated them via scs_malloc
        if (data->A) {
            scs_free(data->A->x);
            scs_free(data->A->i);
            scs_free(data->A->p);
            scs_free(data->A);
        }
        scs_free(data->b);
        scs_free(data->c);
        scs_free(data);
    }

    // 3. Free the cone structure arrays
    if (cone) {
        scs_free(cone->q); // Second-order cone array
        scs_free(cone->s); // Semidefinite cone array
        scs_free(cone->p); // Power cone array (if you used it earlier!)
        scs_free(cone);
    }

    // 4. Free the solution vectors populated by scs_solve()
    if (sol) {
        scs_free(sol->x);
        scs_free(sol->y);
        scs_free(sol->s);
        scs_free(sol);
    }
}

// Returns the compressed 1D vector index for symmetric matrix coordinates (i, j)
inline int get_tri_idx(int i, int j, int n) {
    if (i < j) std::swap(i, j);
    return j * n - (j - 1) * j / 2 + (i - j);
}

inline double compute_lovasz_theta( int n, std::vector<std::pair<int,int>>& edges )
{

    const int num_vars = n * (n + 1) / 2;

    // 2. Objective Function Configuration
    std::vector<scs_float> c(num_vars, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            int idx = get_tri_idx(i, j, n);
            if (i == j) {
                c[idx] = -1.0;
            } else {
                // Keep the off-diagonal multiplier scaled correctly for the objective
                c[idx] = -2.0 / std::sqrt(2.0);
            }
        }
    }

    // 3. Matrix Constraint Setup
    int num_equality = 1 + edges.size();
    int total_constraints = num_equality + num_vars;

    std::vector<scs_float> b(total_constraints, 0.0);
    b[0] = 1.0; // Tr(X) = 1

    std::vector<scs_float> Ax;
    std::vector<scs_int> Ai;
    std::vector<scs_int> Ap;
    Ap.push_back(0);

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            int current_var = get_tri_idx(i, j, n);

            // --- Section A: Equality Constraints ---
            if (i == j) {
                Ax.push_back(1.0);
                Ai.push_back(0);
            }

            for (size_t e = 0; e < edges.size(); ++e) {
                if ((edges[e].first == i && edges[e].second == j) ||
                    (edges[e].first == j && edges[e].second == i)) {
                    Ax.push_back(1.0);
                    Ai.push_back(1 + e);
                }
            }

            // --- Section B: PSD Cone Slacks (-I * x + s = 0) ---
            // FIXED: Do NOT multiply off-diagonals by std::sqrt(2.0) here.
            // Use -1.0 uniformly. SCS handles the structural scaling internally.
            scs_float val = -1.0;
            Ax.push_back(val);
            Ai.push_back(num_equality + current_var);

            Ap.push_back(Ax.size());
        }
    }

    // 4. Initialize Data Frameworks
    ScsData data = {0};
    data.m = total_constraints;
    data.n = num_vars;

    ScsMatrix A = {0};
    A.x = Ax.data();
    A.i = Ai.data();
    A.p = Ap.data();
    A.m = total_constraints;
    A.n = num_vars;
    data.A = &A;

    data.b = b.data();
    data.c = c.data();

    // 5. Structure the Conic Bounds
    ScsCone cone = {0};
    cone.z = num_equality;

    // Fixed array initialization syntax error
    scs_int s_array[1] = { n };
    cone.s = s_array;
    cone.ssize = 1;

    // 6. Execute Solver
    ScsSettings stgs;
    scs_set_default_settings(&stgs);
    stgs.eps_abs = 1e-7;
    stgs.eps_rel = 1e-7;
    stgs.verbose = 0; // decide if you want details printed out

    ScsSolution sol = {0};
    ScsInfo info = {0};

    ScsWork *work = scs_init(&data, &cone, &stgs);
    double out = 0.0;
    if (work) {
        scs_solve(work, &sol, &info, false);

/*        // 7. Verify Results
        std::cout << "\n=========================================" << std::endl;
        std::cout << "Solver Run Status: " << info.status << std::endl;
        std::cout << "Computed Lovasz Theta: " << -info.pobj << std::endl;
        std::cout << "Exact Target Value (sqrt(5)): " << std::sqrt(5.0) << std::endl;
        std::cout << "=========================================" << std::endl;*/

        out = -info.pobj;
        scs_finish(work);
    }
    // if (sol.x) scs_free_sol(&sol);

    // if (sol.x) cleanup_scs(nullptr,nullptr,&sol,nullptr);
    return out;
}

/*
inline double compute_lovasz_theta( int dim, std::vector<std::pair<int,int>>& edges ) {
    // 1. Define a simple Graph (e.g., Cycle graph C_5)
    struct Graph {
        int num_vertices;
        std::vector<std::pair<int, int>> edges;
    };
    Graph g;
    g.num_vertices = dim;
    g.edges = edges;

    // 2. Set up SCS Data Structures
    ScsData* data = (ScsData*)scs_calloc(1, sizeof(ScsData));
    ScsCone* cone = (ScsCone*)scs_calloc(1, sizeof(ScsCone));
    ScsSettings* settings = (ScsSettings*)scs_calloc(1, sizeof(ScsSettings));

    // Set default SCS configuration parameters
    scs_set_default_settings(settings);
    settings->eps_abs = 1e-6;
    settings->eps_rel = 1e-6;

       3. Problem Setup Strategy:
       - The decision vector 'x' contains: [t, variables representing elements in A]
       - Matrix 'A' inside SCS defines the linear mapping: A*x + s = b, where s \in Cone
       - Set cone->p length to map the size of the semidefinite block (n * (n + 1) / 2)

    int num_edges = g.edges.size();
    data->n = 1 + num_edges; // Number of optimization variables (t + active edge variables)

    // Define the SDP cone size
    cone->psize = 1;
    cone->p = (scs_float*)scs_malloc(sizeof(scs_int) * 1);
    cone->p[0] = g.num_vertices; // One Semidefinite cone block of dimension n x n

    // Total rows in the constraint matrix
    data->m = (g.num_vertices * (g.num_vertices + 1)) / 2;

    // TODO: Populate data->A (in CSC format), data->b (vectorizing J), and data->c ([1, 0, 0...])
    // vector c represents minimizing 1*t + 0*A_edges
    data->c = (scs_float*)scs_calloc(data->n, sizeof(scs_float));
    data->c[0] = 1.0;

    // 4. Execute SCS Solver
    ScsSolution* sol = (ScsSolution*)scs_calloc(1, sizeof(ScsSolution));
    ScsInfo* info = (ScsInfo*)scs_calloc(1, sizeof(ScsInfo));

    scs_int status = scs(data, cone, settings, sol, info);

    // 5. Print Results
    // std::cout << "Solver Status: " << info->status << std::endl;
    // std::cout << "Lovasz Theta value: " << sol->x[0] << std::endl;

    double out = sol->x[0];

    // Clean up allocated memory
    cleanup_scs(data,cone,sol,nullptr);
    // scs_free_data(data, cone);
    // scs_free_sol(sol);
    // scs_free_info(info);
    free(settings);

    return out;
}
*/
#endif //FLAGCALC_SDP_H
