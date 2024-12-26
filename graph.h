#pragma once
#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <stack>
#include <set>
#include <unordered_map>
#include <climits>

enum Color {WHITE, GRAY, BLACK, RED};

class Graph {
public:
    // V is the size of vertices
    // vertices are [0, V - 1]
    Graph(int V);

    void add_edge(int u, int v);

    void add_edge(const std::vector<std::pair<int, int>> &edges);

    void print_graph();

    // use 3 colors to indicate visited
    void bfs(int s);

    // use bool to indicate visited
    void simple_bfs(int s);

    void dfs_iterative(int s);

    void dfs();

    // print a path from s to v
    void print_path(int s, int v);
    void print_all_pairs_shortest_path(int i, int j);

    void initialize_single_source(int s);
    void relax(int u, int v);

    void dijkstra(int s);

    bool bellman_ford(int s);

protected:
    // distance from the source (used in BFS)
    // or time discovered (used in DFS)
    std::vector<int> d;

private:
    void dfs_visit(int u, int &time);
    void print_path_recursive(int s, int v);

private:
    // size of the vertices
    int V;

    // (1/2) Adjacency list 
    std::vector<std::vector<int>> adj;
    // (2/2) if node is string, and edges have a value
    std::unordered_map<std::string, std::unordered_map<std::string, double>> adj2;
    // adj2["node1"]["node2"] = 3.14;

    // (1/2) color of each vertices
    std::vector<Color> color;
    // (2/2) can also use bool to indicate visited;
    std::vector<bool> visited;

    // parent of each vertice
    std::vector<int> pi;

    // time finished discovery (only in DFS)
    std::vector<int> f;

    std::vector<std::vector<int>> weight;

    // for all-pairs shortest paths
    // adjacency matrix
    std::vector<std::vector<int>> adj_matrix;
    // results
    std::vector<std::vector<int>> D;
    // predecessor matrix
    std::vector<std::vector<int>> PI;
};