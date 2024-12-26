#include "graph.h"

// V is the size of vertices
// vertices are [0, V - 1]
Graph::Graph(int V) : 
    V(V), adj(V), color(V, WHITE), d(V, INT_MAX), f(V), pi(V, -1), weight(V, std::vector<int>(V, 1)) {}

void Graph::add_edge(int u, int v) {
    adj[u].push_back(v);
}

void Graph::add_edge(const std::vector<std::pair<int, int>> &edges) {
    for (auto &e : edges) {
        add_edge(e.first, e.second);
    }
}

void Graph::print_graph() {
    for (int i = 0; i < V; ++i) {
        std::cout << i << ": ";
        for (auto &j : adj[i]) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

// use 3 colors to indicate visited
void Graph::bfs(int s) {
    for (int i = 0; i < V; ++i) {
        color[i] = WHITE;
        d[i] = INT_MAX;
        pi[i] = -1;
    }

    color[s] = GRAY;
    d[s] = 0;
    pi[s] = -1;

    std::queue<int> q;
    
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto &v : adj[u]) {
            if (color[v] == WHITE) {
                color[v] = GRAY;
                d[v] = d[u] + 1;
                pi[v] = u;
                q.push(v);
            }
        }
        color[u] = BLACK;
    }
}

// use bool to indicate visited
void Graph::simple_bfs(int s) {
    visited = std::vector<bool>(V, false);

    std::queue<int> qu;
    qu.push(s);
    visited[s] = true;
    while (!qu.empty()) {
        int u = qu.front();
        qu.pop();
        for (auto v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                qu.push(v);
            }
        }
        // processed all of u's neighbors
    }
}

void Graph::dfs_iterative(int s) {
    for (int i = 0; i < V; ++i) {
        color[i] = WHITE;
        d[i] = INT_MAX;
        pi[i] = -1;
    }

    color[s] = GRAY;
    d[s] = 0;
    pi[s] = -1;

    // the only thing changed is from queue to stack
    std::stack<int> q;
    
    q.push(s);
    while (!q.empty()) {
        int u = q.top();
        q.pop();
        for (auto &v : adj[u]) {
            if (color[v] == WHITE) {
                color[v] = GRAY;
                d[v] = d[u] + 1;
                pi[v] = u;
                q.push(v);
            }
        }
        color[u] = BLACK;
    }
}

void Graph::dfs_visit(int u, int &time) {
    color[u] = GRAY;
    d[u] = ++time;
    for (auto &v : adj[u]) {
        if (color[v] == WHITE) {
            pi[v] = u;
            dfs_visit(v, time);
        }
    }
    color[u] = BLACK;
    f[u] = ++time;
}

// produces a depth-ﬁrst forest
void Graph::dfs() {
    for (int i = 0; i < V; ++i) {
        color[i] = WHITE;
        pi[i] = -1;
    }
    int time = 0;
    for (int i = 0; i < V; ++i) {
        if (color[i] == WHITE) {
            dfs_visit(i, time);
        }
    }
}

// print a path from s to v
void Graph::print_path_recursive(int s, int v) {
    if (v == s) {
        std::cout << s << " ";
    } else if (pi[v] == -1) {
        std::cout << "no path from " << s << " to " << v << " exists" << std::endl;
    } else {
        print_path_recursive(s, pi[v]);
        std::cout << v << " ";
    }
}

void Graph::print_path(int s, int v) {
    print_path_recursive(s, v);
    std::cout << std::endl;
}

void Graph::print_all_pairs_shortest_path(int i, int j) {
    if (i == j) {
        std::cout << i << " ";
    } else if (PI[i][j] == -1) {
        std::cout << "no path from " << i << " to " << j << " exists" << std::endl;
    } else {
        print_all_pairs_shortest_path(i, PI[i][j]);
        std::cout << j << " ";
    }
}

void Graph::initialize_single_source(int s) {
    for (int i = 0; i < V; ++i) {
        d[i] = INT_MAX;
        pi[i] = -1;
    }
    d[s] = 0;
}

void Graph::relax(int u, int v) {
    if (d[v] > d[u] + weight[u][v]) {
        d[v] = d[u] + weight[u][v];
        pi[v] = u;
    }
}

void Graph::dijkstra(int s) {
    initialize_single_source(s);
    std::set<std::pair<int, int>> S;
    // priority queue
}

bool Graph::bellman_ford(int s) {
    initialize_single_source(s);
    for (int i = 0; i < V - 1; ++i) {
        for (int u = 0; u < V; ++u) {
            for (auto &v : adj[u]) {
                relax(u, v);
            }
        }
    }
    for (int u = 0; u < V; ++u) {
        for (auto &v : adj[u]) {
            if (d[v] > d[u] + weight[u][v]) {
                return false;
            }
        }
    }
    return true;
}
