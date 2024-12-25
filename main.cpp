#include <algorithm>
#include <climits>
#include <cmath>
#include <forward_list>
#include <format>
#include <iostream>
#include <iomanip>
#include <list>
#include <stack>
#include <vector>
#include <queue>
#include <set>
#include <thread>
#include <complex>
#include <functional>
#include <unordered_map>
#include "sort.h"
#include "graph.h"
#include "tree.h"

using namespace std;

const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");

// return the ith smallest element in v[p, r]
int randomized_select(std::vector<int> &v, int p, int r, int i) {
    if (p == r) {
        return v[p];
    }
    int q = randomized_partition(v, p, r);
    int k = q - p + 1;
    if (i == k) {
        return v[q];
    } else if (i < k) {
        return randomized_select(v, p, q - 1, i);
    } else {
        return randomized_select(v, q + 1, r, i - k);
    }
}

void transpose_matrix(std::vector<std::vector<int>> &A) {
    for (int i = 0; i < A.size(); ++i) {
        for (int j = i + 1; j < A[0].size(); ++j) {
            std::swap(A[i][j], A[j][i]);
        }
    }
}

// A must be a square matrix
void transpose_recursive(std::vector<std::vector<int>> &A, int index = 0){
    if (index == A.size() - 1) {
        return;
    }
    for (int i = index + 1; i < A[0].size(); ++i) {
        std::swap(A[index][i], A[i][index]);
    }
    transpose_recursive(A, index + 1);
}

std::vector<std::vector<int>> matrix_multiply(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B) {
    std::vector<std::vector<int>> C(A.size(), std::vector<int>(B[0].size()));
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < B[0].size(); ++j) {
            C[i][j] = 0;
            for (int k = 0; k < A[0].size(); ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

std::vector<std::vector<int>> matrix_add(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B) {
    std::vector<std::vector<int>> C(A.size(), std::vector<int>(A[0].size()));
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

// A and B are square matrices
std::vector<std::vector<int>> matrix_multiply_recursive(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B) {
    if (A.size() != A[0].size() or B.size() != B[0].size() or A.size() != B.size()) {
        std::cerr << "Error: A and B are not square matrices or A and B are not the same size." << std::endl;
        exit(1);
    }
    std::vector<std::vector<int>> C(A.size(), std::vector<int>(B[0].size()));
    if (A.size() == 1) {
        C[0][0] = A[0][0] * B[0][0];
    } else {
        std::vector<std::vector<int>> A11(A.size() / 2, std::vector<int>(A.size() / 2));
        std::vector<std::vector<int>> A12(A.size() / 2, std::vector<int>(A.size() / 2));
        std::vector<std::vector<int>> A21(A.size() / 2, std::vector<int>(A.size() / 2));
        std::vector<std::vector<int>> A22(A.size() / 2, std::vector<int>(A.size() / 2));
        std::vector<std::vector<int>> B11(B.size() / 2, std::vector<int>(B.size() / 2));
        std::vector<std::vector<int>> B12(B.size() / 2, std::vector<int>(B.size() / 2));
        std::vector<std::vector<int>> B21(B.size() / 2, std::vector<int>(B.size() / 2));
        std::vector<std::vector<int>> B22(B.size() / 2, std::vector<int>(B.size() / 2));
        std::vector<std::vector<int>> C11(C.size() / 2, std::vector<int>(C.size() / 2));
        std::vector<std::vector<int>> C12(C.size() / 2, std::vector<int>(C.size() / 2));
        std::vector<std::vector<int>> C21(C.size() / 2, std::vector<int>(C.size() / 2));
        std::vector<std::vector<int>> C22(C.size() / 2, std::vector<int>(C.size() / 2));
        for (int i = 0; i < A.size() / 2; ++i) {
            for (int j = 0; j < A[0].size() / 2; ++j) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + A.size() / 2];
                A21[i][j] = A[i + A.size() / 2][j];
                A22[i][j] = A[i + A.size() / 2][j + A.size() / 2];
            }
        }
        for (int i = 0; i < B.size() / 2; ++i) {
            for (int j = 0; j < B[0].size() / 2; ++j) {
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j + B.size() / 2];
                B21[i][j] = B[i + B.size() / 2][j];
                B22[i][j] = B[i + B.size() / 2][j + B.size() / 2];
            }
        }
        C11 = matrix_add(matrix_multiply_recursive(A11, B11), matrix_multiply_recursive(A12, B21));
        C12 = matrix_add(matrix_multiply_recursive(A11, B12), matrix_multiply_recursive(A12, B22));
        C21 = matrix_add(matrix_multiply_recursive(A21, B11), matrix_multiply_recursive(A22, B21));
        C22 = matrix_add(matrix_multiply_recursive(A21, B12), matrix_multiply_recursive(A22, B22));
        for (int i = 0; i < C.size() / 2; ++i) {
            for (int j = 0; j < C[0].size() / 2; ++j) {
                C[i][j] = C11[i][j];
                C[i][j + C[0].size() / 2] = C12[i][j];
                C[i + C.size() / 2][j] = C21[i][j];
                C[i + C.size() / 2][j + C[0].size() / 2] = C22[i][j];
            }
        }
    }
    return C;
}

void print_matrix(const std::vector<std::vector<int>> &A) {
    for (auto &i : A) {
        for (auto &j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}




class QueueByStacks
{
public:
    void enqueue(int x) {
        s1.push(x);
    }

    int dequeue() {
        if (s2.empty()) {
            if (s1.empty()) {
                std::cerr << "Error: queue underflow." << std::endl;
                exit(1);
            }
            while (!s1.empty()) {
                s2.push(s1.top());
                s1.pop();
            }
        }
        int x = s2.top();
        s2.pop();
        return x;
    }


private:
    std::stack<int> s1;
    std::stack<int> s2;
};

int edit_distance(std::string &x, std::string &y) {
    int m = x.size();
    int n = y.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (x[i - 1] == y[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1});
            }
        }
    }
    return dp[m][n];

}


const std::vector<int> price = {0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30};

// recursive top-down implementation, no dynamic programming
int cut_rod(const std::vector<int> &p, int n) {
    if (n == 0) {
        return 0;
    }
    int q = INT_MIN;
    for (int i = 1; i <= n; ++i) {
        q = std::max(q, p[i] + cut_rod(p, n - i));
    }
    return q;
}

int memoried_cut_rod_aux(const std::vector<int> &p, int n, std::vector<int> &r) {
    if (r[n] >= 0) {
        return r[n];
    }
    int q;
    if (n == 0) {
        q = 0;
    } else {
        q = INT_MIN;
        for (int i = 1; i <= n; ++i) {
            q = std::max(q, p[i] + memoried_cut_rod_aux(p, n - i, r));
        }
    }
    r[n] = q;
    return q;
}

int memoried_cut_rod(const std::vector<int> &p, int n) {
    std::vector<int> r(n + 1, INT_MIN);
    return memoried_cut_rod_aux(p, n, r);
}

int buttom_up_cut_rod(const std::vector<int> &p, int n) {
    std::vector<int> r(n + 1);
    r[0] = 0;
    for (int j = 1; j <= n; ++j) {
        int q = INT_MIN;
        for (int i = 1; i <= j; ++i) {
            q = std::max(q, p[i] + r[j - i]);
        }
        r[j] = q;
    }
    return r[n];
}

std::pair<std::vector<int>, std::vector<int>> extended_buttom_up_cut_rod(const std::vector<int> &p, int n) {
    std::vector<int> r(n + 1);  // the optimal revenue of a rod of length i
    std::vector<int> s(n + 1);  // the optimal size of the first piece to cut off
    r[0] = 0;
    for (int j = 1; j <= n; ++j) {
        int q = INT_MIN;
        for (int i = 1; i <= j; ++i) {
            if (q < p[i] + r[j - i]) {
                q = p[i] + r[j - i];
                s[j] = i;
            }
        }
        r[j] = q;
    }
    return std::make_pair(r, s);
}

void print_cut_rod_solution(const std::vector<int> &p, int n) {
    std::pair<std::vector<int>, std::vector<int>> p_s = extended_buttom_up_cut_rod(p, n);
    std::cout << "cut locations: ";
    while (n > 0) {
        std::cout << p_s.second[n] << " ";      // cut location for length n
        n -= p_s.second[n];                     // length of the reminder of the rod
    }
    std::cout << std::endl;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<char>> >
lcs_length(const std::string &x, const std::string &y) {
    int m = x.size();
    int n = y.size();
    std::vector<std::vector<int>> c(m + 1, std::vector<int>(n + 1));    // c[i][j] is the length of an LCS of x_i and y_j
    std::vector<std::vector<char>> b(m + 1, std::vector<char>(n + 1));  // b[i][j] is the direction to c[i][j]
    for (int i = 1; i <= m; ++i) {
        c[i][0] = 0;
    }
    for (int j = 0; j <= n; ++j) {
        c[0][j] = 0;
    }
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // here we use 1-based index, but the string index is 0-based, so we need to subtract 1
            if (x[i - 1] == y[j - 1]) {
                c[i][j] = c[i - 1][j - 1] + 1;
                b[i][j] = 'd';  // diagonal
            } else if (c[i - 1][j] >= c[i][j - 1]) {
                c[i][j] = c[i - 1][j];
                b[i][j] = 'u';  // up
            } else {
                c[i][j] = c[i][j - 1];
                b[i][j] = 'l';  // left
            }
        }
    }
    return std::make_pair(c, b);
}

void print_b(const std::vector<std::vector<char>> &b, const std::string &x, const std::string &y) {

    std::cout << " ";
    for (int i = 0; i < y.size(); ++i) {
        std::cout << y[i];
    }
    std::cout << std::endl;

    for (int i = 1; i < b.size(); ++i) {
        std::cout << x[i - 1];
        for (int j = 1; j < b[0].size(); ++j) {
            char c = b[i][j];
            if (c == 'd') {
                std::cout << "↖";
            } else if (c == 'u') {
                std::cout << "↑";
            } else {
                std::cout << "←";
            }
        }
        std::cout << std::endl;
    }
}

void print_lcs(const std::vector<std::vector<char>> &b, const std::string &x, int i, int j) {
    if (i == 0 or j == 0) {
        return;
    }
    if (b[i][j] == 'd') {
        print_lcs(b, x, i - 1, j - 1);
        std::cout << x[i - 1];          // only print when it's a diagonal
    } else if (b[i][j] == 'u') {
        print_lcs(b, x, i - 1, j);
    } else {
        print_lcs(b, x, i, j - 1);
    }
}


typedef std::vector<std::pair<int, int>> ParentsIdx;

// return all possible solution in (0, n - 1) in dp, and path to the solution in dir
auto rna_secondary_structure(const std::string &s) {
    int n = s.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n));
    std::vector<std::vector<ParentsIdx>> dir(n, std::vector<ParentsIdx>(n));

    // max distance is n - 1, so k < n
    for (int k = 5; k < n; ++k) {  // k = j - i, k is the distance between i to j
        // max index for i is n - 1 - k, so i < n - k
        for (int i = 0; i < n - k; ++i) {
            int j = i + k;              // j is the right index
            dp[i][j] = dp[i][j - 1];    // if j is not involved in a pair
            dir[i][j].push_back({i, j - 1});        // my parent is to the left
            for (int t = i; t < j - 4; ++t) {  // if j pairs with t for some t < j - 4
                char s_t = s[t];
                char s_j = s[j];
                if ((s_t == 'A' and s_j == 'U') or (s_t == 'U' and s_j == 'A') or (s_t == 'C' and s_j == 'G') or (s_t == 'G' and s_j == 'C')) {
                    int val = dp[i][t - 1] + dp[t + 1][j - 1] + 1;
                    if (val > dp[i][j]) {      // if the number of pairs is better than if j is not involved
                        dp[i][j] = val;
                        dir[i][j].clear();

                        dir[i][j].push_back({i, t - 1});            // t - 1 might be -1, which means t = 0, left half is empty
                        dir[i][j].push_back({t + 1, j - 1});
                    }
                }
            }
        }
    }

    return std::make_pair(dp, dir);
}

// print parent structure and return all pairs
std::vector<std::pair<int, int>> print_rna_secondary_structure(const std::vector<std::vector<int>> &dp,
                                                               const std::vector<std::vector<ParentsIdx>> &dir,
                                                               int i, int j) {

    std::deque<std::pair<int, int>> q;                  // use deque in a iterative BFS way
    std::vector<std::pair<int, int>> pairs;
    q.push_back({i, j});

    while (!q.empty()) {
        auto sz = q.size();
        for (int i = 0; i < sz; ++i) {
            auto p = q.front();
            q.pop_front();
            // show index and the number of pairs
            std::cout << "(" << p.first << ", " << p.second << "):" << dp[p.first][p.second] << " ";

            if (dir[p.first][p.second].size() == 1) {                                      // if only 1 parent, then j is not paired with any other                                                 
                std::cout << "j: " << p.second << " is not involved in a pair";
            } else if (dir[p.first][p.second].size() == 2) {                               // if 2 parents, then j is paired with t
                int t = dir[p.first][p.second][0].second + 1;
                std::cout << std::format("({}, {})", t, p.second) << " is a pair";
                pairs.push_back({t, p.second});
            }

            for (auto &pp : dir[p.first][p.second]) {
                if (pp.second >= 0) {       // this value might be -1
                    q.push_back(pp);
                }
            }
        }
        std::cout << std::endl;
    }

    // print pairs in a readable way
    std::sort(pairs.begin(), pairs.end());
    std::cout << "All pairs (size:" << pairs.size() << "):" << std::endl;
    for (int i = 0; i < pairs.size(); ++i) {
        std::cout << std::setw(3) << pairs[i].first;
    }
    std::cout << std::endl;
    for (int i = 0; i < pairs.size(); ++i) {
        std::cout << std::setw(3) << pairs[i].second;
    }
    std::cout << std::endl;

    return pairs;
}

// print maxium pairs in all possible (0, n - 1)
// red color indicate corresponding index (i, j) is a pair, but don't confuse with the number = dp[i, j], which is the number of maximum pairs
void print_rna_all_solutions(const std::string &s, const std::vector<std::vector<int>> &dp, const std::vector<std::pair<int, int>> &pairs) {
    int n = dp.size();

    // print the index
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(3) << i;
    }
    std::cout << std::endl;

    // print the string
    for (auto &c : s) {
        std::cout << std::setw(3) << c;
    }
    std::cout << std::endl;
    std::cout << "--------------------------" << std::endl;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::find(pairs.begin(), pairs.end(), std::make_pair(i, j)) != pairs.end()) {
                std::cout << red << std::setw(3) << dp[i][j] << reset;
            } else {
                std::cout << std::setw(3) << dp[i][j];
            }
        }
        std::cout << std::endl;
    }
}

int p_fib(int n) {
    if (n <= 1) {
        return n;
    } else {
        int result1, result2;
        std::thread t1([&result1, n] { result1 = p_fib(n - 1); });
        std::thread t2([&result2, n] { result2 = p_fib(n - 2); });
        t1.join();
        t2.join();
        return result1 + result2;
    }
}

// DFT by FFT
// input is a vector of coefficients of a polynomial
// output is evaluation of n complex nth roots of unity
std::vector<std::complex<double>> fft(const std::vector<double> &a) {
    const int n = a.size();
    if (n == 1) {
        return {a[0]};
    }
    auto w = [](int n, int k) {
        return std::polar(1.0, 2 * M_PI * k / n);
    };
    // w_n is principal nth root of unity
    auto w_n = w(n, 1);
    auto w_1 = w(n, 0);

    std::vector<double> a_even(n / 2);
    std::vector<double> a_odd(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }
    std::vector<std::complex<double>> y_even = fft(a_even);
    std::vector<std::complex<double>> y_odd = fft(a_odd);
    std::vector<std::complex<double>> y(n);
    for (int k = 0; k < n / 2; ++k) {
        y[k] = y_even[k] + w_1 * y_odd[k];
        y[k + n / 2] = y_even[k] - w_1 * y_odd[k];
        w_1 *= w_n;
    }
    return y;
}

// An array 𝐴 is wiggly if 𝐴[1] ≤ 𝐴[2] ≥ 𝐴[3] ≤ 𝐴[4] ≥ ⋯ ≤ 𝐴[2𝑛 − 2] ≥ 𝐴[2𝑛 − 1] ≤ 𝐴[2𝑛]

void wiggly_permutation(std::vector<int> &v) {

    // O(nlogn) time complexity
    // std::sort(v.begin(), v.end());
    // for (int i = 1; i < v.size() - 1; i += 2) {
    //     std::swap(v[i], v[i + 1]);
    // }

    // O(n) time complexity
    for (int i = 1; i < v.size(); ++i) {
        if ((i % 2 == 1 and v[i] < v[i - 1]) or (i % 2 == 0 and v[i] > v[i - 1])) {
            std::swap(v[i], v[i - 1]);
        }
    }
}

void verify_wiggly_permutation(const std::vector<int> &v) {
    for (int i = 1; i < v.size() - 1; i += 2) {
        if (v[i] < v[i - 1] or v[i] < v[i + 1]) {
            std::cerr << "Error: not a wiggly permutation." << std::endl;
            exit(1);
        }
    }
}

// Function to generate a vector of random numbers of given length
std::vector<int> generate_random_vector(int length) {
    std::vector<int> v(length);
    for (int i = 0; i < length; ++i) {
        v[i] = rand();
    }
    return v;
}

int main() {
    const int num_tests = 1000;

    // Vector of sorting function pointers
    std::vector<std::function<void(std::vector<int>&)>> sort_functions = {
        bubble_sort,
        bubble_sort2,
        insertion_sort,
        [](std::vector<int>& v) { merge_sort(v, 0, v.size() - 1); },
        selection_sort,
        selection_sort2,
        selection_sort3,
        selection_sort4
    };

    for (int i = 0; i < num_tests; ++i) {
        // Vector to store sorted results
        std::vector<std::vector<int>> sorted_results;

        int length = rand() % 1000;
        length = i;


        std::vector<int> v = generate_random_vector(length);

        // Apply each sort function and store the result
        for (auto& sort_func : sort_functions) {
            std::vector<int> v_copy = v;
            sort_func(v_copy);
            sorted_results.push_back(v_copy);
        }

        // Compare all sorted results
        for (int j = 1; j < sorted_results.size(); ++j) {
            if (sorted_results[0] != sorted_results[j]) {
                std::cerr << "Error: Sorting results are not the same." << std::endl;
                return 1;
            }
        }
        std::cout << "test " << i << " successful!" << std::endl;
    }
    std::cout << "All sorting results are the same." << std::endl;
    
    // Heap<int> h(v, v.size());

    // build_max_heap(h, h.size());
    // heap_sort(h, h.size());
    // randomized_quick_sort(v, 0, v.size() - 1);

    // std::vector<int> v2 = counting_sort(v, 6);
    // for (auto i : v2) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // std::vector<int> v2 = {329, 457, 657, 839, 436, 720, 355};
    // radix_sort(v2, 3);
    // for (auto i : v2) {
    //     std::cout << i << " ";
    // }

    // std::vector<double> v2 = {0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68};
    // std::list<double> l = bucket_sort(v2);
    // for (auto i : l) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;


    // std::pair<double, double> p = maximum_and_minimum<double>(v2);
    // std::cout << p.first << " " << p.second << std::endl;

    // std::vector<std::vector<int>> C = {{1, 2},
    //                                     {3, 4}};
    // std::vector<std::vector<int>> D = {{5, 6},
    //                                     {7, 8}};


    // std::vector<std::vector<int>> A = {{1, 2, 3},
    //                                    {4, 5, 6}};
    // std::vector<std::vector<int>> B = {{7, 8},
    //                                    {9, 10},
    //                                    {11, 12}};

    // std::vector<std::vector<int>> G = {{1, 2, 3, 4},
    //                                    {5, 6, 7, 8}};
    // std::vector<std::vector<int>> H = {{1, 2},
    //                                    {3, 4},
    //                                    {5, 6},
    //                                    {7, 8}};

    // transpose_matrix(A);
    // print_matrix(A);

    // std::vector<std::vector<int>> F = matrix_multiply_recursive(C, D);
    // print_matrix(F);

    // std::vector<std::vector<int>> E = matrix_multiply_recursive(A, B);
    // print_matrix(E);

    // std::vector<std::vector<int>> I = matrix_multiply_recursive(G, H);
    // print_matrix(I);

    // Node<int> *root = new Node(12);
    // Node<int> *n1 = new Node(5);
    // Node<int> *n2 = new Node(18);
    // Node<int> *n3 = new Node(2);
    // Node<int> *n4 = new Node(9);
    // Node<int> *n5 = new Node(15);
    // Node<int> *n6 = new Node(19);
    // Node<int> *n7 = new Node(17);
    
    // BinarySearchTree<int> bst;
    // bst.tree_insert(n1);
    // bst.tree_insert(n2);
    // bst.tree_insert(n3);
    // bst.tree_insert(n4);
    // bst.tree_insert(n5);
    // bst.tree_insert(n6);
    // bst.tree_insert(n7);
    // bst.print_tree();

    // std::cout << "-------------------------- " << std::endl;

    // Node<int> *x = new Node(13);
    // bst.tree_insert(x);
    // bst.print_tree();
    // std::cout << "-------------------------- " << std::endl;
    // bst.tree_delete(n1);
    // bst.print_tree();
    // std::cout << "-------------------------- " << std::endl;
    // bst.tree_delete(n2);
    // bst.print_tree();

    // QueueByStacks q;
    // q.enqueue(1);
    // q.enqueue(2);
    // q.enqueue(3);
    // std::cout << q.dequeue() << std::endl;
    // q.enqueue(4);
    // q.enqueue(5);
    // std::cout << q.dequeue() << std::endl;
    // std::cout << q.dequeue() << std::endl;
    // std::cout << q.dequeue() << std::endl;
    // std::cout << q.dequeue() << std::endl;
    // std::cout << q.dequeue() << std::endl;

    // std::vector<std::string> cities{"Dallas", "Orlando", "Philadelphia", "Miami", "Chicago", "Denver", "Boston", "San Francisco"};

    // RedBlackTree<std::string> rb_tree;

    // for (const std::string city : cities) {
    //     Node<std::string> *x = new Node<std::string>(city);
    //     rb_tree.rb_insert(x);
    //     // rb_tree.tree_insert(x);
    //     rb_tree.print_tree();
    //     std::cout << "-------------------------- " << std::endl;
    // }

    // std::string x = "kitten";
    // std::string y = "sitting";
    // std::cout << edit_distance(x, y) << std::endl;

    // for (int i = 0; i <= 10; ++i) {
    //     std::cout << cut_rod(price, i) << std::endl;
    //     std::cout << memoried_cut_rod(price, i) << std::endl;
    //     std::cout << buttom_up_cut_rod(price, i) << std::endl;
    //     print_cut_rod_solution(price, i);
    //     std::cout << "-------------------------- " << std::endl;
    // }

    // std::string x = "ABCBDAB";
    // std::string y = "BDCABA";
    // auto p = lcs_length(x, y);
    // print_b(p.second, x, y);
    // print_lcs(p.second, x, x.size(), y.size());

    // std::string rna_sequence = "AUGGCUACCGGUCGAUUGAGCGCCAAUGUAAUCAUU";
    
    // auto dp_dir = rna_secondary_structure(rna_sequence);

    // auto pairs = print_rna_secondary_structure(dp_dir.first, dp_dir.second, 0, rna_sequence.size() - 1);

    // print_rna_all_solutions(rna_sequence, dp_dir.first, pairs);

    // Graph g(8);
    // g.add_edge({{0, 1}, {0, 3}, {1, 2}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}});
    // g.print_graph();

    // g.bfs(0);
    // g.print_path(0, 7);

    // g.dfs_iterative(0);
    // g.print_path(0, 7);

    // g.dfs();


    // std::cout << p_fib(15) << std::endl;

    // std::vector<double> a = {0, 1, 2, 3};
    // std::vector<std::complex<double>> b = fft(a);
    // for (auto i : b) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // std::vector<int> v = {3, 2, 1, 6, 5, 4};
    // wiggly_permutation(v);
    // verify_wiggly_permutation(v);

    return 0;
}