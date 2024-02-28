#include <iostream>
#include <vector>
#include <list>
#include <stack>
#include <forward_list>
#include <climits>
#include <cmath>

void insertion_sort(std::vector<int> &v) {
    for (int i = 1; i < v.size(); ++i) {
        int key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }
}

// find max element in v and exchange it with last element in v
// effectively, sort in increasing order
void selection_sort(std::vector<int> &v) {
    for (int right = v.size() - 1; right >= 1; --right) {
        int max_index = 0;
        for (int i = 1; i <= right; ++i) {
            if (v[i] > v[max_index]) {
                max_index = i;
            }
        }
        std::swap(v[right], v[max_index]);
    }
}

void merge(std::vector<int> &v, int p, int q, int r) {
    int n1 = q - p + 1;  // length of [p, q]
    int n2 = r - q;     // length of [q + 1, r]

    std::vector<int> L(n1 + 1);
    std::vector<int> R(n2 + 1);

    for (int i = 0; i < n1; ++i) {
        L[i] = v[p + i];
    }
    for (int j = 0; j < n2; ++j) {
        R[j] = v[q + j + 1];
    }

    // by using sentinel, we can avoid checking whether L or R is empty
    L[n1] = INT_MAX;
    R[n2] = INT_MAX;

    int i = 0; // index of L
    int j = 0; // index of R

    for (int k = p; k <= r; ++k) {
        if (L[i] <= R[j]) {
            v[k] = L[i];
            ++i;
        } else {
            v[k] = R[j];
            ++j;
        }
    }
}

void merge_sort(std::vector<int> &v, int p, int r) {
    if (p < r) {
        int q = (p + r) / 2; // q is the middle point of [p, r]
        merge_sort(v, p, q);
        merge_sort(v, q + 1, r);
        merge(v, p, q, r);
    }
}

void bubble_sort(std::vector<int> &v) {
    for (int i = 0; i < v.size() - 1; ++i) {
        for (int j = v.size() - 1; j > i; --j) {
            if (v[j] < v[j - 1]) {
                std::swap(v[j], v[j - 1]);
            }
        }
    }
}

template <typename T>
struct Heap : public std::vector<T>
{
    Heap(std::vector<T> &v, int heap_size) : std::vector<T>(v), heap_size(heap_size) {}
    int heap_size;
};


// assume left(i) and right(i) are max heaps, but A[i] might be smaller than its children
void max_heapify(Heap<int> &v, int i) {
    int l = 2 * i + 1; // left child
    int r = 2 * i + 2; // right child
    int largest = i;
    if (l < v.heap_size && v[l] > v[i]) {
        largest = l;
    }
    if (r < v.heap_size && v[r] > v[largest]) {
        largest = r;
    }
    if (largest != i) {
        std::swap(v[i], v[largest]);
        max_heapify(v, largest);
    }
}

// build a max heap from V[0, n)
void build_max_heap(Heap<int> &v, int n) {
    v.heap_size = n;
    for (int i = n / 2 - 1; i >= 0; --i) {
        max_heapify(v, i);
    }
}

// sort V[0, n) in ascending order
void heap_sort(Heap<int> &v, int n) {
    build_max_heap(v, n);
    for (int i = n - 1; i >= 1; --i) {
        std::swap(v[0], v[i]);
        v.heap_size -= 1;
        max_heapify(v, 0);
    }
}

int max_heap_maximum(const Heap<int> &v) {
    if (v.heap_size < 1) {
        std::cerr << "Error: heap underflow." << std::endl;
        exit(1);
    }
    return v[0];
}

int max_heap_extract_max(Heap<int> &v) {
    int max = max_heap_maximum(v);
    v[0] = v[v.heap_size - 1];
    v.heap_size -= 1;
    max_heapify(v, 0);
    return max;
}

// increase v[i] to key
void max_heap_increase_key(Heap<int> &v, int i, int key) {
    if (key < v[i]) {
        std::cerr << "Error: new key is smaller than current key." << std::endl;
        exit(1);
    }
    v[i] = key;
    while (i > 0 && v[(i - 1) / 2] < v[i]) {
        std::swap(v[i], v[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

void max_heap_insert(Heap<int> &v, int key) {
    v.heap_size += 1;
    v[v.heap_size - 1] = INT_MIN;
    max_heap_increase_key(v, v.heap_size - 1, key);
}

int partition(std::vector<int> &v, int p, int r) {
    int x = v[r];
    int i = p - 1;
    for (int j = p; j < r; ++j) {
        if (v[j] <= x) {
            ++i;
            std::swap(v[i], v[j]);
        }
    }
    std::swap(v[i + 1], v[r]);
    return i + 1;
}

void quick_sort(std::vector<int> &v, int p, int r) {
    if (p < r) {
        int q = partition(v, p, r);
        quick_sort(v, p, q - 1);
        quick_sort(v, q + 1, r);
    }
}

int randomized_partition(std::vector<int> &v, int p, int r) {
    int i = rand() % (r - p + 1) + p;
    std::swap(v[r], v[i]);
    return partition(v, p, r);
}

void randomized_quick_sort(std::vector<int> &v, int p, int r) {
    if (p < r) {
        int q = randomized_partition(v, p, r);
        randomized_quick_sort(v, p, q - 1);
        randomized_quick_sort(v, q + 1, r);
    }
}

// each of the n elements in A is an integer in the range [0, k]
void counting_sort(std::vector<int> &A, int k) {
    std::vector<int> B(A.size());
    std::vector<int> C(k + 1);
    for (int i = 0; i < A.size(); ++i) {
        C[A[i]] += 1;
    }
    for (int i = 1; i < C.size(); ++i) {
        C[i] += C[i - 1];
    }
    for (int i = A.size() - 1; i >= 0; --i) {
        B[C[A[i]] - 1] = A[i];
        C[A[i]] -= 1;
    }
    A = B;
}

// use ith digit of each element in A to sort A
void counting_sort_on_i(std::vector<int> &A, int k, int i) {
    std::vector<int> B(A.size());
    std::vector<int> C(k + 1);
    for (int j = 0; j < A.size(); ++j) {
        C[(A[j] / (int)pow(10, i)) % 10] += 1;
    }
    for (int j = 1; j < C.size(); ++j) {
        C[j] += C[j - 1];
    }
    for (int j = A.size() - 1; j >= 0; --j) {
        B[C[(A[j] / (int)pow(10, i)) % 10] - 1] = A[j];
        C[(A[j] / (int)pow(10, i)) % 10] -= 1;
    }
    A = B;
}


// assume each element in A has d digits
void radix_sort(std::vector<int> &v, int d) {
    for (int i = 0; i < d; ++i) {
        // use a stable sort to sort A on digit i
        counting_sort_on_i(v, 9, i);
    }
}

// assume each element in v is in [0, 1)
std::list<double> bucket_sort(std::vector<double> &v) {
    // B is a bucket of linked lists
    std::vector<std::list<double>> B(v.size());
    for (int i = 0; i < v.size(); ++i) {
        B[(int)(v.size() * v[i])].push_back(v[i]);
    }
    for (int i = 0; i < v.size(); ++i) {
        B[i].sort();
    }
    std::list<double> C;
    for (int i = 0; i < v.size(); ++i) {
        C.splice(C.end(), B[i]);
    }
    return C;
}

int minimum(const std::vector<int> &v) {
    if (v.size() == 0) {
        std::cerr << "Error: empty vector." << std::endl;
        exit(1);
    }
    int min = v[0];
    for (int i = 1; i < v.size(); ++i) {
        if (v[i] < min) {
            min = v[i];
        }
    }
    return min;
}

template <typename T>
std::pair<T, T> maximum_and_minimum(const std::vector<T> &v) {
    if (v.size() == 0) {
        std::cerr << "Error: empty vector." << std::endl;
        exit(1);
    }
    T max;
    T min;
    // if n is even
    if (v.size() % 2 == 0) {
        if (v[0] > v[1]) {
            max = v[0];
            min = v[1];
        } else {
            max = v[1];
            min = v[0];
        }
        // process next pair
        for (int i = 2; i < v.size(); i += 2) {
            if (v[i] > v[i + 1]) {
                if (v[i] > max) {
                    max = v[i];
                }
                if (v[i + 1] < min) {
                    min = v[i + 1];
                }
            } else {
                if (v[i + 1] > max) {
                    max = v[i + 1];
                }
                if (v[i] < min) {
                    min = v[i];
                }
            }
        }
    } else {
        max = v[0];
        min = v[0];
        // process next pair
        for (int i = 1; i < v.size(); i += 2) {
            if (v[i] > v[i + 1]) {
                if (v[i] > max) {
                    max = v[i];
                }
                if (v[i + 1] < min) {
                    min = v[i + 1];
                }
            } else {
                if (v[i + 1] > max) {
                    max = v[i + 1];
                }
                if (v[i] < min) {
                    min = v[i];
                }
            }
        }
    }

    return std::make_pair(max, min);
}

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

struct Node
{
    Node(int key) : key(key) {}
    int key;
    Node *parent = nullptr;
    Node *left = nullptr;
    Node *right = nullptr;
};

void inorder_tree_walk(Node *x) {
    if (x != nullptr) {
        inorder_tree_walk(x->left);
        std::cout << x->key << " ";
        inorder_tree_walk(x->right);
    }
}

Node *tree_search(Node *x, int key) {
    if (x == nullptr or x->key == key) {
        return x;
    }
    if (key < x->key) {
        return tree_search(x->left, key);
    } else {
        return tree_search(x->right, key);
    }
}

Node *iterative_tree_search(Node *x, int key) {
    while (x != nullptr and x->key != key) {
        if (key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }
    return x;
}

Node *tree_minimum(Node *x) {
    while (x->left != nullptr) {
        x = x->left;
    }
    return x;
}

Node *tree_maximum(Node *x) {
    while (x->right != nullptr) {
        x = x->right;
    }
    return x;
}

Node *tree_successor(Node *x) {
    if (x->right != nullptr) {
        return tree_minimum(x->right);
    }
    Node *y = x->parent;
    while (y != nullptr and x == y->right) {
        x = y;
        y = y->parent;
    }
    return y;
}

Node *tree_predecessor(Node *x) {
    if (x->left != nullptr) {
        return tree_maximum(x->left);
    }
    Node *y = x->parent;
    while (y != nullptr and x == y->left) {
        x = y;
        y = y->parent;
    }
    return y;
}

void tree_insert(Node *root, Node *z) {
    Node *x = root;
    Node *y = nullptr;
    while (x != nullptr) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }
    z->parent = y;
    if (y == nullptr) {
        root = z;
    } else if (z->key < y->key) {
        y->left = z;
    } else {
        y->right = z;
    }
}

void print_tree(Node *root, int level = 0) {
    if (root != nullptr) {
        print_tree(root->right, level + 1);
        for (int i = 0; i < level; ++i) {
            std::cout << "    ";
        }
        std::cout << root->key << std::endl;
        print_tree(root->left, level + 1);
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


int main() {
    // std::vector<int> v = {5, 2, 4, 6, 1, 3};
    // insertion_sort(v);
    // merge_sort(v, 0, v.size() - 1);
    // bubble_sort(v);
    // for (auto i : v) {
    //     std::cout << i << " ";
    // }
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

    // Node *root = new Node(12);
    // Node *n1 = new Node(5);
    // Node *n2 = new Node(18);
    // Node *n3 = new Node(2);
    // Node *n4 = new Node(9);
    // Node *n5 = new Node(15);
    // Node *n6 = new Node(19);
    // Node *n7 = new Node(17);

    // root->left = n1;
    // root->right = n2;
    // n1->parent = root;
    // n2->parent = root;
    // n1->left = n3;
    // n1->right = n4;
    // n3->parent = n1;
    // n4->parent = n1;
    // n2->left = n5;
    // n2->right = n6;
    // n5->parent = n2;
    // n6->parent = n2;
    // n5->right = n7;
    // n7->parent = n5;

    // print_tree(root);

    // Node *x = new Node(13);
    // tree_insert(root, x);
    // print_tree(root);

    QueueByStacks q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    std::cout << q.dequeue() << std::endl;
    q.enqueue(4);
    q.enqueue(5);
    std::cout << q.dequeue() << std::endl;
    std::cout << q.dequeue() << std::endl;
    std::cout << q.dequeue() << std::endl;
    std::cout << q.dequeue() << std::endl;
    std::cout << q.dequeue() << std::endl;

    return 0;
}