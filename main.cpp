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

const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");

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

enum Color {WHITE, GRAY, BLACK, RED};
template<typename T>
struct Node
{
    Node(T key, Color color = RED) : key(key), color(color) {}
    T key;
    Node *parent = nullptr;
    Node *left = nullptr;
    Node *right = nullptr;
    Color color;
};

template<typename T>
class BinarySearchTree
{
public:
    BinarySearchTree() : root(nullptr) {}
    ~BinarySearchTree() {
        free_tree(root);
    }

    void tree_insert(Node<T> *z) {
        Node<T> *x = root;
        Node<T> *y = nullptr;
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
    void inorder_tree_walk(Node<T> *x) {
        if (x != nullptr) {
            inorder_tree_walk(x->left);
            std::cout << x->key << " ";
            inorder_tree_walk(x->right);
        }
    }

    Node<T> *tree_search(Node<T> *x, T key) {
        if (x == nullptr or x->key == key) {
            return x;
        }
        if (key < x->key) {
            return tree_search(x->left, key);
        } else {
            return tree_search(x->right, key);
        }
    }

    Node<T> *iterative_tree_search(Node<T> *x, T key) {
        while (x != nullptr and x->key != key) {
            if (key < x->key) {
                x = x->left;
            } else {
                x = x->right;
            }
        }
        return x;
    }

    Node<T> *tree_minimum(Node<T> *x) {
        while (x->left != nullptr) {
            x = x->left;
        }
        return x;
    }

    Node<T> *tree_maximum(Node<T> *x) {
        while (x->right != nullptr) {
            x = x->right;
        }
        return x;
    }

    Node<T> *tree_successor(Node<T> *x) {
        if (x->right != nullptr) {
            return tree_minimum(x->right);
        }
        Node<T> *y = x->parent;
        while (y != nullptr and x == y->right) {
            x = y;
            y = y->parent;
        }
        return y;
    }

    Node<T> *tree_predecessor(Node<T> *x) {
        if (x->left != nullptr) {
            return tree_maximum(x->left);
        }
        Node<T> *y = x->parent;
        while (y != nullptr and x == y->left) {
            x = y;
            y = y->parent;
        }
        return y;
    }

    void tree_delete(Node<T> *z) {
        if (z->left == nullptr) {
            transplant(root, z, z->right);
        } else if (z->right == nullptr) {
            transplant(root, z, z->left);
        } else {
            Node<T> *y = tree_minimum(z->right);
            if (y->parent != z) {
                transplant(root, y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            transplant(root, z, y);
            y->left = z->left;
            y->left->parent = y;
        }
    }
    void print_tree() {
        print_tree(root);
    }

protected:

    Node<T> *root;

    void free_tree(Node<T> *&root) {
        if (root != nullptr) {
            free_tree(root->left);
            free_tree(root->right);
            delete root;
            root = nullptr;
        }
    }

    void print_tree(Node<T> *root, int level = 0) {
        if (root != nullptr) {
            print_tree(root->right, level + 1);
            for (int i = 0; i < level; ++i) {
                std::cout << "    ";
            }
            if (root->color == RED) {
                std::cout << red << root->key << reset << std::endl;
            } else {
                std::cout << root->key << std::endl;
            }

            print_tree(root->left, level + 1);
        }
    }

    void transplant(Node<T> *&root, Node<T> *u, Node<T> *v) {  // replace u with v
        if (u->parent == nullptr) {  // u is the root
            root = v;
        } else if (u == u->parent->left) {      // u is a left child
            u->parent->left = v;
        } else {                                // u is a right child
            u->parent->right = v;
        }
        if (v != nullptr) {
            v->parent = u->parent;
        }
    }
};


template<typename T>
class RedBlackTree : public BinarySearchTree<T>
{
public:
    RedBlackTree() : BinarySearchTree<T>() {}

    void rb_insert(Node<T> *z) {
        // start by inserting z as if it were an ordinary binary search tree
        this->tree_insert(z);

        z->left = nullptr;
        z->right = nullptr;
        z->color = RED;
        rb_insert_fixup(this->root, z);
    }

private:
    // assume x->right is not nullptr
    // result: indorder tree walk of the input tree and the modified tree produce the same listing of key values
    void left_rotate(Node<T> *&root, Node<T> *x) {
        Node<T> *y = x->right;
        x->right = y->left;     // turn y's left subtree into x's right subtree
        if (y->left != nullptr) {       // if y's left subtree is not nullptr
            y->left->parent = x;        
        }
        y->parent = x->parent;      // x's parent becomes y's parent
        if (x->parent == nullptr) {           // if x was the root
            root = y;                           // ...then y becomes the root
        } else if (x == x->parent->left) {   // otherwise, if x was a left child
            x->parent->left = y;                // ...then y becomes a left child
        } else {
            x->parent->right = y;               // otherwise, y becomes a right child
        }
        y->left = x;
        x->parent = y;
    }

    void right_rotate(Node<T> *&root, Node<T> *y) {
        Node<T> *x = y->left;
        y->left = x->right;
        if (x->right != nullptr) {
            x->right->parent = y;
        }
        x->parent = y->parent;
        if (y->parent == nullptr) {
            root = x;
        } else if (y == y->parent->left) {
            y->parent->left = x;
        } else {
            y->parent->right = x;
        }
        x->right = y;
        y->parent = x;
    }

    void rb_insert_fixup(Node<T> *&root, Node<T> *z) {
        while (z->parent && z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) { // is z's parent a left child? here `z->parent->parent` must exist, since z->parent is red, so z->parent is not the root
                Node<T> *y = z->parent->parent->right;  // y is z's uncle on the right of z's grandparent
                if (y->color == RED) {                  // case 1: z's parent and uncle are both red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;              // z move up 2 levels
                } else {                                // z's uncle is black
                    if (z == z->parent->right) {        // case 2: z's uncle is black and z is a right child
                        z = z->parent;
                        left_rotate(root, z);
                    }
                    z->parent->color = BLACK;           // case 3: z's uncle is black and z is a left child
                    z->parent->parent->color = RED;
                    right_rotate(root, z->parent->parent);
                }
            } else {                                    // z's parent is a right child
                Node<T> *y = z->parent->parent->left;   // y is z's uncle on the left
                Color y_color = BLACK;                  // leaf nodes are black
                if (y != nullptr) {                     // y might be nullptr
                    y_color = y->color;
                }
                if (y_color == RED) {                   // case 1: z's parent and uncle are both red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {                                // z's uncle is black
                    if (z == z->parent->left) {         // case 2: z's uncle is black and z is a left child
                        z = z->parent;
                        right_rotate(root, z);
                    }                                   // case 3: z's uncle is black and z is a right child
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    left_rotate(root, z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }

    void rb_transplant(Node<T> *&root, Node<T> *u, Node<T> *v) {
        if (u->parent == nullptr) {
            root = v;
        } else if (u == u->parent->left) {
            u->parent->left = v;
        } else {
            u->parent->right = v;
        }
        // could be a problem if v is nullptr. the text book take advantage of this
        v->parent = u->parent;
    }
};


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

class Graph {
public:
    Graph(int V) : 
        V(V), adj(V), color(V, WHITE), d(V, INT_MAX), f(V), pi(V, -1) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }

    void add_edge(const std::vector<std::pair<int, int>> &edges) {
        for (auto &e : edges) {
            add_edge(e.first, e.second);
        }
    }

    void print_graph() {
        for (int i = 0; i < V; ++i) {
            std::cout << i << ": ";
            for (auto &j : adj[i]) {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }

    void bfs(int s) {
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

    void dfs_iterative(int s) {
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

    void dfs_visit(int u, int &time) {
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

    void dfs() {
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
    void print_path_recursive(int s, int v) {
        if (v == s) {
            std::cout << s << " ";
        } else if (pi[v] == -1) {
            std::cout << "no path from " << s << " to " << v << " exists" << std::endl;
        } else {
            print_path_recursive(s, pi[v]);
            std::cout << v << " ";
        }
    }

    void print_path(int s, int v) {
        print_path_recursive(s, v);
        std::cout << std::endl;
    }

private:
    // size of the vertices
    int V;
    // Adjacency list 
    std::vector<std::vector<int>> adj;
    // color of each vertice
    std::vector<Color> color;
    // distance from the source, or time discovered
    std::vector<int> d;
    // time finished discovery
    std::vector<int> f;
    // parent of each vertice
    std::vector<int> pi;

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

    Graph g(8);
    g.add_edge({{0, 1}, {0, 3}, {1, 2}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}});
    g.print_graph();

    g.bfs(0);
    g.print_path(0, 7);

    g.dfs_iterative(0);
    g.print_path(0, 7);

    g.dfs();

    return 0;
}