#include <iostream>
#include <vector>
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

int main() {
    std::vector<int> v = {5, 2, 4, 6, 1, 3};
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

    std::vector<int> v2 = {329, 457, 657, 839, 436, 720, 355};
    radix_sort(v2, 3);
    for (auto i : v2) {
        std::cout << i << " ";
    }

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

    // std::vector<std::vector<int>> F = matrix_multiply_recursive(C, D);
    // print_matrix(F);

    // std::vector<std::vector<int>> E = matrix_multiply_recursive(A, B);
    // print_matrix(E);

    // std::vector<std::vector<int>> I = matrix_multiply_recursive(G, H);
    // print_matrix(I);

    return 0;
}