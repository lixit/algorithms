#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <list>

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
// only find max_index and swap once, more efficient
void selection_sort(std::vector<int> &v) {
    // here if v.size() is 0, right will be -1
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

void selection_sort2(std::vector<int> &v) {
    if (v.size() == 0) {
        return;
    }

    for (int left = 0; left < v.size() - 1; ++left) {
        int min_index = left;

        for (int j = left + 1; j < v.size(); ++j) {
            if (v[j] < v[min_index]) {
                min_index = j;
            }
        }
        std::swap(v[min_index], v[left]);
    }
}

// essentially the same as 2, but use library function
// find smallest element, exechange it with index 0. do it n - 1 times
void selection_sort3(std::vector<int> &numbers) {
    if (numbers.size() == 0) {
        return;
    }
    // warning
    for (int i = 0; i < numbers.size() - 1; ++i) {
        int curr = numbers[i];
        std::vector<int>::iterator it = std::min_element(numbers.begin() + i + 1, numbers.end());

        if (*it < curr) {
            std::swap(*it, numbers[i]);
        }
    }
}

// same as 2, but using swap to find smallest element, not efficient
void selection_sort4(std::vector<int> &n) {
    if (n.size() == 0) {
        return;
    }
    // warning: infinity loop
    for (int i = 0; i < n.size() - 1; ++i) {
        for (int j = i + 1; j < n.size(); ++j) {
            if (n[i] > n[j]) {
                std::swap(n[i], n[j]);
            }
        }
    }
}

// merge [p, q] and [q+1, r]
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

// book's idea
// travesal from end to begin, each time the first one is the smallest
// pointer j is align with i, i and j are at most at v.size() - 2
void bubble_sort(std::vector<int> &v) {
    if (v.size() == 0) {
        return;
    }
    // warning, if v.size() == 0, v.size() - 1 is the max unsigned integer
    for (int i = 0; i < v.size() - 1; ++i) {

        for (int j = v.size() - 2; j >= i; --j) {
            if (v[j + 1] < v[j]) {
                std::swap(v[j], v[j + 1]);
            }
        }
    }
}

// since travsel from begin to end. each time last one is the biggest
void bubble_sort2(std::vector<int> &n) {

    int right = n.size() - 1;
    while (right >= 0) {

        for (int i = 0; i < right; ++i) {
            if (n[i] > n[i + 1]) {
                std::swap(n[i], n[i + 1]);
            }
        }
        --right;
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