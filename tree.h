
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