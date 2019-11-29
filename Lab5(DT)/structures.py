import numpy as np
import random


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.y = None
        self.f = None
        self.b = None


def gini(sum_l, sum_r, L, R):
    return (sum_l / L if L > 0 else 0) + \
           (sum_r / R if R > 0 else 0)


class DecisionTree:
    def __init__(self, h=None):
        self.X = None
        self.y = None
        self.n = None
        self.m = None
        self.k = None
        self.root = None
        self.h = h

    def __gini_split(self, init_objs, f):
        T1, T2 = [], init_objs
        l, r = [0] * self.k, [0] * self.k
        best_T1, best_T2 = [], T2
        best_b = -1e9
        for i in init_objs:
            r[self.y[i]] += 1
        sum_l = 0
        sum_r = sum([x * x for x in r])
        best_gini = gini(sum_l, sum_r, len(T1), len(T2))
        for j in range(len(init_objs)):
            i = init_objs[j]
            sum_r += (r[self.y[i]] - 1) ** 2 - r[self.y[i]] ** 2
            r[self.y[i]] -= 1
            sum_l += (l[self.y[i]] + 1) ** 2 - l[self.y[i]] ** 2
            l[self.y[i]] += 1
            T1.append(T2[0])
            T2 = T2[1:]
            tmp = gini(sum_l, sum_r, len(T1), len(T2))
            if best_gini < tmp or \
                    np.isclose(best_gini, tmp) and abs(len(T1) - len(T2)) < abs(len(best_T1) - len(best_T2)):
                best_gini = tmp
                best_T1 = list(T1)
                best_T2 = T2
                best_b = (self.X[i][f] + self.X[init_objs[j + 1]][f]) / 2 if j + 1 < len(init_objs) else 1e9
        return best_T1, best_T2, best_b, best_gini

    def __build(self, T, objects, parent_class, d):
        if len(objects) == 0:
            T.y = parent_class
            return
        ys = [0] * self.k
        for i in objects:
            ys[self.y[i]] += 1
        max_y = np.argmax(ys)
        if ys[max_y] == len(objects) or self.h is not None and d == self.h:
            T.y = max_y
            return
        best_gini = 0
        best_T1, best_T2 = [], []
        for f in range(self.m):
            T1, T2, b, cur_gini = self.__gini_split(sorted(objects, key=lambda obj: self.X[obj][f]), f)
            if best_gini < cur_gini:
                best_gini = cur_gini
                T.f = f
                T.b = b
                best_T1 = T1
                best_T2 = T2

        T.left = Node()
        self.__build(T.left, best_T1, max_y, d + 1)
        T.right = Node()
        self.__build(T.right, best_T2, max_y, d + 1)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n = len(X)
        self.m = len(X[0])
        self.k = max(y) + 1
        self.root = Node()
        self.__build(self.root, range(self.n), -1, 0)

    def __predict(self, node, x):
        if node.left is None:
            return node.y
        if x[node.f] < node.b:
            return self.__predict(node.left, x)
        return self.__predict(node.right, x)

    def predict(self, x):
        return self.__predict(self.root, x)


def random_objects(X, y, subset_size):
    n = len(X)
    new_X, new_y = [], []
    for _ in range(subset_size):
        i = random.randint(0, n - 1)
        new_X.append(X[i])
        new_y.append(y[i])
    return new_X, new_y


def random_features(X, subset_size):
    m = len(X[0])
    indices = []
    for _ in range(subset_size):
        indices.append(random.randint(0, m - 1))
    return [[x[i] for i in indices] for x in X]


class RandomForest:
    def __init__(self, is_random_objects=False, is_random_features=False):
        self.is_random_objects = is_random_objects
        self.is_random_features = is_random_features
        self.trees = None
        self.k = 0

    def fit(self, X, y, count_trees=10):
        self.trees = []
        for _ in range(count_trees):
            tree = DecisionTree()
            final_X, final_y = X, y
            if self.is_random_features:
                final_X = random_features(final_X, int(np.sqrt(len(X[0]))))
            if self.is_random_objects:
                final_X, final_y = random_objects(final_X, final_y, int(np.sqrt(len(X))))
            tree.fit(final_X, final_y)
            self.k = max(self.k, tree.k)
            self.trees.append(tree)

    def predict(self, x):
        ys = [0] * self.k
        for tree in self.trees:
            ys[tree.predict(x)] += 1
        return int(np.argmax(ys))
