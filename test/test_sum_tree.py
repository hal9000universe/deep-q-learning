import unittest
import Base.sum_tree as sum_tree
import random
import numpy
import math


class TestSumTree(unittest.TestCase):

    def test_for_nans(self) -> None:
        size = 100000
        tree = sum_tree.gen_tree(size)

        self.assertFalse(numpy.isnan(tree).any())

        for i in range(1000):
            sum_tree.update(tree, random.randint(0, size), random.uniform(0, 1))

        self.assertFalse(numpy.isnan(tree).any())
        for i in range(1000):
            sum_tree.retrieve(tree, random.uniform(0, tree[1]))

        self.assertFalse(numpy.isnan(tree).any())
        for i in range(10):
            sum_tree.v_retrieve(tree, random.randint(1, 1000))

    def test_gen_tree(self):
        for i in range(4, 10000):
            tree = sum_tree.gen_tree(i)
            self.assertTrue(math.ceil(math.log2(tree.size)) == math.floor(math.log2(tree.size)))
            self.assertEqual(tree[1], 0)

    def test_update(self):
        tree = sum_tree.gen_tree(10000)
        for i in range(100):
            for j in range(100):
                sum_tree.update(tree, j, 0)

            self.assertAlmostEqual(tree[1], 0)
            values = [0.0 for _ in range(100)]
            for j in range(100):
                value = random.uniform(0, 100)
                values[j] = value
                sum_tree.update(tree, j, value)
                self.assertAlmostEqual(tree[1], sum(values))

        for j in range(10000):
            sum_tree.update(tree, j, j)

        for j in range(1, tree.size // 2):
            self.assertAlmostEqual(tree[j], tree[2 * j] + tree[2 * j + 1])

    def test_v_retrieve(self):
        tree = sum_tree.gen_tree(5000)
        for i in range(5000):
            sum_tree.update(tree, i, random.uniform(0, 10))

        for i in range(1000):
            indices = sum_tree.v_retrieve(tree, random.randint(1, 250))
            for index in indices:
                self.assertTrue(0 <= index < 5000)

    def test_retrieve(self):
        tree = sum_tree.gen_tree(10000)

        for i in range(10000):
            sum_tree.update(tree, i, i)











