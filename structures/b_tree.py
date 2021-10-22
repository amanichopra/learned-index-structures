from __future__ import (nested_scopes, generators, division, absolute_import, with_statement,
                        print_function, unicode_literals)
from treelib import Node, Tree

"""A item stored in a BTree node."""
class Item():
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __gt__(self, other):
        if self.k > other.k:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.k >= other.k:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.k == other.k:
            return True
        else:
            return False

    def __le__(self, other):
        if self.k <= other.k:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False
    def __repr__(self):
        return f"{{Key: {self.k}, Value: {self.v}}}"
    
    def __str__(self):
        return f"{{Key: {self.k}, Value: {self.v}}}"
    
"""A BTree implementation with search and insert functions. Capable of any order t."""
class BTree(object):
    """A simple B-Tree Node."""
    class Node(object):
    
        def __init__(self, t):
            self.keys = []
            self.children = []
            self.leaf = True
            # t is the order of the parent B-Tree. Nodes need this value to define max size and splitting.
            self._t = t

        """Split a node and reassign keys/children."""
        def split(self, parent, payload):
            new_node = self.__class__(self._t)
            mid_point = self.size//2
            split_value = self.keys[mid_point]
            parent.add_key(split_value)

            # Add keys and children to appropriate nodes
            new_node.children = self.children[mid_point + 1:]
            self.children = self.children[:mid_point + 1]
            new_node.keys = self.keys[mid_point+1:]
            self.keys = self.keys[:mid_point]

            # If the new_node has children, set it as internal node
            if len(new_node.children) > 0:
                new_node.leaf = False

            parent.children = parent.add_child(new_node)
            if payload < split_value:
                return self
            else:
                return new_node

        @property
        def _is_full(self):
            return self.size == 2 * self._t - 1

        @property
        def size(self):
            return len(self.keys)
        
        """Add a key to a node. The node will have room for the key by definition."""
        def add_key(self, value):
            self.keys.append(value)
            self.keys.sort()

        """Add a child to a node. This will sort the node's children, allowing for children
        to be ordered even after middle nodes are split.
        returns: an order list of child nodes"""
        def add_child(self, new_node):
            i = len(self.children) - 1
            while i >= 0 and self.children[i].keys[0] > new_node.keys[0]:
                i -= 1
            return self.children[:i + 1]+ [new_node] + self.children[i + 1:]

    """
    Create the B-tree. t is the order of the tree. Tree has no keys when created.
    This implementation allows duplicate key values, although that hasn't been checked
    strenuously.
    """
    def __init__(self, t):
    
        self._t = t
        if self._t <= 1:
            raise ValueError("B-Tree must have a degree of 2 or more.")
        self.root = self.Node(t)

    """Insert a new key of value payload into the B-Tree."""   
    def insert(self, payload):
        node = self.root
        # Root is handled explicitly since it requires creating 2 new nodes instead of the usual one.
        if node._is_full:
            new_root = self.Node(self._t)
            new_root.children.append(self.root)
            new_root.leaf = False
            # node is being set to the node containing the ranges we want for payload insertion.
            node = node.split(new_root, payload)
            self.root = new_root
    
        while not node.leaf:
            i = node.size - 1
            while i > 0 and payload < node.keys[i] :
                i -= 1
            if payload > node.keys[i]:
                i += 1

            next = node.children[i]
            if next._is_full:
                node = next.split(node, payload)
            else:
                node = next
        # Since we split all full nodes on the way down, we can simply insert the payload in the leaf.
        node.add_key(payload)

    """Return True if the B-Tree contains a key that matches the value."""
    def search(self, value, node=None):
        if node is None:
            node = self.root
        if value in node.keys:
            return True
        elif node.leaf:
            # If we are in a leaf, there is no more to check.
            return False
        else:
            i = 0
            while i < node.size and value > node.keys[i]:
                i += 1
            return self.search(value, node.children[i])
    
    def predict(self, value, node=None):
        if node is None:
            node = self.root
        if value in node.keys:
            return node.keys[node.keys.index(value)].v
        elif node.leaf:
            # If we are in a leaf, there is no more to check.
            return None
        else:
            i = 0
            while i < node.size and value > node.keys[i]:
                i += 1
            return self.predict(value, node.children[i])

    """Print an level-order representation."""
    def print_order(self, visual=False):
        if visual:
            tree = Tree()
            this_level = [self.root]
            tree.create_node(self.root.keys[0].k, self.root.keys[0].k)
            while this_level:
                next_level = []
                output = ""
                for node in this_level:
                    if node.children:
                        next_level.extend(node.children)
                        for child_node in node.children:
                            for item in child_node.keys:
                                tree.create_node(item.k, item.k, parent=node.keys[0].k)
                    output += str(node.keys) + " "
                this_level = next_level
            return tree.show()
               
        else:
            this_level = [self.root]
            while this_level:
                next_level = []
                output = ""
                for node in this_level:
                    if node.children:
                        next_level.extend(node.children)
                    output += str(node.keys) + " "
                print(output)
                this_level = next_level