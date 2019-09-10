import torch
from itertools import product
from dataclasses import dataclass
from collections import deque
from typing import Any, Hashable


class Vertex:
    def __init__(self, key):
        self.key = key
        self.connected_to = {}

    def add_neighbor(self, nbr, wt=0):
        self.connected_to[nbr] = wt

    def __str__(self):
        return str(self.key) + ' ' + self.connected_to.__repr__()


class Graph:
    """
    Implemented as a dictionary linked vertex names to instances of the Vertex class

    """

    def __init__(self):
        self.vertex_list = {}

    @property
    def num_vertices(self):
        return len(self.vertex_list)

    def add_vertex(self, vert_name):
        new_vertex = Vertex(vert_name)
        self.vertex_list[vert_name] = new_vertex

    def __contains__(self, item):
        return item in self.vertex_list

    def add_edge(self, v1: Hashable, v2: Hashable, wt: Any = 0):
        if v1 not in self.vertex_list:
            self.add_vertex(v1)
        if v2 not in self.vertex_list:
            self.add_vertex(v2)
        self.vertex_list[v1].add_neighbor(self.vertex_list[v2], wt)

    def __iter__(self):
        """
        :return:  returns an iterator on the vertices
        """
        return iter(self.vertex_list.values())


@dataclass(init=False, eq=False)
class SL2Element:
    sign: bool
    exps: tuple

    S = torch.tensor([[0, 1], [-1, 0]], dtype=torch.long)
    S_inv = -S
    R = torch.tensor([[0, -1], [1, 1]], dtype=torch.long)
    R_inv = -R @ R

    def __init__(self, sign: bool, exps: iter):
        for i, exp in enumerate(exps):
            if not isinstance(exp, int):
                raise Exception(f"The {i}th expeonential is not an integer: {exp}")
            elif i == 0 and not 0 <= exp <= 2:
                raise Exception(f"The first exponential in the sequence is invalid of value {exp}")
            elif not (i % 2) and i > 0 and not 1 <= exp <= 2:
                raise Exception(f"The given sequence of exponentials is invalid for r at {i} of value {exp}")
            elif i % 2 and exp != 1:
                raise Exception(f"The given sequence of exponentials is invalid for s at {i} of value {exp}")
        self.sign = bool(sign)
        self.exps = tuple(exps)

    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1:])

    def word_length(self, len_fun: str = 'b') -> int:
        """This is the length as a word in R and S. Acceptable parameters:
        'b': count both S and R,
        'r': count only the R's,
        's': count only the S's,
        'a': count all of R, S, and sign
        'len': count the length the exponential sequence, basically looking at how much the two cyclic subgroups
        interact

        """
        if len_fun == 'b':
            return sum(self.exps)
        elif len_fun == 'r':
            return sum(self.exps[::2])
        elif len_fun == 's':
            return sum(self.exps[1::2])
        elif len_fun == 'len':
            if len(self.exps) == 0:
                return 0
            return len(self.exps) if self.exps[0] else len(self.exps) - 1
        elif len_fun == 'a':
            return int(self.sign) + sum(self.exps)
        else:
            raise Exception(f"{len_fun} is an invalid input into the SL2Element word_length function")

    def __hash__(self):
        return hash((self.sign, self.exps))

    def times_s(self):
        if len(self.exps) == 0:
            return SL2Element(self.sign, (0, 1))
        elif len(self.exps) == 2 and self.exps[0] == 0:
            # Edge case where there's no first, 'r', so removing the 's' would create (0,) instead of ()
            return SL2Element(not self.sign, ())
        elif not (len(self.exps) % 2):
            # If even, the last element is an S, which cancels with the multiplied S, and changes the sign
            return SL2Element(not self.sign, self.exps[:-1])
        else:
            return SL2Element(self.sign, self.exps + (1,))

    def times_r(self):
        if not (len(self.exps) % 2):
            # If even, the last element is an S
            return SL2Element(self.sign, self.exps + (1,))
        elif self.exps[-1] == 1:
            # If odd but the last exponent is a one, then we just increase the last term
            return SL2Element(self.sign, self.exps[:-1] + (2,))
        else:
            return SL2Element(not self.sign, self.exps[:-1])

    def __mul__(self, other):
        """
        A very inefficient multiplication algorithm unless the second element has word length one

        :param other: Another SL2 element
        :return: returns an SL2 element which is the product of the two elements in standard form
        """
        new_ele = SL2Element(self.sign ^ other.sign, self.exps)
        for i, x in enumerate(other.exps):
            if not (i % 2):
                for j in range(x):
                    new_ele = new_ele.times_r()
            else:
                new_ele = new_ele.times_s()
        return new_ele

    def inv(self):
        """

        :return: Outputs the inverse SL2(Z) element
        """
        queue = deque()
        for i, x in enumerate(self.exps):
            if x == 0:
                continue
            else:
                letter = 2 if x == 1 and not (i % 2) else 1
            queue.appendleft(letter)
        if not len(self.exps) % 2:
            # If the length is even, the last letter is 's', so we start at 0 in the inverse
            queue.appendleft(0)
        return SL2Element(self.sign, queue)

    def inv_matrix(self) -> torch.tensor:
        """
        Computes the matrix representation of the word

        :return mat: a torch tensor
        """
        mat = torch.eye(2, dtype=torch.long) if not self.sign else -torch.eye(2, dtype=torch.long)
        for i, g in enumerate(self.exps):
            # If odd, we multiply by S, if even multiply by R
            for j in range(g):
                mat = self.S_inv @ mat if i % 2 else self.R_inv @ mat
        return mat

    def distortion(self, filter_size) -> tuple:
        """
        Computes the filter distortion from the group element by computing the vectors betweeen opposite angles in the
        deformed filter

        :param filter_size: A tuple of the sizes
        :return:
        """
        dist1 = self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t()
        dist2 = self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long).t() \
                - self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long).t()
        dist = [abs(x.item()) if abs(x) >= abs(y) else abs(y.item()) for x, y in zip(dist1, dist2)]
        return tuple(dist)

    def shift(self, filter_size) -> tuple:
        return -min(
            (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t())[0].item(),
            (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long).t())[0].item(),
            (self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long).t())[0].item(),
            0
        ), \
               -min(
                   (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t())[
                       1].item(),
                   (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long)).t()[1].item(),
                   (self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long)).t()[1].item(),
                   0
               )

    def key(self) -> tuple:
        return self.word_length(), len(self.exps), self.exps + (self.sign,)

    def __eq__(self, other):
        return self.sign == other.sign and self.exps == other.exps

    def __le__(self, other):
        return self.key() <= other.key()

    def __lt__(self, other):
        return self.key() < other.key()

    def __ge__(self, other):
        return self.key() >= other.key()

    def __gt__(self, other):
        return self.key() > other.key()

    @classmethod
    def cayley_ball(cls, radius: int, len_fun='b'):
        """
        A depth first search to create a ball in the Cayley graph (technically it's not the Cayley graph because
        we don't count the sign in the length, but close enough.

        :param radius: The radius of the desired ball
        :param len_fun: Flags whether to count S in the word length
        :return: The corresponding ball in the Cayley graph of SL(2,Z)
        """
        cayley = Graph()
        identity = cls(False, ())
        neg_identity = cls(True, ())
        cayley.add_vertex(identity)
        queue = deque([identity, neg_identity])
        placed_in_queue = {identity, neg_identity}
        while queue:
            current = queue.popleft()
            if current.times_r() not in placed_in_queue and current.times_r().word_length(len_fun) <= radius:
                cayley.add_edge(current, current.times_r(), 'R')
                queue.append(current.times_r())
                placed_in_queue.add(current.times_r())
            if current.times_s() not in placed_in_queue and current.times_s().word_length(len_fun) <= radius:
                cayley.add_edge(current, current.times_s(), 'S')
                queue.append(current.times_s())
                placed_in_queue.add(current.times_s())
        return cayley


@dataclass(init=False, eq=False)
class SL2pmElement:
    signs: torch.tensor
    exps: tuple

    S = torch.tensor([[0, 1], [-1, 0]], dtype=torch.long)
    S_inv = -S
    R = torch.tensor([[0, -1], [1, 1]], dtype=torch.long)
    R_inv = -R @ R

    def __init__(self, signs: iter, exps: iter):
        """

        :param signs: A tuple or tensor of +/- one. Unlike SL2Element, don't use booleans.
        :param exps:
        """
        for i, exp in enumerate(exps):
            if not isinstance(exp, int):
                raise Exception(f"The {i}th expeonential is not an integer: {exp}")
            elif i == 0 and not 0 <= exp <= 2:
                raise Exception(f"The first exponential in the sequence is invalid of value {exp}")
            elif not (i % 2) and i > 0 and not 1 <= exp <= 2:
                raise Exception(f"The given sequence of exponentials is invalid for r at {i} of value {exp}")
            elif i % 2 and exp != 1:
                raise Exception(f"The given sequence of exponentials is invalid for s at {i} of value {exp}")
        if isinstance(signs, torch.Tensor):
            if signs.shape != torch.Size([2]):
                raise Exception(
                    f"The signs parameter should be a pair of booleans. Instead got tensor of shape {signs.shape}")
        elif hasattr(signs, '__len__'):
            if len(signs) != 2:
                raise Exception(f"The signs parameter should be a pair of +/- 1. Instead got {len(signs)} elements")
            elif signs[0] not in {1, -1} or signs[1] not in {1, -1}:
                raise Exception(f"The signs parameter should be a pair a pair of +/- 1. Instead got {signs}")

        self.signs = torch.tensor([x for x in signs], dtype=torch.long)
        self.exps = tuple(exps)

    @classmethod
    def from_list(cls, lst):
        return cls([lst[0], lst[1]], lst[2:])

    def word_length(self, len_fun: str = 'b') -> int:
        """This is the length as a word in R and S. Acceptable parameters:
        'b': count both S and R,
        'r': count only the R's,
        's': count only the S's,
        'a': count all of R, S, and sign
        'len': count the length the exponential sequence, basically looking at how much the two cyclic subgroups
        interact


        """
        if len_fun == 'b':
            return sum(self.exps)
        elif len_fun == 'r':
            return sum(self.exps[::2])
        elif len_fun == 's':
            return sum(self.exps[1::2])
        elif len_fun == 'len':
            if len(self.exps) == 0:
                return 0
            return len(self.exps) if self.exps[0] else len(self.exps) - 1
        elif len_fun == 'a':
            return sum(self.exps) + self.signs[self.signs == -1].shape[0]
        else:
            raise Exception(f"{len_fun} is an invalid input into the SL2Element word_length function")

    def __hash__(self):
        return hash((str(self.signs), self.exps))

    def times_s(self):
        if len(self.exps) == 0:
            return SL2pmElement(self.signs, (0, 1))
        elif len(self.exps) == 2 and self.exps[0] == 0:
            # Edge case where there's no first, 'r', so removing the 's' would create (0,) instead of ()
            return SL2pmElement(-self.signs, ())
        elif not (len(self.exps) % 2):
            # If even, the last element is an S, which cancels with the multiplied S, and changes the sign
            return SL2pmElement(-self.signs, self.exps[:-1])
        else:
            return SL2pmElement(self.signs, self.exps + (1,))

    def times_r(self):
        if not (len(self.exps) % 2):
            # If even, the last element is an S
            return SL2pmElement(self.signs, self.exps + (1,))
        elif self.exps[-1] == 1:
            # If odd but the last exponent is a one, then we just increase the last term
            return SL2pmElement(self.signs, self.exps[:-1] + (2,))
        else:
            return SL2pmElement(-self.signs, self.exps[:-1])

    def __mul__(self, other):
        """
        A very inefficient multiplication algorithm unless the second element has word length one

        :param other: Another SL2 element
        :return: returns an SL2 element which is the product of the two elements in standard form
        """
        new_ele = SL2pmElement(self.signs * other.signs, self.exps)
        for i, x in enumerate(other.exps):
            if not (i % 2):
                for j in range(x):
                    new_ele = new_ele.times_r()
            else:
                new_ele = new_ele.times_s()
        return new_ele

    def inv(self):
        """

        :return: Outputs the inverse SL2(Z) element
        """
        queue = deque()
        for i, x in enumerate(self.exps):
            if x == 0:
                continue
            else:
                letter = 2 if x == 1 and not (i % 2) else 1
            queue.appendleft(letter)
        if not len(self.exps) % 2:
            # If the length is even, the last letter is 's', so we start at 0 in the inverse
            queue.appendleft(0)
        return SL2pmElement(self.signs, queue)

    def inv_matrix(self) -> torch.tensor:
        """
        Computes the matrix representation of the word

        :return mat: a torch tensor
        """
        mat = torch.diag(self.signs)
        for i, g in enumerate(self.exps):
            # If odd, we multiply by S, if even multiply by R
            for j in range(g):
                mat = mat @ self.S_inv if i % 2 else mat @ self.R_inv
        return mat

    def distortion(self, filter_size) -> tuple:
        """
        Computes the filter distortion from the group element by computing the vectors betweeen opposite angles in the
        deformed filter

        :param filter_size: A tuple of the sizes
        :return:
        """
        dist1 = self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t()
        dist2 = self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long).t() \
                - self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long).t()
        dist = [abs(x.item()) if abs(x) >= abs(y) else abs(y.item()) for x, y in zip(dist1, dist2)]
        return tuple(dist)

    def shift(self, filter_size) -> tuple:
        return -min(
            (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t())[0].item(),
            (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long).t())[0].item(),
            (self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long).t())[0].item(),
            0
        ), \
               -min(
                   (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, filter_size[1] - 1), dtype=torch.long).t())[
                       1].item(),
                   (self.inv_matrix() @ torch.tensor((filter_size[0] - 1, 0), dtype=torch.long)).t()[1].item(),
                   (self.inv_matrix() @ torch.tensor((0, filter_size[1] - 1), dtype=torch.long)).t()[1].item(),
                   0
               )

    def key(self) -> tuple:
        return self.word_length(), len(self.exps), self.exps + tuple(self.signs)

    def __eq__(self, other):
        return all(self.signs == other.signs) and self.exps == other.exps

    def __le__(self, other):
        return self.key() <= other.key()

    def __lt__(self, other):
        return self.key() < other.key()

    def __ge__(self, other):
        return self.key() >= other.key()

    def __gt__(self, other):
        return self.key() > other.key()

    @classmethod
    def cayley_ball(cls, radius: int, len_fun='b'):
        """
        A depth first search to create a ball in the Cayley graph (technically it's not the Cayley graph because
        we don't count the sign in the length, but close enough.

        :param radius: The radius of the desired ball
        :param len_fun: Flags whether to count S in the word length
        :return: The corresponding ball in the Cayley graph of SL(2,Z)
        """
        cayley = Graph()
        identity = cls((1, 1), ())
        lst = [identity, cls((-1, 1), ()), cls((1, -1), ()), cls((-1, -1), ())]
        cayley.add_vertex(identity)
        queue = deque(lst)
        placed_in_queue = set(lst)
        while queue:
            current = queue.popleft()
            if current.times_r() not in placed_in_queue and current.times_r().word_length(len_fun) <= radius:
                cayley.add_edge(current, current.times_r(), 'R')
                queue.append(current.times_r())
                placed_in_queue.add(current.times_r())
            if current.times_s() not in placed_in_queue and current.times_s().word_length(len_fun) <= radius:
                cayley.add_edge(current, current.times_s(), 'S')
                queue.append(current.times_s())
                placed_in_queue.add(current.times_s())
        return cayley


class CayleyGraph:

    def __init__(self, radius: int, len_fun: str = 'b', filter_size=(1, 1), group='SL2'):
        """
        An abstract Cayley graph class. Stores the inv_mat and distortion of each element for the sake of speed

        :param radius: The radius of the desired ball
        :param len_fun: Flags whether to count S in the word length
        :return: The corresponding ball in the Cayley graph of the group (eg: SL2Element)
        """
        if group == 'SL2':
            group_class = SL2Element
        elif group == 'SL2pm':
            group_class = SL2pmElement
        else:
            raise Exception(f"{group} is not a valid group option")
        self.graph = group_class.cayley_ball(radius, len_fun=len_fun)
        self.dist = {}
        self.inv = {}
        self.shift = {}
        for x in self.graph:
            self.inv[x.key] = x.key.inv_matrix()
            self.dist[x.key] = x.key.distortion(filter_size)
            self.shift[x.key] = x.key.shift(filter_size)

    def __iter__(self):
        """
        Iterates over the vertex names

        :return: Iterator over the vertex names
        """
        return iter(self.graph.vertex_list.keys())

    def __len__(self) -> int:
        return len(self.graph.vertex_list)

    def max_filter_index(self) -> tuple:
        """

        :return: largest absolute filter position after applying the elements of the Cayley ball to the self.filter_size
        """
        max_filter_dimensions = [0, 0]
        for g in self:
            for i in range(2):
                max_filter_dimensions[i] = max(max_filter_dimensions[i], self.dist[g][i] + 1)
        return tuple(max_filter_dimensions)

    @classmethod
    def num_of_ele(cls, radius: int, len_fun: str = 'b') -> int:
        """

        :param radius: radius of the Cayley graph ball
        :param len_fun: length function to use to define ball, options are 'b', 's', 'r', or 'len'
        :return: Number of elements in the Cayley graph ball
        """
        return len(cls(radius, len_fun))


if __name__ == '__main__':
    pass
