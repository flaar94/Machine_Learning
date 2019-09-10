import pytest
import my_layers
from SL2_classes import SL2Element, CayleyGraph, Vertex, SL2pmElement
from my_layers import GroupLocalSL2, LatticeLocalSL2
from random import shuffle
import torch
from itertools import product, combinations


def test_sl2_init_error_index_0():
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (3, 1, 1))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (-1, 1, 1))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1.5,))


def test_sl2_init_error_index_r():
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1, 1, 3))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1, 1, 0))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1, 1, 5, 1))


def test_sl2_init_error_index_s():
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1, 0))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (1, 2, 1))
    with pytest.raises(Exception):
        my_layers.SL2Element(False, (2, 1, 2, 5))


def test_mult1():
    a = my_layers.SL2Element(False, (2, 1, 1))
    b = my_layers.SL2Element(False, (2, 1, 1))
    assert a * b == my_layers.SL2Element(True, ())


def test_mult2():
    a = my_layers.SL2Element(False, (2, 1, 1, 1))
    b = my_layers.SL2Element(False, (2, 1, 1))
    assert a * b == my_layers.SL2Element(False, (2, 1, 1, 1, 2, 1, 1))


def test_mult3():
    a = my_layers.SL2Element(False, (2, 1, 1, 1))
    b = my_layers.SL2Element(False, (1,))
    c = my_layers.SL2Element(False, (0, 1))
    assert a * b == a.times_r()
    assert a * c == a.times_s()


def test_mult4():
    a = my_layers.SL2Element(False, (2, 1, 2))
    b = my_layers.SL2Element(False, (1,))
    c = my_layers.SL2Element(False, (0, 1))
    assert a * b == a.times_r()
    assert a * c == a.times_s()


def test_mult_s():
    identity = my_layers.SL2Element(False, ())
    neg_identity = my_layers.SL2Element(True, ())
    s = my_layers.SL2Element(False, (0, 1))
    a = my_layers.SL2Element(False, (2,))
    b = my_layers.SL2Element(False, (2, 1))
    c = my_layers.SL2Element(True, (2,))
    assert identity.times_s() == s
    assert a.times_s() == b
    assert b.times_s() == c
    assert s.times_s() == neg_identity


def test_mult_r():
    identity = my_layers.SL2Element(False, ())
    r = my_layers.SL2Element(False, (1,))
    a = my_layers.SL2Element(False, (2, 1))
    b = my_layers.SL2Element(False, (2, 1, 1))
    c = my_layers.SL2Element(False, (2, 1, 2))
    assert identity.times_r() == r
    assert a.times_r() == b
    assert b.times_r() == c


def test_vertex_class_add_neighbor():
    v1 = my_layers.Vertex("Potato")
    assert v1.key == "Potato"
    assert v1.connected_to == {}
    v2 = my_layers.Vertex("Cake")
    v1.add_neighbor(v2, 's1')
    assert v1.connected_to == {v2: 's1'}
    v3 = my_layers.Vertex("Pie")
    v1.add_neighbor(v3)
    v2.add_neighbor(v3, 5)
    assert v2.connected_to == {v3: 5}
    assert v1.connected_to == {v3: 0, v2: 's1'}


def test_graph_class_adding():
    g = my_layers.Graph()
    g.add_vertex("Monkey King")
    g.add_vertex("Tangseng")
    g.add_vertex("Zhu Bajie")
    g.add_edge("Tangseng", "Monkey King", "Guanyin")
    g.add_edge("Tangseng", "pie", 5)
    g.add_edge("Peter Pan", "Lost boys")
    assert g.num_vertices == 6
    new_set = set()
    for x in g:
        new_set.add(x.key)
    assert new_set == {"Monkey King", "Tangseng", "Zhu Bajie", "pie", "Peter Pan", "Lost boys"}
    assert g.vertex_list["Monkey King"].connected_to == {}
    assert set(g.vertex_list["Tangseng"].connected_to.values()) == {"Guanyin", 5}
    assert list(g.vertex_list["Peter Pan"].connected_to.values()) == [0]
    assert list(g.vertex_list["Lost boys"].connected_to) == []
    assert "Zhu Bajie" in g
    assert "Lost boys" in g


def test_sl2_element_equality():
    x = my_layers.SL2Element(False, (2, 1))
    y = my_layers.SL2Element(False, (2,))
    z = y.times_s()
    assert x != y
    assert x == z


def test_sl2_cayley_ball():
    for i in range(10):
        assert my_layers.SL2Element.cayley_ball(i)
    assert my_layers.SL2Element.cayley_ball(1).num_vertices == 6
    assert my_layers.SL2Element.cayley_ball(2).num_vertices == 12
    assert my_layers.SL2Element.cayley_ball(3).num_vertices == 20
    set_o_elements = {my_layers.SL2Element(False, ()), my_layers.SL2Element(False, (1,)),
                      my_layers.SL2Element(False, (0, 1)), my_layers.SL2Element(False, (2,)),
                      my_layers.SL2Element(False, (1, 1)), my_layers.SL2Element(False, (0, 1, 1)),
                      my_layers.SL2Element(True, ()), my_layers.SL2Element(True, (1,)),
                      my_layers.SL2Element(True, (0, 1)), my_layers.SL2Element(True, (2,)),
                      my_layers.SL2Element(True, (1, 1)), my_layers.SL2Element(True, (0, 1, 1))
                      }
    assert set(my_layers.SL2Element.cayley_ball(2).vertex_list.keys()) == set_o_elements


def test_count_only_s():
    a = my_layers.SL2Element(True, (1, 1, 1))
    b = my_layers.SL2Element(True, (1, 1, 1, 1))
    c = my_layers.SL2Element(False, (2, 1, 2))
    d = my_layers.SL2Element(True, (2,))
    assert a.word_length(len_fun='r') == 2
    assert b.word_length(len_fun='r') == 2
    assert c.word_length(len_fun='r') == 4
    assert d.word_length(len_fun='r') == 2


def test_matrix_inv():
    a = my_layers.SL2Element(False, (0, 1))
    for i, j in product(range(2), range(2)):
        assert a.inv_matrix()[i, j] == torch.tensor([[0, -1], [1, 0]], dtype=torch.int)[i, j]


def test_distortion():
    a = my_layers.SL2Element(False, (0, 1))
    assert a.distortion([2, 2]) == (1, 1)
    b = my_layers.SL2Element(False, (1,))
    assert b.distortion([2, 2]) == (2, 1)
    c = a
    for i in range(10):
        c = a * c
        assert c.distortion([2, 2]) == (1, 1)


def test_cayley_class():
    cayley = my_layers.CayleyGraph(5)
    assert cayley.graph.vertex_list.keys() == my_layers.SL2Element.cayley_ball(5).vertex_list.keys()
    a = my_layers.SL2Element(False, (0, 1))
    assert cayley.dist[a] == a.distortion([1, 1])
    b = my_layers.SL2Element(False, (1,))
    assert cayley.dist[b] == b.distortion([1, 1])


def test_shift():
    a = my_layers.SL2Element(False, (0, 1))
    assert a.shift((2, 2)) == (1, 0)


def test_inv():
    a = my_layers.SL2Element(False, (0, 1))
    assert a.inv() == a
    b = my_layers.SL2Element(True, (1,))
    assert b.inv() == my_layers.SL2Element(True, (2,))
    c = my_layers.SL2Element(True, (1, 1, 2, 1))
    assert c.inv() == my_layers.SL2Element(True, (0, 1, 1, 1, 2))
    d = my_layers.SL2Element(False, (1, 1, 2, 1, 1))
    assert d.inv() == my_layers.SL2Element(False, (2, 1, 1, 1, 2))


def test_word_length_options():
    a = SL2Element(True, (0, 1, 2, 1, 1, 1))
    assert a.word_length('s') == 3
    assert a.word_length('len') == 5
    b = SL2Element(False, ())
    assert b.word_length('s') == 0
    assert b.word_length('len') == 0
    assert b.word_length('a') == 0
    assert b.word_length('r') == 0
    c = SL2Element(True, ())
    assert c.word_length('s') == 0
    assert c.word_length('len') == 0
    assert c.word_length('a') == 1
    assert c.word_length('r') == 0
    with pytest.raises(Exception):
        a.word_length('potato')


def test_inequalities():
    assert SL2Element(True, (0, 1, 2)) > SL2Element(True, (0, 1))
    assert SL2Element(True, (0, 1, 2)) >= SL2Element(True, (0, 1))
    assert SL2Element(True, (0, 1, 2)) >= SL2Element(True, (0, 1, 2))
    assert SL2Element(True, (2, 1)) < SL2Element(True, (1, 1, 1, 1))
    assert SL2Element(True, (2, 1)) <= SL2Element(True, (1, 1, 1, 1))
    assert SL2Element(True, (2, 1)) <= SL2Element(True, (2, 1))


def test_iter():
    cayley = CayleyGraph(1, 'len')
    assert len(cayley) == 8
    eles = set()
    for ele in cayley:
        eles.add(ele)
    assert eles == {SL2Element(False, ()), SL2Element(False, (1,)), SL2Element(False, (2,)), SL2Element(False, (0, 1)),
                    SL2Element(True, ()), SL2Element(True, (1,)), SL2Element(True, (2,)), SL2Element(True, (0, 1))}
    assert cayley.max_filter_index() == (1, 1)


def test_num_of_ele():
    assert CayleyGraph.num_of_ele(1, 'len') == 8
    assert CayleyGraph.num_of_ele(2, 'len') == 16


def test_sign_length():
    assert SL2Element(False, (2, 1, 1)).word_length('a') == 4
    assert SL2Element(True, (1, 1, 2, 1)).word_length('a') == 6


def test_string_rep():
    assert isinstance(str(SL2Element), str)
    cayley = CayleyGraph(5, 'len')
    for x, y in combinations(cayley, 2):
        assert str(x) != str(y)
    v1 = Vertex("Hello")
    v2 = Vertex("Goodbye!")
    v3 = Vertex(5)
    v4 = Vertex((1, 2, 3))
    assert isinstance(str(v1), str)
    assert isinstance(str(v3), str)
    assert isinstance(str(v4), str)
    assert v1 != v2


def test_pm_word_length_options():
    a = SL2pmElement((1, 1), (0, 1, 2, 1, 1, 1))
    assert a.word_length('s') == 3
    assert a.word_length('len') == 5
    b = SL2pmElement((-1, -1), ())
    assert b.word_length('s') == 0
    assert b.word_length('len') == 0
    # assert b.word_length('a') == 2
    assert b.word_length('r') == 0
    with pytest.raises(Exception):
        a.word_length('potato')


def test_sl2pmballs():
    assert len(CayleyGraph(1, 'len', group='SL2pm')) == 16
    assert len(CayleyGraph(2, 'len', group='SL2pm')) == 32
