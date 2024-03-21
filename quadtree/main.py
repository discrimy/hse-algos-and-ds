import dataclasses
import math
import random
from pprint import pprint
from typing import Union, Iterator

from matplotlib import pyplot as plt, patches
from sortedcontainers import SortedList


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y)**2)

@dataclasses.dataclass
class Surface:
    top: float
    left: float
    bottom: float
    right: float

    def is_inside(self, point: Point) -> bool:
        return self.top <= point.y <= self.bottom and self.left <= point.x <= self.right

@dataclasses.dataclass
class QuadTreeNode:
    tl: Union['QuadTreeNode', None]
    tr: Union['QuadTreeNode', None]
    bl: Union['QuadTreeNode', None]
    br: Union['QuadTreeNode', None]

    surface: Surface
    points: list[Point] | None



def build_quadtree(points: list[Point], surface: Surface, max_depth: int) -> QuadTreeNode:
    if len(points) < 2 or max_depth == 0:
        return QuadTreeNode(None, None, None, None, surface, points)

    middle_x = (surface.right + surface.left) / 2
    middle_y = (surface.bottom + surface.top) / 2

    tl = []
    tr = []
    bl = []
    br = []
    for point in points:
        if point.y < middle_y:
            if point.x < middle_x:
                tl.append(point)
            else:
                tr.append(point)
        else:
            if point.x < middle_x:
                bl.append(point)
            else:
                br.append(point)

    if tl:
        tl_node = build_quadtree(tl, Surface(surface.top, surface.left, middle_y, middle_x), max_depth - 1)
    else:
        tl_node = None
    if tr:
        tr_node = build_quadtree(tr, Surface(surface.top, middle_x, middle_y, surface.right), max_depth - 1)
    else:
        tr_node = None
    if bl:
        bl_node = build_quadtree(bl, Surface(middle_y, surface.left, surface.bottom, middle_x), max_depth - 1)
    else:
        bl_node = None
    if br:
        br_node = build_quadtree(br, Surface(middle_y, middle_x, surface.bottom, surface.right), max_depth - 1)
    else:
        br_node = None
    node = QuadTreeNode(tl_node, tr_node, bl_node, br_node, surface, points=None)

    return node


def all_nodes(quadtree: QuadTreeNode) -> Iterator[QuadTreeNode]:
    yield quadtree
    if quadtree.tl:
        yield from all_nodes(quadtree.tl)
    if quadtree.tr:
        yield from all_nodes(quadtree.tr)
    if quadtree.bl:
        yield from all_nodes(quadtree.bl)
    if quadtree.br:
        yield from all_nodes(quadtree.br)

def visualize_quadtree(quadtree: QuadTreeNode) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    nodes = list(all_nodes(quadtree))
    for node in nodes:
        x = node.surface.left
        y = quadtree.surface.bottom - node.surface.bottom
        width = node.surface.right - node.surface.left
        height = node.surface.bottom - node.surface.top
        ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))

    x = [point.x for node in nodes if node.points for point in node.points]
    y = [quadtree.surface.bottom - point.y for node in nodes if node.points for point in node.points]
    plt.plot(x, y, 'ro')
    plt.show()

def find_top_n_nearest(quadtree: QuadTreeNode, point: Point, top_n: int) -> list[Point]:
    nodes_stack = []

    node = quadtree
    while node.points is None:
        nodes_stack.append(node)
        if node.tl and node.tl.surface.is_inside(point):
            node = node.tl
        elif node.tr and node.tr.surface.is_inside(point):
            node = node.tr
        elif node.bl and node.bl.surface.is_inside(point):
            node = node.bl
        elif node.br and node.br.surface.is_inside(point):
            node = node.br
        else:
            break

    # pprint([node.surface for node in nodes_stack])
    candidates = set()
    while len(candidates) < top_n and nodes_stack:
        parent = nodes_stack.pop()
        candidates.update({
            point
            for node in all_nodes(parent)
            for point in node.points or []
        })

    # result = sorted(candidates, key=lambda p: point.distance(p))[:top_n]
    # return result
    return candidates

def visualize_quadtree_search(quadtree: QuadTreeNode, point: Point, top_n: int) -> None:
    top_n_points = find_top_n_nearest(quadtree, point, top_n)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    nodes = list(all_nodes(quadtree))
    for node in nodes:
        x = node.surface.left
        y = quadtree.surface.bottom - node.surface.bottom
        width = node.surface.right - node.surface.left
        height = node.surface.bottom - node.surface.top
        ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))

    all_points = [point for node in nodes if node.points for point in node.points]
    x = [point.x for point in all_points]
    y = [quadtree.surface.bottom - point.y for point in all_points]
    plt.plot(x, y, 'ro')

    top_n_x = [point.x for point in top_n_points]
    top_n_y = [quadtree.surface.bottom - point.y for point in top_n_points]
    plt.plot(top_n_x, top_n_y, 'ro', color='lightblue')
    plt.plot(point.x, quadtree.surface.bottom - point.y, 'ro', color='blue')
    plt.show()

surface = Surface(0, 0, 100, 100)
points = [
    Point(x=random.uniform(surface.left, surface.right), y=random.uniform(surface.top, surface.bottom))
    for _ in range(55)
]

quadtree = build_quadtree(points, surface, max_depth=55)
pprint(quadtree)

visualize_quadtree(quadtree)

visualize_quadtree_search(quadtree, Point(56, 89), 10)