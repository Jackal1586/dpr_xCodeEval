from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class NodeData:
    sub_id: int

class Node:
    children: Dict[str, Node]
    n_endpoints: int
    data: List[NodeData]

    def __init__(self) -> None:
        self.children = dict()
        self.n_endpoints = 0
        self.data = list()

    def update_endpoint(self, data) -> None:
        self.n_endpoints += 1
        self.data.append(data)

class TrieTree:
    root: Node

    def __init__(self) -> None:
        self.root = Node()

    def traversal(self, path_str: str, child_not_found_fn: Optional[Callable[[Node, str], Any]]) -> Optional[Node]:
        cur_node = self.root

        for char in path_str:
            if cur_node is None:
                break

            if char not in cur_node.children and child_not_found_fn is not None:
                child_not_found_fn(cur_node, char)
            
            cur_node = cur_node.children.get(char)

        return cur_node

    def insert(self, insert_str: str, data: NodeData) -> None:
        def child_not_found_fn(node: Node, char: str) -> None:
            node.children[char] = Node()

        cur_node = self.traversal(insert_str, child_not_found_fn)
        
        cur_node.update_endpoint(data)

    def search(self, search_str: str) -> Optional[List[NodeData]]:
        cur_node = self.traversal(search_str, None)

        return cur_node if cur_node is None else cur_node.data