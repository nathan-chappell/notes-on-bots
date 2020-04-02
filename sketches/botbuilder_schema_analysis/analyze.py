# analyze.py

# analyze the schema_models.json file and output a graph based on textual
# relationship

import re
import json
from pprint import pprint as pp
from typing import Callable, Dict, List, Any, Tuple, Optional, Set
from typing_extensions import Literal

indegree_key = 'indegree'

Json = Dict[str,Any]
Attributes = Dict[str,Any]
NodeID = str
GraphID = str
Neighborhood = Set[NodeID]

class Node:
  N: Neighborhood
  attrs: Attributes
  graph: 'Graph'

  def __init__(self, graph: 'Graph'):
    self.N = set()
    self.attrs = {}
    self.graph = graph

class UID:
  _uid: int
  def __init__(self):
    self._uid = 0

  def get(self) -> int:
    self._uid += 1
    return self._uid

class Graph:
  _uid: UID = UID()
  nodes: Dict[NodeID,Node]
  attrs: Attributes

  def __init__(self):
    self.nodes = {}
    self.attrs = {'uid':Graph._uid.get()}

  @property
  def name(self) -> str:
    if 'name' in self.attrs:
      return self.attrs['name']
    else: return str(self.attrs['uid'])

  @name.setter
  def name(self, value: str) -> None:
    self.attrs['name'] = value

  def has_node(self,nodeID: NodeID) -> bool:
    return nodeID in self.nodes

  def assert_node(self, nodeID: NodeID, should_exist=True):
    if should_exist and not self.has_node(nodeID):
      raise KeyError(f'{nodeID} already in graph {self.name}')
    if not should_exist and self.has_node(nodeID):
      raise KeyError(f'{nodeID} not in graph {self.name}')

  def add_node(self, nodeID: NodeID, node: Node):
    self.assert_node(nodeID, should_exist=False)
    self.nodes[nodeID] = node

  def pop_node(self, nodeID: NodeID):
    self.assert_node(nodeID)
    return self.attrs.pop(nodeID)

def calculate_in_degree(graph: Graph) -> None:
  for node in graph.nodes:
    graph.nodes[node].attrs[indegree_key] = 0
  for node in graph.nodes:
    for neighbour in graph.nodes[node].N:
      graph.nodes[neighbour].attrs[indegree_key] += 1

def add_dot_attr(node: Node, k:str, v:str):
  if 'dot_attrs' not in node.attrs:
    node.attrs['dot_attrs'] = {}
  node.attrs['dot_attrs'][k] = v


class JsonGraph(Graph):
  filename: str
  stem_min_count: int

  json_text: str
  data: Json
  stems: List[str]

  def __init__(self, filename: str, stem_min_count: int = 5):
    super().__init__()
    self.filename = filename
    self.stem_min_count = stem_min_count
    self.read()
    self.calc_stems()
    self.build_graph()
    calculate_in_degree(self)

  def read(self) -> None:
    with open(self.filename) as f:
      self.json_text = f.read()
      self.data = json.loads(self.json_text)

  def calc_stems(self) -> None:
    stems = []
    for k in self.data.keys():
      stems += re.findall(r'[A-Z][a-z]*',k)
    self.stems = list(set(filter(
      lambda s: self.json_text.count(s) > self.stem_min_count,
      stems)))

  def build_graph(self) -> None:
    node: Node
    # stem nodes and neighbourhood
    for s in self.stems:
      node = Node(self)
      # edges
      node.N = set([k for k in self.data if s in k])
      add_dot_attr(node,'shape','diamond')
      self.add_node(s,node)

    # key nodes
    for k in self.data:
      if self.has_node(k): # node may have been a stem
        node = self.nodes[k]
      else:
        node = Node(self)
        self.add_node(k,node)
      add_dot_attr(node,'shape','box')
      # edges
      for field,ftype in self.data[k].items():
        if type(ftype) == type([]):
          ftype = ftype[0]
        if ftype in self.data:
          node.N.add(ftype)

NodePredicate = Callable[[NodeID,Graph],bool]

def indegree_predicate(degree: int) -> NodePredicate:
  def p(nodeID: NodeID, graph: Graph):
    graph.assert_node(nodeID)
    attrs = graph.nodes[nodeID].attrs
    if indegree_key not in attrs:
      raise KeyError(f'{indegree_key} must be in {nodeID}.attrs')
    return attrs[indegree_key] >= degree
  return p

def stem_predicate(stem: str) -> NodePredicate:
  def p(nodeID: NodeID, graph: Graph):
    graph.assert_node(nodeID)
    return stem in nodeID
  return p

class Subgraph:
  name: str
  graph: Graph
  nodes: List[NodeID]
  predicate: NodePredicate
  attrs: Attributes
  cluster: bool

  def __init__(
      self,
      name: str,
      graph: Graph,
      predicate: NodePredicate,
      node_attrs: Attributes = {},
      cluster: bool = True
    ):
    self.name = name
    self.graph = graph
    self.predicate = predicate #type: ignore
    self.nodes = [nodeID for nodeID in graph.nodes 
                         if predicate(nodeID,graph)]
    self.node_attrs = node_attrs
    self.cluster = cluster

class DotFormatter:

  @staticmethod
  def fattribute(k: str, v:str) -> str:
    return f'{k}={v}'

  @staticmethod
  def fattributes(attrs: Attributes) -> str:
    attr_list: List[str] = [DotFormatter.fattribute(k,v) 
                            for k,v in attrs.items()]
    return '[' + ', '.join(attr_list) + ']'

  @staticmethod
  def fnode(nodeID: NodeID, graph: Graph):
    return nodeID + ' ' + DotFormatter.fattributes(
              graph.nodes[nodeID].attrs.get('dot_attrs',{}))

  @staticmethod
  def fsubgraph(subgraph: Subgraph, indent=2):
    br1 = '\n' + ' '*indent
    br2 = '\n' + ' '*(indent+2)
    name: str = subgraph.name
    if subgraph.cluster: name = 'cluster_' + name
    open_subgraph = br1 + 'subgraph ' + name + ' {' + br2
    subgraph_attr_line = 'label=' + subgraph.name + br2
    node_attr_line = 'node ' + DotFormatter.fattributes(subgraph.node_attrs)
    nodes_str = br2.join(
        DotFormatter.fnode(nodeID,subgraph.graph)
        for nodeID in subgraph.nodes)
    close_subgraph = br1 + '}' + br1
    return open_subgraph + subgraph_attr_line + node_attr_line + br2 + nodes_str + close_subgraph

  @staticmethod
  def fedge(l: NodeID, r: NodeID) -> str:
    return l + ' -> ' + r

  @staticmethod
  def fedges(graph: Graph, indent=2) -> str:
    br = '\n' + ' '*indent
    return br.join(
            br.join(
              [f'//{l}'] + 
              list(DotFormatter.fedge(l,r) for r in graph.nodes[l].N)
            ) for l in graph.nodes)

  @staticmethod
  def fgraph(graph: Graph, subgraphs: List[Subgraph] = []) -> str:
    br = '\n' + ' '*2
    open_graph = "strict digraph " + graph.name + " {" + br
    graph_attr_line = "rankdir=LR" + br
    subgraphs_str = br.join(DotFormatter.fsubgraph(subgraph)
                            for subgraph in subgraphs)
    edges = DotFormatter.fedges(graph)
    close_graph= "\n}"
    return open_graph + graph_attr_line + subgraphs_str + edges + close_graph

if __name__ == '__main__':

  stem_node_attrs = {
      'style':'filled',
      'fillcolor':'gainsboro',
  }
  indegree_node_attrs = {
      'style':'filled',
      'fillcolor':'darkkhaki',
  }

  subgraph_terms = [
    'Card',
    'Response',
    'Conversation',
  ]

  jsonGraph: JsonGraph = JsonGraph('schema_models.json')
  subgraphs: List[Subgraph] = [ 
      Subgraph('highIndegree', jsonGraph, indegree_predicate(5), 
                  indegree_node_attrs, cluster=False)] +  [
      Subgraph(term, jsonGraph, stem_predicate(term), stem_node_attrs)
      for term in subgraph_terms
    ]

  print(DotFormatter.fgraph(jsonGraph, subgraphs))
