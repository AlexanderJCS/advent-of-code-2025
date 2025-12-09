from dataclasses import dataclass
import math


@dataclass
class GraphNode:
    id: int
    connections: list["GraphNode"]


class Graph:
    def __init__(self, n):
        self.nodes: list[GraphNode] = [GraphNode(i, []) for i in range(n)]
    
    def add_conn(self, a, b):
        self.nodes[a].connections.append(self.nodes[b])
        self.nodes[b].connections.append(self.nodes[a])
    
    def _search(self, node: GraphNode, found: set[int] = None) -> set[int]:
        if found is None:
            found = {node.id}
        
        for subnode in node.connections:
            if subnode.id in found:
                continue
            
            found.add(subnode.id)
            self._search(subnode, found)
        
        return found
    
    def circuit_sizes(self):
        visited = set()
        
        circuits = []
        for node in self.nodes:
            if node.id in visited:
                continue
            
            searched = self._search(node)
            circuits.append(len(searched))
            visited.update(searched)
        
        return circuits


def main():
    with open("input.txt") as f:
        coords_txt = f.read().splitlines()
    
    coords = []
    for coord_txt in coords_txt:
        coords.append(tuple(map(int, coord_txt.split(","))))
    
    g = Graph(len(coords))
    
    distances = []
    for i, coord1 in enumerate(coords[:-1]):
        for j, coord2 in enumerate(coords[i + 1:]):
            idx2 = j + i + 1
            dist = math.dist(coord1, coord2)
            distances.append((dist, i, idx2))
    
    distances.sort(key=lambda x: x[0])
    
    total_connections = 1000
    for dist in distances[:total_connections]:
        g.add_conn(dist[1], dist[2])
    
    part_1 = math.prod(sorted(g.circuit_sizes(), reverse=True)[:3])
    print(f"{part_1=}")
    
    for dist in distances:
        g.add_conn(dist[1], dist[2])
        
        if len(g.circuit_sizes()) == 1:
            part_2 = coords[dist[1]][0] * coords[dist[2]][0]
            print(f"{part_2=}")
            break
    

if __name__ == "__main__":
    main()
