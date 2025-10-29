from graphviz import Digraph

dot = Digraph()

edges = [((2,1), (2,0)), ((2,0), (3,0)), ((2,1), (1,1)), ((1,1), (0,1)), ((0,1), (0,0)), ((0,0), (1,0)), ((0,1), (0,2))]
for parent, child in edges:
    dot.node(str(parent))
    dot.node(str(child))
    dot.edge(str(parent), str(child))

dot.render('dfs_tree', view=True)
