import pickle

# Load the learned tree
with open("Meat learned_tree.pkl", "rb") as f:
    Tree = pickle.load(f)

# Retrieve the bestmodel from each node (if available)
best_models = {}
for node_idx, node in Tree.items():
    if hasattr(node, 'bestmodel'):
        best_models[node_idx] = node.bestmodel
        print(f"Node {node_idx} best model: {node.bestmodel}")
    else:
        print(f"Node {node_idx} does not have a bestmodel attribute.")