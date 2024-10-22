import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import random
from torchviz import make_dot

# Define a few different types of models
class SmallFCNN(nn.Module):
    def __init__(self):
        super(SmallFCNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MediumCNN(nn.Module):
    def __init__(self):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LargeDeepCNN(nn.Module):
    def __init__(self):
        super(LargeDeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        self.fc = nn.Linear(20, 1)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 20)
        x, _ = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return x

# Visualization function
def torch_model_visualize(model, title="Model Visualization"):
    def extract_layers_from_model(model):
        layers = []
        for name, module in model.named_modules():
            if len(name) > 0:
                layer_type = str(type(module)).split('.')[-1].replace("'>", "")
                layers.append((name, layer_type, module))
        return layers

    def limit_nodes_display(layer_name, num_nodes, max_display=5):
        if num_nodes > max_display:
            return [
                f"{layer_name}_1", f"{layer_name}_2", "...",
                f"{layer_name}_{num_nodes-1}", f"{layer_name}_{num_nodes}"
            ]
        else:
            return [f"{layer_name}_{i+1}" for i in range(num_nodes)]

    def get_layer_description(layer_type, module):
        if layer_type == "Conv2d":
            return f"Conv2D\nIn: {module.in_channels}, Out: {module.out_channels}\nKernel: {module.kernel_size}"
        elif layer_type == "Linear":
            return f"Linear\nIn: {module.in_features}, Out: {module.out_features}"
        elif layer_type == "Dropout":
            return f"Dropout\nP: {module.p}"
        elif layer_type == "RNN":
            return f"RNN\nIn: {module.input_size}, Out: {module.hidden_size}\nLayers: {module.num_layers}"
        else:
            return layer_type

    def determine_layer_role(idx, total_layers):
        if idx == 0:
            return "Input Layer"
        elif idx == total_layers - 1:
            return "Output Layer"
        else:
            return "Hidden Layer"

    def generate_graph(layers, aggregate_nodes=True):
        G = nx.DiGraph()
        pos = {}
        x_offset = 0

        previous_layer = None
        for idx, (layer_name, layer_type, module) in enumerate(layers):
            layer_role = determine_layer_role(idx, len(layers))
            if aggregate_nodes:
                node_name = f"{layer_name}"
                description = get_layer_description(layer_type, module)
                G.add_node(node_name, label=description, layer_type=layer_type)
                pos[node_name] = (x_offset, 0)
                current_layer_nodes = [node_name]
            else:
                num_nodes = random.randint(3, 6) if not hasattr(module, 'out_channels') else module.out_channels
                display_nodes = limit_nodes_display(layer_name, num_nodes)
                y_offset = -(len(display_nodes) - 1) / 2
                current_layer_nodes = []
                for idx, display_name in enumerate(display_nodes):
                    label = display_name
                    G.add_node(display_name, label=label, layer_type=layer_type)
                    pos[display_name] = (x_offset, y_offset + idx)
                    current_layer_nodes.append(display_name)
                if previous_layer:
                    for prev in previous_layer:
                        for curr in current_layer_nodes:
                            if "..." not in prev and "..." not in curr:
                                G.add_edge(prev, curr)

            # Adjust position for annotation above nodes
            pos_x, pos_y = pos[current_layer_nodes[0]]
            ax.text(pos_x, pos_y + 1.5, f"{layer_role}\n{get_layer_description(layer_type, module)}", 
                    fontsize=8, ha='center', va='bottom', color="black")

            previous_layer = current_layer_nodes
            x_offset += 2

        return G, pos

    layers = extract_layers_from_model(model)
    fig, ax = plt.subplots(figsize=(16, 9))
    G, pos = generate_graph(layers, aggregate_nodes=False)

    # Define color map
    layer_color_map = {
        "Conv2d": "skyblue",
        "Linear": "lightgreen",
        "Dropout": "lightcoral",
        "MaxPool2d": "orange",
        "RNN": "violet",
        "Default": "lightgrey"
    }

    node_colors = []
    for node in G.nodes(data=True):
        layer_type = node[1].get('layer_type', "Default")
        color = layer_color_map.get(layer_type, layer_color_map["Default"])
        node_colors.append(color)

    nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color='blue',
            node_size=2000, font_size=10, font_color="white", ax=ax)

    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="black", ax=ax)

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Instantiate models
    models = [
        (SmallFCNN(), "Small Fully Connected NN"),
        (MediumCNN(), "Medium CNN"),
        (LargeDeepCNN(), "Large Deep CNN"),
        (SimpleRNN(), "Simple RNN")
    ]

    # Visualize each model
    for model, title in models:
        torch_model_visualize(model, title)
