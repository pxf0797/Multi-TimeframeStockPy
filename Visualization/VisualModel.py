#!usr/bin/env python3
#coding: utf-8
#VisualModel.py

# 可视化
import matplotlib.pyplot as plt
import networkx as nx

'''
抽取tensorflow.keras模型中的每层信息
'''
def utils_nn_config(model):
    lst_layers = []
    if "Sequential" in str(model): #-> Sequential不显示输入层
        layer = model.layers[0]
        lst_layers.append({"name":"input", "in":int(layer.input.shape[-1]), "neurons":0, 
                           "out":int(layer.input.shape[-1]), "activation":None,
                           "params":0, "bias":0})
    for layer in model.layers:
        try:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]}
        except:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0}
        lst_layers.append(dic_layer)
    return lst_layers
 
 
 
'''
绘制神经网络的草图
visualize_nn(model, description=True, figsize=(10,8))
'''
def visualize_nn(model, description=False, figsize=(10,8)):
    # 获取层次信息
    lst_layers = utils_nn_config(model)
    layer_sizes = [layer["out"] for layer in lst_layers]
    
    # 绘图设置
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    # 中间节点
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes)-1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color
        
        ## 添加信息说明
        if (description is True):
            d = i if i == 0 else i-0.5
            if layer['activation'] is None:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left+d*x_space, y=top-p, fontsize=10, color=color, s=layer['activation']+" (")
                plt.text(x=left+d*x_space, y=top-2*p, fontsize=10, color=color, s="Σ"+str(layer['in'])+"[X*w]+b")
                out = " Y"  if i == len(layer_sizes)-1 else " out"
                plt.text(x=left+d*x_space, y=top-3*p, fontsize=10, color=color, s=") = "+str(layer['neurons'])+out)
        
        ## 遍历
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ## 添加文本说明
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    # 添加链接箭头等
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i+1]
        color = "green" if i == len(layer_sizes)-2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], 
                                  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()
    
def visualize_multitimeframe_lstm(model, figsize=(12, 8)):
    """
    Visualize the MultiTimeframeLSTM model architecture.
    
    :param model: The MultiTimeframeLSTM model instance
    :param figsize: Figure size (width, height) in inches
    """
    def add_node(graph, node_id, label, color, pos):
        graph.add_node(node_id, label=label, color=color, pos=pos)

    def add_edge(graph, start_node, end_node):
        graph.add_edge(start_node, end_node)

    # Create a new graph
    G = nx.DiGraph()

    # Define positions for each layer
    pos = {
        'input': (0, 0),
        'lstm': (1, 0),
        'attention': (2, 0),
        'dynamic_weight': (1.5, 1),
        'dropout': (3, 0),
        'output': (4, 0)
    }

    # Add nodes
    add_node(G, 'input', 'Input', 'lightblue', pos['input'])
    add_node(G, 'lstm', f'LSTM\n{model.lstm.input_size} -> {model.lstm.hidden_size}', 'lightgreen', pos['lstm'])
    add_node(G, 'attention', f'MultiheadAttention\n{model.attention.num_heads} heads', 'lightyellow', pos['attention'])
    add_node(G, 'dynamic_weight', 'DynamicWeight', 'lightpink', pos['dynamic_weight'])
    add_node(G, 'dropout', f'Dropout\n{model.dropout.p}', 'lightgray', pos['dropout'])
    add_node(G, 'output', f'FC\n{model.fc.in_features} -> {model.fc.out_features}', 'lightcoral', pos['output'])

    # Add edges
    add_edge(G, 'input', 'lstm')
    add_edge(G, 'lstm', 'attention')
    add_edge(G, 'lstm', 'dynamic_weight')
    add_edge(G, 'dynamic_weight', 'attention')
    add_edge(G, 'attention', 'dropout')
    add_edge(G, 'dropout', 'output')

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw the graph
    nx.draw(G, pos,
            node_color=[node[1]['color'] for node in G.nodes(data=True)],
            labels={node[0]: node[1]['label'] for node in G.nodes(data=True)},
            with_labels=True,
            node_size=3000,
            node_shape='s',
            ax=ax,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1')

    # Add title
    plt.title("MultiTimeframeLSTM Model Architecture", fontsize=16)

    # Remove axis
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Usage example:
# visualize_multitimeframe_lstm(model)
