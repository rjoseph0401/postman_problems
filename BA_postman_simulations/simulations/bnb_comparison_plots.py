import matplotlib.pyplot as plt

def create_four_part_comparison(results):
    import pandas as pd
    algorithms = results['algorithm'].unique()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))    
    filtered_results = results
    ax = axs[0, 0]
    
    all_nodes_explored_values = sorted(filtered_results['nodes_explored'].unique())
    
    for i, alg in enumerate(algorithms):
        data = filtered_results[filtered_results['algorithm'] == alg]
        if not data.empty:
            grouped = data.groupby('nodes_explored')['runtime'].mean().sort_index()
            ax.plot(grouped.index, grouped.values, label=alg, marker='o', color=colors[i % len(colors)])
            ax.set_xlabel('Nodes Explored')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime vs Nodes Explored')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axs[0, 1]
    for i, alg in enumerate(algorithms):
        data = filtered_results[filtered_results['algorithm'] == alg]
        if not data.empty:
            means = data.groupby('n_edges')['nodes_explored'].mean()
            ax.plot(means.index, means.values, label=alg, marker='o', color=colors[i % len(colors)])
    ax.set_xlabel('Graph Size (edges)')
    ax.set_ylabel('Nodes Explored')
    ax.set_title('Nodes Explored vs Number of Edges')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    for i, alg in enumerate(algorithms):
        data = filtered_results[(filtered_results['algorithm'] == alg) & (filtered_results['success'])]
        if not data.empty:
            means = data.groupby('n_edges')['cost'].mean()
            ax.plot(means.index, means.values, label=alg, marker='o', color=colors[i % len(colors)])
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Average Cost (successful runs)')
    ax.set_title('Average Cost vs Number of Edges')
    ax.legend()
    ax.grid(True, alpha=0.3)    
    ax = axs[1, 1]

    for i, alg in enumerate(algorithms):
        data = filtered_results[filtered_results['algorithm'] == alg]
        if not data.empty:
            means = data.groupby('n_edges')['exploration_rate'].mean()
            ax.plot(means.index, means.values, label=alg, marker='o', color=colors[i % len(colors)])
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Exploration Rate (nodes/sec)')
    ax.set_title('Exploration Rate vs Number of Edges')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    return fig
