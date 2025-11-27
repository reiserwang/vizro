#!/usr/bin/env python3
"""
Causal Network Utilities Module
Handles network visualization and cycle resolution for causal analysis.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np
from causalnex.structure import StructureModel

def create_network_plot(sm, edge_stats, theme, show_all_relationships=False):
    """Create network visualization using Plotly with proper hover interactions"""
    try:
        # Convert to NetworkX graph
        G = nx.DiGraph()

        # Add edges with weights
        edge_dict = {}
        for stat in edge_stats:
            source = stat['source']
            target = stat['target']
            correlation = stat['correlation']
            p_value = stat['p_value']

            # Filter based on significance if not showing all
            if not show_all_relationships and p_value >= 0.05:
                continue

            G.add_edge(source, target, weight=abs(correlation), correlation=correlation, p_value=p_value)
            edge_dict[(source, target)] = stat

        if len(G.edges()) == 0:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No significant relationships found.<br>Try lowering the significance threshold or minimum correlation.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='gray')
            )
            fig.update_layout(
                title="Causal Network (No Relationships Found)",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=600
            )
            return fig

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            # Calculate node statistics
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            node_info.append(f"Variable: {node}<br>Incoming: {in_degree}<br>Outgoing: {out_degree}")

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_info,
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Variables'
        )

        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges():
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Get edge statistics
            correlation = G[source][target]['correlation']
            p_value = G[source][target]['p_value']
            edge_info.append(f"{source} → {target}<br>Correlation: {correlation:.3f}<br>P-value: {p_value:.3f}")

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            name='Relationships'
        )

        # Create arrows for directed edges
        arrow_traces = []
        for edge in G.edges():
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Calculate arrow position (80% along the edge)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)

            # Calculate arrow direction
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length

                # Create arrow
                arrow_trace = go.Scatter(
                    x=[arrow_x],
                    y=[arrow_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='red',
                        angle=np.degrees(np.arctan2(dy, dx))
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                arrow_traces.append(arrow_trace)

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace] + arrow_traces)

        # Update layout
        template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'

        fig.update_layout(
            title=f"Causal Network ({len(G.edges())} relationships)",
            title_font_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Arrows show causal direction. Hover over nodes and edges for details.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=template,
            height=600
        )

        return fig

    except Exception as e:
        # Return error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating network plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

def has_cycles(structure_model):
    """Check if the structure model contains cycles"""
    try:
        # Try to create a NetworkX graph and check for cycles
        import networkx as nx

        # Create directed graph from structure model
        G = nx.DiGraph()
        edges = structure_model.edges()
        G.add_edges_from(edges)

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                print(f"🔍 Found {len(cycles)} cycle(s): {cycles}")
                return True
            return False
        except:
            # If cycle detection fails, assume no cycles
            return False

    except Exception as e:
        print(f"⚠️ Cycle detection failed: {e}")
        return False

def resolve_cycles(structure_model, df_numeric):
    """Attempt to resolve cycles by removing weakest edges"""
    try:
        import networkx as nx
        from causalnex.structure import StructureModel

        # Create NetworkX graph
        G = nx.DiGraph()
        edges = structure_model.edges()

        # Add edges with weights (correlations)
        edge_weights = {}
        for source, target in edges:
            if source in df_numeric.columns and target in df_numeric.columns:
                corr = abs(df_numeric[source].corr(df_numeric[target]))
                edge_weights[(source, target)] = corr
                G.add_edge(source, target, weight=corr)

        # Find and break cycles by removing weakest edges
        cycles = list(nx.simple_cycles(G))
        edges_to_remove = []

        for cycle in cycles:
            if len(cycle) >= 2:
                # Find weakest edge in cycle
                min_weight = float('inf')
                weakest_edge = None

                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]

                    if (source, target) in edge_weights:
                        weight = edge_weights[(source, target)]
                        if weight < min_weight:
                            min_weight = weight
                            weakest_edge = (source, target)

                if weakest_edge and weakest_edge not in edges_to_remove:
                    edges_to_remove.append(weakest_edge)
                    print(f"🔧 Removing weak edge to break cycle: {weakest_edge[0]} -> {weakest_edge[1]} (correlation: {min_weight:.3f})")

        # Create new structure model without cyclic edges
        remaining_edges = [edge for edge in edges if edge not in edges_to_remove]

        # Create new StructureModel
        new_sm = StructureModel()
        new_sm.add_edges_from(remaining_edges)

        print(f"✅ Cycle resolution complete. Removed {len(edges_to_remove)} edges, kept {len(remaining_edges)} edges.")
        return new_sm

    except Exception as e:
        print(f"❌ Cycle resolution failed: {e}")
        # Return original structure model if resolution fails
        return structure_model
