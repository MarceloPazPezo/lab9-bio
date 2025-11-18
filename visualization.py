"""
Utilidades de visualización con matplotlib y networkx.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional, Dict, Tuple, List


def draw_graph(graph: nx.Graph, ax: Optional[plt.Axes] = None, 
               title: str = "", node_labels: Optional[Dict] = None,
               edge_labels: Optional[Dict] = None, 
               pos: Optional[Dict] = None) -> plt.Figure:
    """
    Dibuja un grafo usando networkx y matplotlib.
    
    Args:
        graph: Grafo de networkx
        ax: Ejes de matplotlib (si None, crea nueva figura)
        title: Título del gráfico
        node_labels: Diccionario de etiquetas de nodos
        edge_labels: Diccionario de etiquetas de arcos
        pos: Posiciones de los nodos
        
    Returns:
        Figura de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Calcular posiciones si no se proporcionan
    if pos is None:
        pos = nx.spring_layout(graph, k=1.5, iterations=50)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='lightblue',
                          node_size=1000, alpha=0.9)
    
    # Dibujar arcos
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.6, width=2)
    
    # Etiquetas de nodos - usar las proporcionadas o del grafo
    if node_labels is None:
        node_labels = {}
        for node in graph.nodes():
            # Intentar obtener label del nodo
            label = graph.nodes[node].get('label', None)
            if label:
                node_labels[node] = str(label)
            else:
                node_labels[node] = str(node)
    else:
        # Asegurar que todos los nodos tengan label
        for node in graph.nodes():
            if node not in node_labels:
                label = graph.nodes[node].get('label', None)
                node_labels[node] = str(label) if label else str(node)
    nx.draw_networkx_labels(graph, pos, node_labels, ax=ax, font_size=11, font_weight='bold')
    
    # Etiquetas de arcos (pesos)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax, font_size=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def draw_additive_tree(tree: nx.Graph, ax: Optional[plt.Axes] = None,
                       title: str = "", node_labels: Optional[Dict] = None,
                       edge_labels: Optional[Dict] = None) -> plt.Figure:
    """
    Dibuja un árbol aditivo con nodos en línea recta y aristas en forma de V.
    
    Args:
        tree: Árbol de networkx
        ax: Ejes de matplotlib
        title: Título del gráfico
        node_labels: Etiquetas de nodos
        edge_labels: Etiquetas de arcos (distancias)
        
    Returns:
        Figura de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Posicionar nodos en una línea horizontal
    nodes = list(tree.nodes())
    n = len(nodes)
    pos = {}
    
    # Distribuir nodos uniformemente en una línea horizontal (y=0)
    for i, node in enumerate(nodes):
        x = (i - (n-1)/2) * 2  # Centrar en x=0
        pos[node] = (x, 0)
    
    # Para cada arista, crear un punto medio en forma de V
    # Crear un grafo modificado para visualización con nodos intermedios
    display_graph = nx.Graph()
    display_pos = {}
    display_labels = {}
    edge_label_pos = {}
    edge_label_text = {}
    
    # Agregar nodos originales
    for node in nodes:
        display_graph.add_node(node)
        display_pos[node] = pos[node]
        if node_labels and node in node_labels:
            display_labels[node] = node_labels[node]
        else:
            label = tree.nodes[node].get('label', str(node))
            display_labels[node] = str(label)
    
    # Agregar nodos intermedios y aristas en V
    intermediate_node_id = max(nodes) + 1 if nodes else 0
    
    for u, v in tree.edges():
        # Crear nodo intermedio arriba del punto medio
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = 2.0  # Altura de la V
        
        intermediate_node = f"int_{intermediate_node_id}"
        intermediate_node_id += 1
        
        display_graph.add_node(intermediate_node)
        display_pos[intermediate_node] = (mid_x, mid_y)
        display_labels[intermediate_node] = ""  # Sin etiqueta para nodo intermedio
        
        # Agregar aristas desde nodos originales al intermedio
        display_graph.add_edge(u, intermediate_node)
        display_graph.add_edge(v, intermediate_node)
        
        # Etiquetas de distancia
        if edge_labels and (u, v) in edge_labels:
            dist_text = edge_labels[(u, v)]
        elif edge_labels and (v, u) in edge_labels:
            dist_text = edge_labels[(v, u)]
        elif 'distance' in tree[u][v]:
            dist_text = f"{tree[u][v]['distance']:.2f}"
        elif 'weight' in tree[u][v]:
            dist_text = f"{tree[u][v]['weight']:.2f}"
        else:
            dist_text = ""
        
        # Posicionar etiqueta en el punto medio de cada rama de la V
        if dist_text:
            # Etiqueta para rama izquierda (u -> intermedio)
            edge_label_pos[(u, intermediate_node)] = (
                (pos[u][0] + mid_x) / 2,
                (0 + mid_y) / 2
            )
            edge_label_text[(u, intermediate_node)] = dist_text
            
            # Etiqueta para rama derecha (v -> intermedio)
            edge_label_pos[(v, intermediate_node)] = (
                (pos[v][0] + mid_x) / 2,
                (0 + mid_y) / 2
            )
            edge_label_text[(v, intermediate_node)] = dist_text
    
    # Dibujar nodos originales (en la línea)
    original_nodes = [n for n in display_graph.nodes() if not str(n).startswith('int_')]
    nx.draw_networkx_nodes(display_graph, display_pos, nodelist=original_nodes,
                          ax=ax, node_color='lightgreen', node_size=1000, alpha=0.9)
    
    # Dibujar nodos intermedios (puntos pequeños)
    intermediate_nodes = [n for n in display_graph.nodes() if str(n).startswith('int_')]
    if intermediate_nodes:
        nx.draw_networkx_nodes(display_graph, display_pos, nodelist=intermediate_nodes,
                              ax=ax, node_color='gray', node_size=50, alpha=0.5)
    
    # Dibujar aristas
    nx.draw_networkx_edges(display_graph, display_pos, ax=ax, alpha=0.7, width=2, 
                          edge_color='gray')
    
    # Etiquetas de nodos originales
    nx.draw_networkx_labels(display_graph, display_pos, display_labels, ax=ax, 
                           font_size=11, font_weight='bold')
    
    # Etiquetas de distancias
    for edge, pos_label in edge_label_pos.items():
        ax.text(pos_label[0], pos_label[1], edge_label_text[edge],
               fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Ajustar límites para mostrar todo el árbol
    ax.set_xlim(min(p[0] for p in display_pos.values()) - 1, 
                max(p[0] for p in display_pos.values()) + 1)
    ax.set_ylim(-0.5, max(p[1] for p in display_pos.values()) + 1)
    
    plt.tight_layout()
    return fig


def draw_tree(tree: nx.Graph, ax: Optional[plt.Axes] = None,
              title: str = "", node_labels: Optional[Dict] = None,
              edge_labels: Optional[Dict] = None) -> plt.Figure:
    """
    Dibuja un árbol filogenético.
    
    Args:
        tree: Árbol de networkx
        ax: Ejes de matplotlib
        title: Título del gráfico
        node_labels: Etiquetas de nodos
        edge_labels: Etiquetas de arcos (distancias)
        
    Returns:
        Figura de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    ax.clear()
    
    # Usar layout jerárquico para árboles
    try:
        pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
    except:
        # Si graphviz no está disponible, usar spring layout
        pos = nx.spring_layout(tree, k=2, iterations=50)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(tree, pos, ax=ax, node_color='lightgreen',
                          node_size=800, alpha=0.9)
    
    # Dibujar arcos
    nx.draw_networkx_edges(tree, pos, ax=ax, alpha=0.7, width=2, 
                          edge_color='gray')
    
    # Etiquetas de nodos - usar las proporcionadas o del grafo
    if node_labels is None:
        node_labels = {}
        for node in tree.nodes():
            # Intentar obtener label del nodo
            label = tree.nodes[node].get('label', None)
            if label:
                node_labels[node] = str(label)
            else:
                node_labels[node] = str(node)
    else:
        # Asegurar que todos los nodos tengan label
        for node in tree.nodes():
            if node not in node_labels:
                label = tree.nodes[node].get('label', None)
                node_labels[node] = str(label) if label else str(node)
    nx.draw_networkx_labels(tree, pos, node_labels, ax=ax, font_size=11, font_weight='bold')
    
    # Etiquetas de arcos (distancias)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(tree, pos, edge_labels, ax=ax, 
                                    font_size=8, bbox=dict(boxstyle='round,pad=0.3',
                                                          facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def draw_matrix(matrix: np.ndarray, ax: Optional[plt.Axes] = None,
                title: str = "", labels: Optional[List[str]] = None,
                col_labels: Optional[List[str]] = None) -> plt.Figure:
    """
    Dibuja una matriz como tabla.
    
    Args:
        matrix: Matriz numpy (puede contener números o strings)
        ax: Ejes de matplotlib
        title: Título del gráfico
        labels: Etiquetas para filas
        col_labels: Etiquetas para columnas (opcional, usa labels si no se proporciona)
        
    Returns:
        Figura de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.clear()
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    if labels is None:
        labels = [f'Node {i+1}' for i in range(matrix.shape[0])]
    
    if col_labels is None:
        col_labels = labels[:matrix.shape[1]] if matrix.shape[1] <= len(labels) else [f'Col {i+1}' for i in range(matrix.shape[1])]
    
    table_data = []
    for i, row in enumerate(matrix):
        row_data = []
        for val in row:
            # Verificar si es string o número
            if isinstance(val, str):
                row_data.append(val)
            elif isinstance(val, (int, float, np.number)):
                row_data.append(f'{val:.2f}')
            else:
                row_data.append(str(val))
        table_data.append(row_data)
    
    table = ax.table(cellText=table_data, rowLabels=labels[:len(table_data)],
                    colLabels=col_labels[:matrix.shape[1]], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Colorear encabezados correctamente
    # En matplotlib table con rowLabels y colLabels:
    # - (0, 0) = esquina vacía
    # - (0, j) para j>=0 = encabezados de columna
    # - (i, -1) = etiquetas de fila (formato principal)
    # - (i, 0) para i>0 = también puede contener etiquetas de fila
    # - (i, j) para i>0, j>0 = datos
    
    # Obtener todas las celdas disponibles
    cells = table._cells
    
    # Primero, establecer todas las celdas en blanco por defecto
    for key, cell in cells.items():
        row, col = key
        # Solo las celdas de datos (row > 0 y col > 0) deben ser blancas
        if row > 0 and col > 0:
            cell.set_facecolor('white')
            cell.set_text_props(weight='normal', color='black')
    
    # Luego, colorear solo los encabezados
    for key, cell in cells.items():
        row, col = key
        # Encabezados de columna: toda la fila 0
        if row == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        # Etiquetas de fila: columna -1 (formato estándar)
        elif col == -1:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def draw_calculation_text(calculation_texts: List[str], ax: Optional[plt.Axes] = None,
                          title: str = "") -> plt.Figure:
    """
    Dibuja cálculos como texto formateado con ecuaciones.
    
    Args:
        calculation_texts: Lista de strings con los cálculos formateados
        ax: Ejes de matplotlib
        title: Título del gráfico
        
    Returns:
        Figura de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    ax.clear()
    ax.axis('off')
    
    # Calcular espaciado dinámico basado en número de cálculos
    n_calcs = len(calculation_texts)
    if n_calcs == 0:
        return fig
    
    # Ajustar espaciado vertical según cantidad - más espacio
    if n_calcs <= 3:
        y_spacing = 0.22
        y_start = 0.80
        font_size = 10
    elif n_calcs <= 5:
        y_spacing = 0.16
        y_start = 0.85
        font_size = 9
    else:
        y_spacing = 0.12
        y_start = 0.88
        font_size = 8
    
    # Título más compacto
    ax.text(0.5, 0.97, title, ha='center', va='top', 
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Dibujar cada cálculo en una sola línea más compacta
    for idx, calc_text in enumerate(calculation_texts):
        y_pos = y_start - (idx * y_spacing)
        
        # Dividir el texto en partes
        parts = calc_text.split(' → ')
        main_eq = parts[0] if len(parts) > 0 else calc_text
        result = parts[1] if len(parts) > 1 else ""
        
        # Renderizar solo la ecuación principal (sin el máximo en el texto)
        # El máximo se mostrará separado
        equation_text = main_eq
        
        # Renderizar ecuación principal centrada
        ax.text(0.5, y_pos, equation_text, ha='center', va='center',
               fontsize=font_size, fontfamily='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.12', facecolor='#E3F2FD', 
                        edgecolor='#1976D2', linewidth=0.8, alpha=0.7))
        
        # Resaltar el valor máximo a la derecha, separado
        if result:
            # Extraer solo el número del máximo
            max_value = result.replace("Max: ", "").replace("Máximo: ", "").strip()
            # Dibujar el valor máximo resaltado a la derecha
            ax.text(0.95, y_pos, f"→ Máx: {max_value}", ha='right', va='center',
                   fontsize=font_size, fontweight='bold', color='#D32F2F',
                   fontfamily='monospace', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='#FFEBEE', 
                            edgecolor='#D32F2F', linewidth=1, alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_canvas(figure: plt.Figure, parent) -> FigureCanvasTkAgg:
    """
    Crea un canvas de Tkinter desde una figura de matplotlib.
    
    Args:
        figure: Figura de matplotlib
        parent: Widget padre de Tkinter
        
    Returns:
        Canvas de Tkinter
    """
    canvas = FigureCanvasTkAgg(figure, parent)
    return canvas

