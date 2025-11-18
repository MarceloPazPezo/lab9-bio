"""
Módulo para construcción de árboles aditivos.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional


class AdditiveTreeBuilder:
    """
    Constructor de árboles aditivos paso a paso.
    """
    
    def __init__(self, Mh: np.ndarray, Ml: np.ndarray, labels: Optional[List[str]] = None):
        """
        Inicializa el constructor con las matrices de pesos y distancias.
        
        Args:
            Mh: Matriz de pesos (pesos de los arcos)
            Ml: Matriz de distancias (distancias entre nodos)
            labels: Etiquetas para los nodos (opcional)
        """
        self.Mh = Mh.copy()
        self.Ml = Ml.copy()
        self.n = Mh.shape[0]
        self.cw= Mh.shape[0]
        self.labels = labels if labels else [f'Node {i+1}' for i in range(self.n)]
        self.steps = []
        self.current_step = 0
        
    def build(self) -> List[Dict[str, Any]]:
        """
        Construye el árbol aditivo paso a paso.
        
        Returns:
            Lista de pasos con información para visualización
        """
        self.steps = []
        
        # Paso 1: Grafo inicial desde Mh
        step1 = self._create_initial_graph()
        self.steps.append(step1)
        
        # Paso 2: Spanning tree sin ciclos
        step2 = self._create_spanning_tree()
        self.steps.append(step2)
        
        # Paso 3: Identificar arcos de mayor valor
        step3 = self._identify_max_edges()
        self.steps.append(step3)
        
        # Paso 4: Calcular Cw para cada arco usando Ml
        step4 = self._calculate_cw()
        self.steps.append(step4)
        
        # Paso 5: Crear matriz con resultados de Cw
        step5 = self._create_cw_matrix()
        self.steps.append(step5)
        
        # Paso 6: Crear árbol final
        step6 = self._create_final_tree()
        self.steps.append(step6)
        
        return self.steps
    
    def _create_initial_graph(self) -> Dict[str, Any]:
        """Paso 1: Crear grafo inicial desde matriz Mh."""
        G = nx.Graph()
        
        # Agregar nodos
        for i in range(self.n):
            G.add_node(i, label=self.labels[i])
        
        # Agregar arcos con pesos de Mh
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.Mh[i, j] > 0:
                    G.add_edge(i, j, weight=self.Mh[i, j])
        
        return {
            'step': 1,
            'title': 'Paso 1: Grafo Inicial (Matriz Mh)',
            'graph': G,
            'matrix': self.Mh,
            'description': f'Grafo completo con {G.number_of_nodes()} nodos y {G.number_of_edges()} arcos'
        }
    
    def _create_spanning_tree(self) -> Dict[str, Any]:
        """Paso 2: Generar spanning tree sin ciclos."""
        # Crear grafo completo con pesos de Mh
        G_full = nx.Graph()
        for i in range(self.n):
            G_full.add_node(i, label=self.labels[i])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_full.add_edge(i, j, weight=self.Mh[i, j])
        
        # Usar algoritmo de Kruskal para encontrar spanning tree
        T = nx.minimum_spanning_tree(G_full, weight='weight', algorithm='kruskal')
        # Copiar labels a T
        for node in T.nodes():
            if node in G_full.nodes() and 'label' in G_full.nodes[node]:
                T.nodes[node]['label'] = G_full.nodes[node]['label']
            elif node < len(self.labels):
                T.nodes[node]['label'] = self.labels[node]
        
        return {
            'step': 2,
            'title': 'Paso 2: Spanning Tree sin Ciclos',
            'graph': T,
            'matrix': self.Mh,
            'description': f'Árbol de expansión mínima con {T.number_of_edges()} arcos',
            'edges': list(T.edges(data=True))
        }
    
    def _find_path_edges(self, T, source, target):
        """Encuentra los arcos en el camino entre source y target en el árbol T."""
        try:
            path = nx.shortest_path(T, source, target)
            edges_in_path = []
            for i in range(len(path) - 1):
                edges_in_path.append((path[i], path[i+1]))
            return edges_in_path
        except:
            return []
    
    def _get_max_weight_edge(self, T):
        """Obtiene el arco de mayor peso en el spanning tree."""
        max_weight = -1
        max_edge = None
        for u, v in T.edges():
            weight = T[u][v].get('weight', self.Mh[u, v])
            if weight > max_weight:
                max_weight = weight
                max_edge = (u, v)
        return max_edge, max_weight
    
    def _identify_max_edges(self) -> Dict[str, Any]:
        """Paso 3: Crear matriz con arcos de mayor valor para cada par de nodos."""
        # Obtener spanning tree del paso anterior
        G_full = nx.Graph()
        for i in range(self.n):
            G_full.add_node(i, label=self.labels[i])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_full.add_edge(i, j, weight=self.Mh[i, j])
        
        T = nx.minimum_spanning_tree(G_full, weight='weight', algorithm='kruskal')
        # Copiar labels a T
        for node in T.nodes():
            if node in G_full.nodes() and 'label' in G_full.nodes[node]:
                T.nodes[node]['label'] = G_full.nodes[node]['label']
            elif node < len(self.labels):
                T.nodes[node]['label'] = self.labels[node]
        
        # Encontrar el arco de mayor peso
        max_edge, max_weight = self._get_max_weight_edge(T)
        
        # Crear matriz que muestra qué arco se usa para cada par de nodos
        # Para cada par (i,j), encontrar el camino en el spanning tree
        # y determinar qué arco del camino tiene mayor peso
        path_matrix = np.zeros((self.n, self.n), dtype=object)
        edge_usage = {}  # Para cada arco, qué pares lo usan
        
        for i in range(self.n):
            for j in range(self.n):
                if i >= j:
                    path_matrix[i, j] = '-'
                else:
                    # Encontrar camino entre i y j
                    path_edges = self._find_path_edges(T, i, j)
                    if path_edges:
                        # Encontrar el arco de mayor peso en el camino
                        max_edge_in_path = None
                        max_w = -1
                        for edge in path_edges:
                            u, v = edge
                            w = self.Mh[u, v]
                            if w > max_w:
                                max_w = w
                                max_edge_in_path = edge
                        
                        if max_edge_in_path:
                            u, v = max_edge_in_path
                            # Guardar en formato (uv) usando etiquetas
                            label_u = self.labels[u]
                            label_v = self.labels[v]
                            path_matrix[i, j] = f"({label_u}{label_v})"
                            
                            # Registrar uso del arco
                            edge_key = tuple(sorted(max_edge_in_path))
                            if edge_key not in edge_usage:
                                edge_usage[edge_key] = []
                            edge_usage[edge_key].append((i, j))
                    else:
                        path_matrix[i, j] = '-'
        
        # Ordenar arcos por peso (mayor a menor)
        edges_with_weights = [(u, v, self.Mh[u, v]) for u, v in T.edges()]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        
        description = f"Arco de mayor peso: {self.labels[max_edge[0]]}{self.labels[max_edge[1]]} (peso: {max_weight:.3f})\n"
        description += f"Matriz muestra el arco de mayor peso usado en el camino entre cada par de nodos."
        self.cw=path_matrix
        return {
            'step': 3,
            'title': 'Paso 3: Matriz con Arcos de Mayor Valor',
            'graph': None,  # No mostrar grafo, solo la matriz
            'matrix': path_matrix,
            'path_matrix': path_matrix,  # Matriz con los arcos
            'description': description,
            'max_edge': max_edge,
            'max_weight': max_weight,
            'edges_ordered': edges_with_weights,
            'edge_usage': edge_usage
        }
    
    def _calculate_cw(self) -> Dict[str, Any]:
        """Paso 4: Calcular Cw para cada arco sumando distancias de pares que lo usan."""
        # Obtener spanning tree
        G_full = nx.Graph()
        for i in range(self.n):
            G_full.add_node(i, label=self.labels[i])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_full.add_edge(i, j, weight=self.Ml[i, j])
        
        T = nx.minimum_spanning_tree(G_full, weight='weight', algorithm='kruskal')
        # Copiar labels a T
        for node in T.nodes():
            if node in G_full.nodes() and 'label' in G_full.nodes[node]:
                T.nodes[node]['label'] = G_full.nodes[node]['label']
            elif node < len(self.labels):
                T.nodes[node]['label'] = self.labels[node]
        
        # Para cada arco del spanning tree, encontrar todos los pares de nodos
        # cuyo camino pasa por ese arco, y sumar sus distancias de Ml
        cw_values = {}
        edges_info = []
        edge_pairs = {}  # Para cada arco, lista de pares que lo usan
        
        # Crear matriz de cálculo de Cw
        # Filas: arcos, Columnas: información del cálculo
        max_pairs = 0
        for edge in T.edges():
            u, v = edge
            edge_key = tuple(sorted(edge))
            
            # Encontrar todos los pares de nodos cuyo camino pasa por este arco
            pairs_using_edge = []
            distances_list = []
            cw_sum = 0.0
            
            # Dividir el árbol en dos componentes al remover este arco
            T_copy = T.copy()
            T_copy.remove_edge(u, v)
            
            # Encontrar componentes conectadas
            components = list(nx.connected_components(T_copy))
            if len(components) == 2:
                comp1, comp2 = components
                # Todos los pares entre comp1 y comp2 pasan por el arco (u,v)
                for node1 in comp1:
                    for node2 in comp2:
                        pairs_using_edge.append((node1, node2))
                        dist = self.Ml[node1, node2]
                        distances_list.append(dist)
                        cw_sum += dist
            
            max_pairs = max(max_pairs, len(pairs_using_edge))
            cw_values[edge_key] = cw_sum
            edge_pairs[edge_key] = pairs_using_edge
            
            # Crear información del arco con formato detallado
            label_u = self.labels[u]
            label_v = self.labels[v]
            
            # Formatear pares y distancias
            pairs_labels = [f"{self.labels[p[0]]}{self.labels[p[1]]}" for p in pairs_using_edge]
            pairs_str = '; '.join(pairs_labels)
            distances_str = '{' + ', '.join([f"{d:.3f}" for d in distances_list]) + '}'
            
            # Encontrar el valor máximo de las distancias
            max_dist = max(distances_list) if distances_list else 0.0
            
            edges_info.append({
                'edge': edge,
                'edge_label': f"{label_u}{label_v}",
                'weight_Mh': self.Ml[u, v],
                'Cw': cw_sum,
                'pairs_count': len(pairs_using_edge),
                'pairs': pairs_using_edge,
                'pairs_labels': pairs_labels,
                'distances': distances_list,
                'pairs_str': pairs_str,
                'distances_str': distances_str,
                'max_distance': max_dist
            })
        
        # Crear información de cálculo para renderizado como texto
        # No crear matriz, solo información estructurada
        calculation_texts = []
        
        for info in edges_info:
            # Formato: CW[AG] = {AC; AB; AD} = {1.000, 2.000, 3.000} → Máximo: 3.000
            calc_text = f"CW[{info['edge_label']}] = "
            calc_text += '{' + info['pairs_str'] + '} = '
            calc_text += info['distances_str']
            calc_text += f" → Máximo: {info['max_distance']:.3f}"
            calculation_texts.append(calc_text)
        
        # Crear descripción detallada
        description = f"Cálculo de Cw para {len(edges_info)} arcos:\n\n"
        for info in edges_info:
            description += f"CW[{info['edge_label']}] = "
            description += '{' + info['pairs_str'] + '} = '
            description += info['distances_str']
            description += f" → Máximo: {info['max_distance']:.3f}\n"
        
        return {
            'step': 4,
            'title': 'Paso 4: Cálculo de Cw para cada Arco',
            'graph': None,  # No mostrar grafo, solo el cálculo
            'matrix': None,  # No usar matriz
            'calculation_texts': calculation_texts,  # Textos formateados para renderizar
            'description': description,
            'cw_values': cw_values,
            'edges_info': edges_info,
            'edge_pairs': edge_pairs
        }
    
    def _create_cw_matrix(self) -> Dict[str, Any]:
        """Paso 5: Crear matriz con resultados de Cw."""
        # Obtener spanning tree y valores Cw
        G_full = nx.Graph()
        for i in range(self.n):
            
            G_full.add_node(i, label=self.labels[i])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_full.add_edge(i, j, weight=self.Mh[i, j])
        
        T = nx.minimum_spanning_tree(G_full, weight='weight', algorithm='kruskal')
        # Copiar labels a T
        for node in T.nodes():
            if node in G_full.nodes() and 'label' in G_full.nodes[node]:
                T.nodes[node]['label'] = G_full.nodes[node]['label']
            elif node < len(self.labels):
                T.nodes[node]['label'] = self.labels[node]
        
        # Crear matriz Cw (inicialmente con distancias de Ml)
        cw_matrix = self.Ml.copy()
        
        # Marcar arcos del spanning tree con sus valores Cw
        cw_values = {}
        for u, v in T.edges():
            cw = self.Ml[u, v]
            cw_values[(u, v)] = cw
        
        return {
            'step': 5,
            'title': 'Paso 5: Matriz con Resultados de Cw',
            'graph': T,
            'matrix': cw_matrix,
            'description': 'Matriz de distancias (Ml) que representa los valores Cw',
            'cw_matrix': cw_matrix,
            'cw_values': cw_values
        }
    
    def _create_final_tree(self) -> Dict[str, Any]:
        """Paso 6: Crear árbol final."""
        # Obtener spanning tree
        G_full = nx.Graph()
        for i in range(self.n):
            G_full.add_node(i, label=self.labels[i])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_full.add_edge(i, j, weight=self.Mh[i, j])

        T = nx.minimum_spanning_tree(G_full, weight='weight', algorithm='kruskal')
        # Copiar labels a T
        for node in T.nodes():
            if node in G_full.nodes() and 'label' in G_full.nodes[node]:
                T.nodes[node]['label'] = G_full.nodes[node]['label']
            elif node < len(self.labels):
                T.nodes[node]['label'] = self.labels[node]
        
        # Calcular valores Cw para cada arco (igual que en paso 4)
        cw_values = {}
        for edge in T.edges():
            u, v = edge
            edge_key = tuple(sorted(edge))
            
            # Dividir el árbol en dos componentes al remover este arco
            T_copy = T.copy()
            T_copy.remove_edge(u, v)
            components = list(nx.connected_components(T_copy))
            
            cw_sum = 0.0
            if len(components) == 2:
                comp1, comp2 = components
                for node1 in comp1:
                    for node2 in comp2:
                        cw_sum += self.Ml[node1, node2]
            
            cw_values[edge_key] = cw_sum
        
        # Encontrar el arco con mayor Cw (nodo más lejano)
        max_cw_edge = max(cw_values.items(), key=lambda x: x[1])
        max_cw_value = max_cw_edge[1]
        max_edge = max_cw_edge[0]
        
        # Distribuir valores: dividir el valor máximo entre los dos lados
        edge_distances = {}
        distribution_info = []
        
        for edge in T.edges():
            u, v = edge
            edge_key = tuple(sorted(edge))
            cw = cw_values[edge_key]
            
            # Si es el arco con mayor Cw, dividir por 2
            if edge_key == max_edge:
                dist_u = max_cw_value / 2.0
                dist_v = max_cw_value / 2.0
                edge_distances[edge_key] = (dist_u, dist_v)
                label_u = self.labels[u]
                label_v = self.labels[v]
                distribution_info.append(
                    f"Arco {label_u}{label_v}: Cw={cw:.3f} (máximo) → "
                    f"Dividido: {dist_u:.3f} por lado"
                )
            else:
                # Para otros arcos, usar una proporción del Cw
                dist = cw / 2.0
                edge_distances[edge_key] = (dist, dist)
                label_u = self.labels[u]
                label_v = self.labels[v]
                distribution_info.append(
                    f"Arco {label_u}{label_v}: Cw={cw:.3f} → "
                    f"Distancia: {dist:.3f}"
                )
            
            # Asignar distancia al arco (usar el promedio de las dos distancias)
            avg_dist = sum(edge_distances[edge_key]) / 2.0
            T[u][v]['distance'] = avg_dist
            T[u][v]['weight'] = self.Mh[u, v]
            T[u][v]['cw'] = cw
        
        description = f"Árbol Aditivo Final\n\n"
        description += f"Arco con mayor Cw: {self.labels[max_edge[0]]}{self.labels[max_edge[1]]} "
        description += f"(Cw = {max_cw_value:.3f})\n"
        description += f"Valor dividido: {max_cw_value/2.0:.3f} por cada lado\n\n"
        description += "Distribución de valores:\n"
        for info in distribution_info:
            description += f"  {info}\n"
        
        return {
            'step': 6,
            'title': 'Paso 6: Árbol Aditivo Final',
            'graph': T,
            'tree': T,
            'matrix': None,
            'description': description,
            'cw_values': cw_values,
            'max_cw_edge': max_edge,
            'max_cw_value': max_cw_value,
            'edge_distances': edge_distances
        }
    
    def get_step(self, step_index: int) -> Dict[str, Any]:
        """Obtiene un paso específico."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def get_total_steps(self) -> int:
        """Retorna el número total de pasos."""
        return len(self.steps)

