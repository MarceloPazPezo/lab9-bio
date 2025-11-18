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
        
        description = f"Arco de mayor peso: {self.labels[max_edge[0]]}{self.labels[max_edge[1]]} (peso: {max_weight:.2f})\n"
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
        """Paso 4: Calcular Cw para cada arco usando el máximo de distancias de pares que lo usan."""
        # Obtener la matriz de arcos del paso 3
        if not hasattr(self, 'cw') or self.cw is None:
            # Si no hay matriz del paso 3, retornar vacío
            return {
                'step': 4,
                'title': 'Paso 4: Cálculo de Cw para cada Arco',
                'graph': None,
                'matrix': None,
                'calculation_texts': [],
                'description': 'Error: No se encontró la matriz del paso 3',
                'cw_values': {},
                'edges_info': [],
                'edge_pairs': {}
            }
        
        path_matrix = self.cw
        
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
        
        # Extraer los arcos únicos que aparecen en la matriz del paso 3
        unique_edges = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cell_value = path_matrix[i, j]
                if cell_value != '-' and isinstance(cell_value, str):
                    # Extraer el arco del formato (AB)
                    edge_str = cell_value.strip('()')
                    if edge_str:
                        # Encontrar los índices de los nodos
                        # El formato es (labelU labelV)
                        for u in range(self.n):
                            for v in range(u + 1, self.n):
                                label_u = self.labels[u]
                                label_v = self.labels[v]
                                if edge_str == f"{label_u}{label_v}" or edge_str == f"{label_v}{label_u}":
                                    unique_edges.add(tuple(sorted([u, v])))
        
        # Calcular Cw solo para los arcos únicos
        cw_values = {}
        edges_info = []
        edge_pairs = {}
        
        for edge_key in unique_edges:
            u, v = edge_key
            
            # Encontrar todos los pares de nodos cuyo camino pasa por este arco
            # según la matriz del paso 3
            pairs_using_edge = []
            distances_list = []
            
            # Buscar en la matriz del paso 3 qué pares usan este arco
            label_u = self.labels[u]
            label_v = self.labels[v]
            edge_label = f"{label_u}{label_v}"
            
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    cell_value = path_matrix[i, j]
                    if cell_value == f"({edge_label})" or cell_value == f"({label_v}{label_u})":
                        pairs_using_edge.append((i, j))
                        dist = self.Ml[i, j]
                        distances_list.append(dist)
            
            # Encontrar el valor MÁXIMO de las distancias (no la suma)
            max_dist = max(distances_list) if distances_list else 0.0
            
            cw_values[edge_key] = max_dist
            edge_pairs[edge_key] = pairs_using_edge
            
            # Formatear pares y distancias
            pairs_labels = [f"{self.labels[p[0]]}{self.labels[p[1]]}" for p in pairs_using_edge]
            pairs_str = '; '.join(pairs_labels)
            distances_str = '{' + ', '.join([f"{d:.2f}" for d in distances_list]) + '}'
            
            edges_info.append({
                'edge': (u, v),
                'edge_label': edge_label,
                'weight_Mh': self.Mh[u, v],
                'Cw': max_dist,
                'pairs_count': len(pairs_using_edge),
                'pairs': pairs_using_edge,
                'pairs_labels': pairs_labels,
                'distances': distances_list,
                'pairs_str': pairs_str,
                'distances_str': distances_str,
                'max_distance': max_dist
            })
        
        # Crear información de cálculo para renderizado como texto
        calculation_texts = []
        
        for info in edges_info:
            # Formato: CW[AB] = {AC; AE; BC; BD; BE} = {4.000, 5.000, 3.000, 4.000, 5.000, 4.000} → Max: 5.000
            calc_text = f"CW[{info['edge_label']}] = "
            calc_text += '{' + info['pairs_str'] + '} = '
            calc_text += info['distances_str']
            calc_text += f" → Max: {info['max_distance']:.2f}"
            calculation_texts.append(calc_text)
        
        # Crear descripción detallada
        description = f"Cálculo de Cw para {len(edges_info)} arcos únicos del paso 3:\n\n"
        for info in edges_info:
            description += f"CW[{info['edge_label']}] = "
            description += '{' + info['pairs_str'] + '} = '
            description += info['distances_str']
            description += f" → Max: {info['max_distance']:.2f}\n"
        
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
        """Paso 5: Crear matriz con resultados de Cw mostrando los valores numéricos máximos."""
        # Obtener la matriz de arcos del paso 3 y los valores Cw del paso 4
        if not hasattr(self, 'cw') or self.cw is None:
            # Si no hay matriz del paso 3, retornar matriz vacía
            return {
                'step': 5,
                'title': 'Paso 5: Matriz con Resultados de Cw',
                'graph': None,
                'matrix': np.zeros((self.n, self.n)),
                'description': 'Error: No se encontró la matriz del paso 3',
                'cw_matrix': np.zeros((self.n, self.n)),
                'cw_values': {}
            }
        
        path_matrix = self.cw
        
        # Obtener los valores Cw calculados en el paso 4
        # Necesitamos recalcular o usar los valores del paso anterior
        # Extraer los arcos únicos y sus valores Cw
        unique_edges_cw = {}
        
        # Extraer los arcos únicos que aparecen en la matriz del paso 3
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cell_value = path_matrix[i, j]
                if cell_value != '-' and isinstance(cell_value, str):
                    edge_str = cell_value.strip('()')
                    if edge_str:
                        # Encontrar los índices de los nodos
                        for u in range(self.n):
                            for v in range(u + 1, self.n):
                                label_u = self.labels[u]
                                label_v = self.labels[v]
                                if edge_str == f"{label_u}{label_v}" or edge_str == f"{label_v}{label_u}":
                                    edge_key = tuple(sorted([u, v]))
                                    if edge_key not in unique_edges_cw:
                                        # Calcular Cw para este arco
                                        distances_list = []
                                        
                                        # Buscar qué pares usan este arco
                                        for ii in range(self.n):
                                            for jj in range(ii + 1, self.n):
                                                cell_val = path_matrix[ii, jj]
                                                if cell_val == f"({label_u}{label_v})" or cell_val == f"({label_v}{label_u})":
                                                    dist = self.Ml[ii, jj]
                                                    distances_list.append(dist)
                                        
                                        max_dist = max(distances_list) if distances_list else 0.0
                                        unique_edges_cw[edge_key] = max_dist
        
        # Crear matriz resultado con los valores numéricos de Cw
        cw_result_matrix = np.zeros((self.n, self.n), dtype=object)
        
        for i in range(self.n):
            for j in range(self.n):
                if i >= j:
                    cw_result_matrix[i, j] = '-'
                else:
                    # Obtener el arco que se usa para este par
                    cell_value = path_matrix[i, j]
                    if cell_value != '-' and isinstance(cell_value, str):
                        edge_str = cell_value.strip('()')
                        if edge_str:
                            # Buscar el valor Cw de este arco
                            for edge_key, cw_value in unique_edges_cw.items():
                                u, v = edge_key
                                label_u = self.labels[u]
                                label_v = self.labels[v]
                                if edge_str == f"{label_u}{label_v}" or edge_str == f"{label_v}{label_u}":
                                    # Mostrar el valor numérico de Cw
                                    cw_result_matrix[i, j] = cw_value
                                    break
                            else:
                                cw_result_matrix[i, j] = '-'
                        else:
                            cw_result_matrix[i, j] = '-'
                    else:
                        cw_result_matrix[i, j] = '-'
        
        # Crear descripción con los valores Cw
        description = "Matriz con valores Cw (máximos) para cada par de nodos:\n\n"
        description += "Valores Cw calculados por arco:\n"
        for edge_key, cw_value in sorted(unique_edges_cw.items(), key=lambda x: x[1], reverse=True):
            u, v = edge_key
            label_u = self.labels[u]
            label_v = self.labels[v]
            description += f"  ({label_u}{label_v}): Cw = {cw_value:.2f}\n"
        
        return {
            'step': 5,
            'title': 'Paso 5: Matriz con Resultados de Cw',
            'graph': None,
            'matrix': cw_result_matrix,
            'description': description,
            'cw_matrix': cw_result_matrix,
            'cw_values': unique_edges_cw,
            'cw_result_matrix': cw_result_matrix  # Matriz con valores numéricos
        }
    
    def _create_final_tree(self) -> Dict[str, Any]:
        """Paso 6: Crear árbol final usando valores máximos de Cw del paso 5."""
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
        
        # Obtener valores Cw MÁXIMOS del paso 5 (no suma)
        if not hasattr(self, 'cw') or self.cw is None:
            return {
                'step': 6,
                'title': 'Paso 6: Árbol Aditivo Final',
                'graph': T,
                'tree': T,
                'matrix': None,
                'description': 'Error: No se encontró información del paso 3',
                'cw_values': {},
                'max_cw_edge': None,
                'max_cw_value': 0,
                'edge_distances': {}
            }
        
        path_matrix = self.cw
        
        # Calcular valores Cw MÁXIMOS para cada arco (igual que en paso 4 y 5)
        cw_values = {}
        unique_edges_cw = {}
        
        # Extraer los arcos únicos que aparecen en la matriz del paso 3
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cell_value = path_matrix[i, j]
                if cell_value != '-' and isinstance(cell_value, str):
                    edge_str = cell_value.strip('()')
                    if edge_str:
                        # Encontrar los índices de los nodos
                        for u in range(self.n):
                            for v in range(u + 1, self.n):
                                label_u = self.labels[u]
                                label_v = self.labels[v]
                                if edge_str == f"{label_u}{label_v}" or edge_str == f"{label_v}{label_u}":
                                    edge_key = tuple(sorted([u, v]))
                                    if edge_key not in unique_edges_cw:
                                        # Calcular Cw MÁXIMO para este arco
                                        distances_list = []
                                        
                                        # Buscar qué pares usan este arco
                                        for ii in range(self.n):
                                            for jj in range(ii + 1, self.n):
                                                cell_val = path_matrix[ii, jj]
                                                if cell_val == f"({label_u}{label_v})" or cell_val == f"({label_v}{label_u})":
                                                    dist = self.Ml[ii, jj]
                                                    distances_list.append(dist)
                                        
                                        max_dist = max(distances_list) if distances_list else 0.0
                                        unique_edges_cw[edge_key] = max_dist
        
        cw_values = unique_edges_cw
        
        # Encontrar el arco con mayor Cw (valor máximo)
        if cw_values:
            max_cw_edge = max(cw_values.items(), key=lambda x: x[1])
            max_cw_value = max_cw_edge[1]
            max_edge = max_cw_edge[0]
        else:
            max_edge = None
            max_cw_value = 0
        
        # Distribuir valores: usar los valores máximos de Cw
        edge_distances = {}
        distribution_info = []
        
        for edge_key, cw_max in cw_values.items():
            u, v = edge_key
            
            # Usar el valor máximo de Cw directamente, dividido por 2 para cada lado
            dist_u = cw_max / 2.0
            dist_v = cw_max / 2.0
            edge_distances[edge_key] = (dist_u, dist_v)
            
            label_u = self.labels[u]
            label_v = self.labels[v]
            
            if edge_key == max_edge:
                distribution_info.append(
                    f"Arco {label_u}{label_v}: Cw={cw_max:.2f} (máximo) → "
                    f"Distancia por lado: {dist_u:.2f}"
                )
            else:
                distribution_info.append(
                    f"Arco {label_u}{label_v}: Cw={cw_max:.2f} → "
                    f"Distancia por lado: {dist_u:.2f}"
                )
            
            # Asignar distancia al arco en el grafo
            if T.has_edge(u, v):
                T[u][v]['distance'] = dist_u  # Usar dist_u como la distancia del arco
                T[u][v]['weight'] = self.Mh[u, v]
                T[u][v]['cw'] = cw_max
        
        description = "Árbol Aditivo Final\n\n"
        if max_edge:
            description += f"Arco con mayor Cw: {self.labels[max_edge[0]]}{self.labels[max_edge[1]]} "
            description += f"(Cw = {max_cw_value:.2f})\n"
            description += f"Distancia por lado: {max_cw_value/2.0:.2f}\n\n"
        description += "Distribución de valores (usando máximos de Cw):\n"
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
            'edge_distances': edge_distances,
            'is_additive_tree': True  # Indicador para usar visualización especial
        }
    
    def get_step(self, step_index: int) -> Dict[str, Any]:
        """Obtiene un paso específico."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def get_total_steps(self) -> int:
        """Retorna el número total de pasos."""
        return len(self.steps)

