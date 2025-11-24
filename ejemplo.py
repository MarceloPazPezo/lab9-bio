"""
Módulo para construcción de árboles aditivos.
Algoritmo que toma una matriz de distancias y construye el árbol nodo por nodo,
resolviendo sistemas de ecuaciones para cada nodo interno.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional


class AdditiveTreeBuilder:
    """
    Constructor de árboles aditivos paso a paso.
    Dada una matriz de distancias, construye el árbol incrementalmente,
    resolviendo sistemas de ecuaciones para cada nodo interno.
    """
    
    def __init__(self, distance_matrix: np.ndarray, labels: Optional[List[str]] = None):
        """
        Inicializa el constructor con la matriz de distancias.
        
        Args:
            distance_matrix: Matriz de distancias entre nodos
            labels: Etiquetas para los nodos (opcional)
        """
        self.distance_matrix = distance_matrix.copy()
        self.n = distance_matrix.shape[0]
        self.labels = labels if labels else [f'Node {i+1}' for i in range(self.n)]
        self.steps = []
        self.tree = nx.Graph()
        self.node_counter = self.n
        
    def build(self) -> List[Dict[str, Any]]:
        """
        Construye el árbol aditivo paso a paso.
        
        Returns:
            Lista de pasos con información para visualización
        """
        self.steps = []
        self.tree = nx.Graph()
        
        # Paso 0: Mostrar matriz inicial
        step0 = {
            'step': 0,
            'title': 'Paso 0: Matriz de Distancias Inicial',
            'matrix': self.distance_matrix.copy(),
            'description': f'Matriz de distancias inicial con {self.n} nodos',
            'tree': self.tree.copy()
        }
        self.steps.append(step0)
        
        if self.n < 2:
            return self.steps
        
        # Paso 1: Inicializar con los primeros dos nodos
        self.tree.add_node(0, label=self.labels[0], is_leaf=True)
        self.tree.add_node(1, label=self.labels[1], is_leaf=True)
        distance_01 = self.distance_matrix[0, 1]
        self.tree.add_edge(0, 1, distance=distance_01)
        
        step1 = {
            'step': 1,
            'title': f'Paso 1: Inicialización - Conectar {self.labels[0]} y {self.labels[1]}',
            'matrix': self.distance_matrix.copy(),
            'tree': self.tree.copy(),
            'description': f'Se conectan los primeros dos nodos: {self.labels[0]} y {self.labels[1]} con distancia {distance_01:.2f}',
            'equation_info': {
                'nodes': [0, 1],
                'distance': distance_01,
                'equation': f'd({self.labels[0]}, {self.labels[1]}) = {distance_01:.2f}'
            }
        }
        self.steps.append(step1)
        
        # Pasos 2 a n-1: Agregar cada nodo restante
        for node_idx in range(2, self.n):
            step = self._add_node_to_tree(node_idx)
            self.steps.append(step)
        
        # Paso final: Árbol completo
        final_step = {
            'step': len(self.steps),
            'title': 'Paso Final: Árbol Aditivo Completo',
            'matrix': self.distance_matrix.copy(),
            'tree': self.tree.copy(),
            'description': f'Árbol aditivo completo con {self.n} nodos construido',
            'is_additive_tree': True
        }
        self.steps.append(final_step)
        
        return self.steps
    
    def _add_node_to_tree(self, node_idx: int) -> Dict[str, Any]:
        """
        Agrega un nuevo nodo al árbol existente, resolviendo un sistema de ecuaciones
        para determinar dónde insertarlo y las distancias.
        
        Args:
            node_idx: Índice del nodo a agregar
            
        Returns:
            Diccionario con información del paso
        """
        new_node = node_idx
        self.tree.add_node(new_node, label=self.labels[new_node], is_leaf=True)
        
        # Encontrar el mejor lugar para insertar el nodo
        # Esto se hace encontrando el arco donde insertar el nodo interno
        best_edge, internal_node, distances = self._find_best_insertion(new_node)
        
        if best_edge is None:
            # Si no se puede insertar en un arco, conectar directamente a un nodo existente
            # Encontrar el nodo más cercano
            min_dist = np.inf
            closest_node = 0
            for existing_node in self.tree.nodes():
                if existing_node != new_node:
                    dist = self.distance_matrix[new_node, existing_node]
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = existing_node
            
            self.tree.add_edge(new_node, closest_node, distance=min_dist)
            
            return {
                'step': node_idx,
                'title': f'Paso {node_idx}: Agregar {self.labels[new_node]}',
                'matrix': self.distance_matrix.copy(),
                'tree': self.tree.copy(),
                'description': f'Se conecta {self.labels[new_node]} directamente a {self.labels[closest_node]} con distancia {min_dist:.2f}',
                'equation_info': {
                    'nodes': [new_node, closest_node],
                    'distance': min_dist,
                    'equation': f'd({self.labels[new_node]}, {self.labels[closest_node]}) = {min_dist:.2f}'
                }
            }
        
        # Insertar nodo interno en el arco
        u, v = best_edge
        self.tree.remove_edge(u, v)
        
        # Crear nodo interno
        internal_node_id = self.node_counter
        self.node_counter += 1
        self.tree.add_node(internal_node_id, label=f'U{internal_node_id}', is_leaf=False)
        
        # Conectar nodos
        dist_u_to_internal = distances[0]
        dist_v_to_internal = distances[1]
        dist_new_to_internal = distances[2]
        
        self.tree.add_edge(u, internal_node_id, distance=dist_u_to_internal)
        self.tree.add_edge(v, internal_node_id, distance=dist_v_to_internal)
        self.tree.add_edge(new_node, internal_node_id, distance=dist_new_to_internal)
        
        # Crear información de ecuaciones
        equations = self._format_equations(u, v, new_node, internal_node_id, distances)
        
        return {
            'step': node_idx,
            'title': f'Paso {node_idx}: Agregar {self.labels[new_node]} (Sistema de Ecuaciones)',
            'matrix': self.distance_matrix.copy(),
            'tree': self.tree.copy(),
            'description': f'Se inserta {self.labels[new_node]} en el arco ({self.labels[u]}, {self.labels[v]}) resolviendo un sistema de ecuaciones',
            'equation_info': {
                'nodes': [u, v, new_node, internal_node_id],
                'distances': distances,
                'equations': equations,
                'system_solved': True
            },
            'internal_node': internal_node_id
        }
    
    def _find_best_insertion(self, new_node: int) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[List[float]]]:
        """
        Encuentra el mejor arco donde insertar el nuevo nodo.
        Resuelve un sistema de ecuaciones para cada arco posible.
        
        Args:
            new_node: Índice del nodo a insertar
            
        Returns:
            Tupla con (mejor_arco, nodo_interno, distancias) o (None, None, None) si no se puede insertar
        """
        best_edge = None
        best_internal = None
        best_distances = None
        best_error = np.inf
        
        # Probar insertar en cada arco del árbol
        for edge in self.tree.edges():
            u, v = edge
            distances, error = self._solve_equations_for_edge(new_node, u, v)
            
            if distances is not None and error < best_error:
                # Verificar que las distancias sean positivas
                if all(d > 0 for d in distances):
                    best_error = error
                    best_edge = (u, v)
                    best_distances = distances
        
        if best_edge is None:
            return None, None, None
        
        # Crear nodo interno temporal para el cálculo
        internal_node = self.node_counter
        
        return best_edge, internal_node, best_distances
    
    def _solve_equations_for_edge(self, new_node: int, u: int, v: int) -> Tuple[Optional[List[float]], float]:
        """
        Resuelve el sistema de ecuaciones para insertar new_node en el arco (u, v).
        
        Sistema de ecuaciones:
        d(u, new) = d(u, internal) + d(internal, new)
        d(v, new) = d(v, internal) + d(internal, new)
        d(u, v) = d(u, internal) + d(v, internal)
        
        Donde:
        - d(u, internal) = x
        - d(v, internal) = y
        - d(internal, new) = z
        
        Entonces:
        x + z = d(u, new)
        y + z = d(v, new)
        x + y = d(u, v)
        
        Resolviendo:
        z = (d(u, new) + d(v, new) - d(u, v)) / 2
        x = d(u, new) - z
        y = d(v, new) - z
        
        Args:
            new_node: Nodo a insertar
            u: Primer nodo del arco
            v: Segundo nodo del arco
            
        Returns:
            Tupla con (distancias [x, y, z], error) o (None, error) si no hay solución válida
        """
        d_u_new = self.distance_matrix[u, new_node]
        d_v_new = self.distance_matrix[v, new_node]
        d_u_v = self.distance_matrix[u, v]
        
        # Calcular z (distancia del nodo interno al nuevo nodo)
        z = (d_u_new + d_v_new - d_u_v) / 2.0
        
        # Calcular x (distancia de u al nodo interno)
        x = d_u_new - z
        
        # Calcular y (distancia de v al nodo interno)
        y = d_v_new - z
        
        # Verificar que todas las distancias sean positivas
        if x < 0 or y < 0 or z < 0:
            # Calcular error de validación
            error = abs(x) + abs(y) + abs(z) if x < 0 or y < 0 or z < 0 else 0
            return None, error
        
        # Calcular error de validación (verificar que se cumplan las ecuaciones)
        error1 = abs((x + z) - d_u_new)
        error2 = abs((y + z) - d_v_new)
        error3 = abs((x + y) - d_u_v)
        total_error = error1 + error2 + error3
        
        return [x, y, z], total_error
    
    def _format_equations(self, u: int, v: int, new_node: int, internal_node: int, distances: List[float]) -> List[str]:
        """
        Formatea las ecuaciones del sistema resuelto.
        
        Args:
            u, v: Nodos del arco original
            new_node: Nuevo nodo insertado
            internal_node: Nodo interno creado
            distances: [dist_u_internal, dist_v_internal, dist_new_internal]
            
        Returns:
            Lista de strings con las ecuaciones formateadas
        """
        x, y, z = distances
        d_u_new = self.distance_matrix[u, new_node]
        d_v_new = self.distance_matrix[v, new_node]
        d_u_v = self.distance_matrix[u, v]
        
        equations = [
            f"d({self.labels[u]}, {self.labels[new_node]}) = d({self.labels[u]}, U{internal_node}) + d(U{internal_node}, {self.labels[new_node]})",
            f"  → {d_u_new:.2f} = {x:.2f} + {z:.2f}",
            f"",
            f"d({self.labels[v]}, {self.labels[new_node]}) = d({self.labels[v]}, U{internal_node}) + d(U{internal_node}, {self.labels[new_node]})",
            f"  → {d_v_new:.2f} = {y:.2f} + {z:.2f}",
            f"",
            f"d({self.labels[u]}, {self.labels[v]}) = d({self.labels[u]}, U{internal_node}) + d({self.labels[v]}, U{internal_node})",
            f"  → {d_u_v:.2f} = {x:.2f} + {y:.2f}",
            f"",
            f"Solución:",
            f"  d({self.labels[u]}, U{internal_node}) = {x:.2f}",
            f"  d({self.labels[v]}, U{internal_node}) = {y:.2f}",
            f"  d(U{internal_node}, {self.labels[new_node]}) = {z:.2f}"
        ]
        
        return equations
    
    def get_step(self, step_index: int) -> Dict[str, Any]:
        """Obtiene un paso específico."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def get_total_steps(self) -> int:
        """Retorna el número total de pasos."""
        return len(self.steps)

