"""
Módulo para construcción de árboles ultramétricos usando Neighbor-Joining.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional


class UltrametricTreeBuilder:
    """
    Constructor de árboles ultramétricos usando algoritmo Neighbor-Joining paso a paso.
    """
    
    def __init__(self, distance_matrix: np.ndarray, labels: Optional[List[str]] = None):
        """
        Inicializa el constructor con la matriz de distancias.
        
        Args:
            distance_matrix: Matriz de distancias
            labels: Etiquetas para los nodos (opcional)
        """
        self.distance_matrix = distance_matrix.copy()
        self.n = distance_matrix.shape[0]
        self.labels = labels if labels else [f'Node {i+1}' for i in range(self.n)]
        self.steps = []
        self.current_matrix = distance_matrix.copy()
        self.node_counter = self.n
        self.tree = nx.Graph()
        self.node_distances = {}  # Almacena distancias de nodos a raíz
        
    def build(self) -> List[Dict[str, Any]]:
        """
        Construye el árbol ultramétrico paso a paso usando Neighbor-Joining.
        
        Returns:
            Lista de pasos con información para visualización
        """
        self.steps = []
        
        # Inicializar árbol con nodos hoja
        for i in range(self.n):
            self.tree.add_node(i, label=self.labels[i], is_leaf=True)
        
        # Paso inicial: mostrar matriz original
        step0 = {
            'step': 0,
            'title': 'Paso 0: Matriz de Distancias Inicial',
            'matrix': self.current_matrix.copy(),
            'description': f'Matriz de distancias inicial con {self.n} nodos',
            'tree': self.tree.copy()
        }
        self.steps.append(step0)
        
        # Iterar hasta que queden 2 nodos
        iteration = 1
        current_labels = self.labels.copy()
        current_matrix = self.current_matrix.copy()
        active_nodes = list(range(self.n))
        
        while len(active_nodes) > 2:
            # Calcular matriz Q
            Q_matrix = self._calculate_Q_matrix(current_matrix)
            
            # Encontrar par de nodos más cercanos
            min_i, min_j = self._find_min_pair(Q_matrix, current_matrix.shape[0])
            
            # Crear nuevo nodo interno
            new_node = self.node_counter
            self.node_counter += 1
            
            # Calcular distancias del nuevo nodo a los nodos existentes
            u_distances = self._calculate_u_distances(current_matrix, min_i, min_j, current_matrix.shape[0])
            
            # Actualizar árbol
            node_i = active_nodes[min_i]
            node_j = active_nodes[min_j]
            
            # Calcular distancias de las hojas al nuevo nodo
            dist_i_to_u = 0.5 * (current_matrix[min_i, min_j] + 
                                (np.sum(current_matrix[min_i, :]) - np.sum(current_matrix[min_j, :])) / 
                                (len(active_nodes) - 2))
            dist_j_to_u = current_matrix[min_i, min_j] - dist_i_to_u
            
            self.tree.add_node(new_node, label=f'U{new_node}', is_leaf=False)
            self.tree.add_edge(node_i, new_node, distance=dist_i_to_u)
            self.tree.add_edge(node_j, new_node, distance=dist_j_to_u)
            
            # Crear nueva matriz de distancias
            new_matrix, new_labels, new_active_nodes = self._update_distance_matrix(
                current_matrix, current_labels, active_nodes, min_i, min_j, new_node, u_distances
            )
            
            # Guardar paso
            step = {
                'step': iteration,
                'title': f'Paso {iteration}: Unión de {current_labels[min_i]} y {current_labels[min_j]}',
                'matrix': current_matrix.copy(),
                'Q_matrix': Q_matrix,
                'min_pair': (min_i, min_j),
                'new_node': new_node,
                'distances': {
                    'i_to_u': dist_i_to_u,
                    'j_to_u': dist_j_to_u
                },
                'description': f'Se unen {current_labels[min_i]} y {current_labels[min_j]} creando nodo U{new_node}',
                'tree': self.tree.copy(),
                'new_matrix': new_matrix.copy()
            }
            self.steps.append(step)
            
            # Actualizar para siguiente iteración
            current_matrix = new_matrix
            current_labels = new_labels
            active_nodes = new_active_nodes
            iteration += 1
        
        # Paso final: unir los dos últimos nodos
        if len(active_nodes) == 2:
            node_i = active_nodes[0]
            node_j = active_nodes[1]
            final_distance = current_matrix[0, 1]
            
            # Si no hay raíz, crear una
            if not any(not self.tree.nodes[n].get('is_leaf', True) for n in self.tree.nodes()):
                root = self.node_counter
                self.tree.add_node(root, label='Root', is_leaf=False)
                self.tree.add_edge(node_i, root, distance=final_distance / 2)
                self.tree.add_edge(node_j, root, distance=final_distance / 2)
            else:
                # Conectar directamente
                self.tree.add_edge(node_i, node_j, distance=final_distance)
        
        # Paso final
        final_step = {
            'step': iteration,
            'title': 'Paso Final: Árbol Ultramétrico Completo',
            'matrix': current_matrix.copy(),
            'description': 'Árbol ultramétrico completo construido',
            'tree': self.tree.copy()
        }
        self.steps.append(final_step)
        
        return self.steps
    
    def _calculate_Q_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz Q para Neighbor-Joining.
        Q(i,j) = (n-2)*d(i,j) - sum(d(i,k)) - sum(d(j,k))
        """
        n = matrix.shape[0]
        Q = np.zeros_like(matrix, dtype=float)
        
        for i in range(n):
            for j in range(i + 1, n):
                sum_i = np.sum(matrix[i, :])
                sum_j = np.sum(matrix[j, :])
                Q[i, j] = (n - 2) * matrix[i, j] - sum_i - sum_j
                Q[j, i] = Q[i, j]
        
        # Diagonal en infinito para evitar seleccionarla
        np.fill_diagonal(Q, np.inf)
        return Q
    
    def _find_min_pair(self, Q_matrix: np.ndarray, n: int) -> Tuple[int, int]:
        """Encuentra el par de nodos con menor valor en Q."""
        min_val = np.inf
        min_i, min_j = 0, 1
        
        for i in range(n):
            for j in range(i + 1, n):
                if Q_matrix[i, j] < min_val:
                    min_val = Q_matrix[i, j]
                    min_i, min_j = i, j
        
        return min_i, min_j
    
    def _calculate_u_distances(self, matrix: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
        """
        Calcula las distancias del nuevo nodo u a todos los demás nodos.
        d(u,k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        """
        u_distances = np.zeros(n)
        for k in range(n):
            if k != i and k != j:
                u_distances[k] = 0.5 * (matrix[i, k] + matrix[j, k] - matrix[i, j])
        return u_distances
    
    def _update_distance_matrix(self, matrix: np.ndarray, labels: List[str],
                               active_nodes: List[int], i: int, j: int,
                               new_node: int, u_distances: np.ndarray) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Actualiza la matriz de distancias después de unir dos nodos.
        """
        n = matrix.shape[0]
        new_n = n - 1
        new_matrix = np.zeros((new_n, new_n))
        new_labels = []
        new_active_nodes = []
        
        # Agregar nuevo nodo
        new_labels.append(f'U{new_node}')
        new_active_nodes.append(new_node)
        
        # Copiar distancias de otros nodos
        new_idx = 1
        old_to_new = {}
        for k in range(n):
            if k != i and k != j:
                old_to_new[k] = new_idx
                new_labels.append(labels[k])
                new_active_nodes.append(active_nodes[k])
                new_idx += 1
        
        # Llenar nueva matriz
        # Distancia del nuevo nodo a los demás
        new_idx = 1
        for k in range(n):
            if k != i and k != j:
                new_matrix[0, new_idx] = u_distances[k]
                new_matrix[new_idx, 0] = u_distances[k]
                new_idx += 1
        
        # Distancias entre nodos existentes
        for k in range(n):
            if k != i and k != j:
                for l in range(k + 1, n):
                    if l != i and l != j:
                        new_k = old_to_new[k]
                        new_l = old_to_new[l]
                        new_matrix[new_k, new_l] = matrix[k, l]
                        new_matrix[new_l, new_k] = matrix[k, l]
        
        return new_matrix, new_labels, new_active_nodes
    
    def get_step(self, step_index: int) -> Dict[str, Any]:
        """Obtiene un paso específico."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def get_total_steps(self) -> int:
        """Retorna el número total de pasos."""
        return len(self.steps)

