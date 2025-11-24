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
        
        description = f"Paso 1: Inicialización\n\n"
        description += f"Se conectan los primeros dos nodos del árbol:\n"
        description += f"- Nodo {self.labels[0]} ↔ Nodo {self.labels[1]}\n"
        description += f"- Distancia: {distance_01:.2f}\n\n"
        description += f"Este es el árbol inicial con 2 nodos conectados directamente."
        
        step1 = {
            'step': 1,
            'title': f'Paso 1: Inicialización - Conectar {self.labels[0]} y {self.labels[1]}',
            'matrix': self.distance_matrix.copy(),
            'tree': self.tree.copy(),
            'description': description,
            'equation_info': {
                'nodes': [0, 1],
                'distance': distance_01,
                'equation': f'd({self.labels[0]}, {self.labels[1]}) = {distance_01:.2f}'
            },
            'show_tree': True
        }
        self.steps.append(step1)
        
        # Pasos 2 a n-1: Agregar cada nodo restante (uno a la vez)
        for node_idx in range(2, self.n):
            step = self._add_node_to_tree(node_idx, len(self.steps))
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
    
    def _add_node_to_tree(self, node_idx: int, step_number: int) -> Dict[str, Any]:
        """
        Agrega un nuevo nodo al árbol existente, resolviendo un sistema de ecuaciones
        para determinar dónde insertarlo y las distancias.
        
        Args:
            node_idx: Índice del nodo a agregar
            step_number: Número de paso secuencial
            
        Returns:
            Diccionario con información del paso
        """
        new_node = node_idx
        self.tree.add_node(new_node, label=self.labels[new_node], is_leaf=True)
        
        # Encontrar el mejor lugar para insertar el nodo
        best_edge, internal_node, distances = self._find_best_insertion(new_node)
        
        if best_edge is None:
            # Si no se puede insertar en un arco, conectar directamente a un nodo existente
            # Esto solo debería pasar si el árbol tiene solo un arco
            min_dist = np.inf
            closest_node = 0
            for existing_node in self.tree.nodes():
                if existing_node != new_node:
                    # Calcular distancia (existing_node puede ser interno)
                    if existing_node < self.n:
                        dist = self.distance_matrix[new_node, existing_node]
                    else:
                        dist = self._get_distance_to_leaf(existing_node, new_node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = existing_node
            
            self.tree.add_edge(new_node, closest_node, distance=min_dist)
            
            # Crear ecuación simple para mostrar
            equations = [
                f"d({self.labels[new_node]}, {self.labels[closest_node]}) = {min_dist:.2f}",
                f"",
                f"Como el árbol solo tiene un arco, se conecta directamente."
            ]
            
            label_new = self._get_node_label(new_node)
            label_closest = self._get_node_label(closest_node)
            
            description = f"Paso {step_number}: Agregar nodo {label_new}\n\n"
            description += f"Como el árbol solo tiene un arco, se conecta directamente:\n"
            description += f"- {label_new} ↔ {label_closest}\n"
            description += f"- Distancia: {min_dist:.2f}"
            
            return {
                'step': step_number,
                'title': f'Paso {step_number}: Agregar {label_new}',
                'matrix': self.distance_matrix.copy(),
                'tree': self.tree.copy(),
                'description': description,
                'equation_info': {
                    'nodes': [new_node, closest_node],
                    'distance': min_dist,
                    'equation': f'd({label_new}, {label_closest}) = {min_dist:.2f}',
                    'equations': equations
                },
                'show_tree': True
            }
        
        # Insertar nodo interno en el arco
        u, v = best_edge
        self.tree.remove_edge(u, v)
        
        # Obtener distancias calculadas
        dist_u_to_internal = distances[0]
        dist_v_to_internal = distances[1]
        dist_new_to_internal = distances[2]
        
        # Tolerancia para considerar distancia como cero
        tolerance = 1e-6
        
        # Si alguna distancia es 0, fusionar nodos en lugar de crear uno nuevo
        if abs(dist_u_to_internal) < tolerance:
            # El nodo interno coincide con u, conectar directamente
            self.tree.add_edge(u, v, distance=dist_v_to_internal)
            self.tree.add_edge(u, new_node, distance=dist_new_to_internal)
            internal_node_id = u  # Usar u como nodo interno
        elif abs(dist_v_to_internal) < tolerance:
            # El nodo interno coincide con v, conectar directamente
            self.tree.add_edge(u, v, distance=dist_u_to_internal)
            self.tree.add_edge(v, new_node, distance=dist_new_to_internal)
            internal_node_id = v  # Usar v como nodo interno
        elif abs(dist_new_to_internal) < tolerance:
            # El nuevo nodo coincide con el nodo interno, conectar directamente a u o v
            # Usar el nodo más cercano
            if dist_u_to_internal < dist_v_to_internal:
                self.tree.add_edge(u, v, distance=dist_u_to_internal + dist_v_to_internal)
                self.tree.add_edge(u, new_node, distance=0.0)
                internal_node_id = u
            else:
                self.tree.add_edge(u, v, distance=dist_u_to_internal + dist_v_to_internal)
                self.tree.add_edge(v, new_node, distance=0.0)
                internal_node_id = v
        else:
            # Crear nodo interno solo si todas las distancias son significativas
            internal_node_id = self.node_counter
            self.node_counter += 1
            self.tree.add_node(internal_node_id, label=f'U{internal_node_id}', is_leaf=False)
            
            # Conectar nodos
            self.tree.add_edge(u, internal_node_id, distance=dist_u_to_internal)
            self.tree.add_edge(v, internal_node_id, distance=dist_v_to_internal)
            self.tree.add_edge(new_node, internal_node_id, distance=dist_new_to_internal)
        
        # Crear información de ecuaciones
        equations = self._format_equations(u, v, new_node, internal_node_id, distances)
        
        # Obtener labels de manera segura
        label_new = self._get_node_label(new_node)
        label_u = self._get_node_label(u)
        label_v = self._get_node_label(v)
        
        # Crear descripción detallada de la unión
        description = f"Paso {step_number}: Agregar nodo {label_new}\n\n"
        description += f"1. Se identifica el arco ({label_u}, {label_v}) como el mejor lugar para insertar {label_new}\n"
        
        # Determinar si se creó un nodo interno nuevo o se usó uno existente
        tolerance = 1e-6
        if abs(dist_u_to_internal) < tolerance:
            internal_label = label_u
            description += f"2. El nodo interno coincide con {label_u} (distancia ≈ 0)\n"
            description += f"3. Se resuelve el sistema de ecuaciones:\n"
            description += f"   - {label_u} ↔ {label_v}: {dist_v_to_internal:.2f}\n"
            description += f"   - {label_u} ↔ {label_new}: {dist_new_to_internal:.2f}\n"
            description += f"4. Se conectan: {label_u} ↔ {label_v} y {label_u} ↔ {label_new}"
        elif abs(dist_v_to_internal) < tolerance:
            internal_label = label_v
            description += f"2. El nodo interno coincide con {label_v} (distancia ≈ 0)\n"
            description += f"3. Se resuelve el sistema de ecuaciones:\n"
            description += f"   - {label_u} ↔ {label_v}: {dist_u_to_internal:.2f}\n"
            description += f"   - {label_v} ↔ {label_new}: {dist_new_to_internal:.2f}\n"
            description += f"4. Se conectan: {label_u} ↔ {label_v} y {label_v} ↔ {label_new}"
        elif abs(dist_new_to_internal) < tolerance:
            internal_label = label_u if dist_u_to_internal < dist_v_to_internal else label_v
            description += f"2. El nuevo nodo se conecta directamente (distancia ≈ 0)\n"
            description += f"3. Se resuelve el sistema de ecuaciones:\n"
            description += f"   - {label_u} ↔ {label_v}: {dist_u_to_internal + dist_v_to_internal:.2f}\n"
            description += f"   - {internal_label} ↔ {label_new}: 0.00\n"
            description += f"4. Se conectan: {label_u} ↔ {label_v} y {internal_label} ↔ {label_new}"
        else:
            internal_label = f'U{internal_node_id}'
            description += f"2. Se crea un nodo interno {internal_label} en este arco\n"
            description += f"3. Se resuelve el sistema de ecuaciones para calcular las distancias:\n"
            description += f"   - Distancia de {label_u} a {internal_label}: {dist_u_to_internal:.2f}\n"
            description += f"   - Distancia de {label_v} a {internal_label}: {dist_v_to_internal:.2f}\n"
            description += f"   - Distancia de {label_new} a {internal_label}: {dist_new_to_internal:.2f}\n"
            description += f"4. Se conectan los nodos: {label_u} ↔ {internal_label} ↔ {label_v} y {label_new} ↔ {internal_label}"
        
        # Determinar el label del nodo interno para el título
        tolerance = 1e-6
        if abs(dist_u_to_internal) < tolerance or (abs(dist_new_to_internal) < tolerance and dist_u_to_internal < dist_v_to_internal):
            internal_label_title = label_u
        elif abs(dist_v_to_internal) < tolerance or (abs(dist_new_to_internal) < tolerance):
            internal_label_title = label_v
        else:
            internal_label_title = f'U{internal_node_id}'
        
        return {
            'step': step_number,
            'title': f'Paso {step_number}: Agregar {label_new} - Unión en {internal_label_title}',
            'matrix': self.distance_matrix.copy(),
            'tree': self.tree.copy(),
            'description': description,
            'equation_info': {
                'nodes': [u, v, new_node, internal_node_id],
                'distances': distances,
                'equations': equations,
                'system_solved': True,
                'union_info': {
                    'new_node': label_new,
                    'edge': (label_u, label_v),
                    'internal_node': internal_label_title,
                    'distances': {
                        f'{label_u}-{internal_label_title}': dist_u_to_internal,
                        f'{label_v}-{internal_label_title}': dist_v_to_internal,
                        f'{label_new}-{internal_label_title}': dist_new_to_internal
                    }
                }
            },
            'internal_node': internal_node_id,
            'show_tree': True  # Forzar mostrar el árbol
        }
    
    def _get_tree_distance(self, u: int, v: int) -> float:
        """
        Calcula la distancia entre dos nodos en el árbol actual,
        sumando las distancias de las aristas a lo largo del camino.
        
        Args:
            u: Primer nodo
            v: Segundo nodo
            
        Returns:
            Distancia total en el árbol, o distancia de la matriz si no hay camino
        """
        if u == v:
            return 0.0
        
        # Si ambos nodos están en el árbol, calcular distancia en el árbol
        if u in self.tree.nodes() and v in self.tree.nodes():
            try:
                path = nx.shortest_path(self.tree, u, v)
                total_distance = 0.0
                for i in range(len(path) - 1):
                    edge_data = self.tree[path[i]][path[i+1]]
                    if 'distance' in edge_data:
                        total_distance += edge_data['distance']
                    elif 'weight' in edge_data:
                        total_distance += edge_data['weight']
                return total_distance
            except nx.NetworkXNoPath:
                pass
        
        # Si alguno no está en el árbol o son nodos hoja, usar distancia de la matriz
        if u < self.n and v < self.n:
            return self.distance_matrix[u, v]
        
        # Si uno es interno y otro es hoja, calcular desde el árbol
        # Esto no debería pasar normalmente, pero por seguridad
        return 0.0
    
    def _get_distance_to_leaf(self, node: int, leaf: int) -> float:
        """
        Calcula la distancia desde un nodo (puede ser interno) hasta un nodo hoja.
        
        Args:
            node: Nodo origen (puede ser interno o hoja)
            leaf: Nodo hoja destino (índice < n)
            
        Returns:
            Distancia desde node hasta leaf
        """
        if node == leaf:
            return 0.0
        
        # Si node es un nodo hoja, usar la matriz directamente
        if node < self.n:
            return self.distance_matrix[node, leaf]
        
        # Si node es un nodo interno, calcular distancia en el árbol
        if node in self.tree.nodes():
            try:
                path = nx.shortest_path(self.tree, node, leaf)
                total_distance = 0.0
                for i in range(len(path) - 1):
                    edge_data = self.tree[path[i]][path[i+1]]
                    if 'distance' in edge_data:
                        total_distance += edge_data['distance']
                    elif 'weight' in edge_data:
                        total_distance += edge_data['weight']
                return total_distance
            except nx.NetworkXNoPath:
                pass
        
        # Fallback: retornar 0 si no se puede calcular
        return 0.0
    
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
            
            if distances is not None:
                # Verificar que las distancias sean no negativas (con tolerancia)
                tolerance = 1e-6
                if all(d >= -tolerance for d in distances):
                    # Preferir soluciones con menor error
                    if error < best_error:
                        best_error = error
                        best_edge = (u, v)
                        best_distances = distances
        
        # Si no encontramos ninguna solución válida, usar el primer arco disponible
        # y forzar una solución (esto no debería pasar normalmente)
        if best_edge is None and len(list(self.tree.edges())) > 0:
            # Tomar el primer arco y crear una solución aproximada
            first_edge = list(self.tree.edges())[0]
            u, v = first_edge
            d_u_new = self._get_distance_to_leaf(u, new_node)
            d_v_new = self._get_distance_to_leaf(v, new_node)
            d_u_v = self._get_tree_distance(u, v)
            
            # Calcular solución
            z = max(0, (d_u_new + d_v_new - d_u_v) / 2.0)
            x = max(0, d_u_new - z)
            y = max(0, d_v_new - z)
            
            # Ajustar si x + y no coincide con d_u_v
            if abs((x + y) - d_u_v) > 1e-6:
                # Redistribuir proporcionalmente
                if (x + y) > 0:
                    factor = d_u_v / (x + y)
                    x = x * factor
                    y = y * factor
                else:
                    x = d_u_v / 2.0
                    y = d_u_v / 2.0
            
            best_edge = (u, v)
            best_distances = [x, y, z]
        
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
        # Calcular distancias desde u y v hasta new_node
        # u y v pueden ser nodos hoja o internos, pero new_node siempre es hoja
        d_u_new = self._get_distance_to_leaf(u, new_node)
        d_v_new = self._get_distance_to_leaf(v, new_node)
        
        # IMPORTANTE: Usar la distancia en el árbol actual entre u y v
        # (suma de aristas a lo largo del camino), no la distancia de la matriz
        d_u_v = self._get_tree_distance(u, v)
        
        # Calcular z (distancia del nodo interno al nuevo nodo)
        z = (d_u_new + d_v_new - d_u_v) / 2.0
        
        # Calcular x (distancia de u al nodo interno)
        x = d_u_new - z
        
        # Calcular y (distancia de v al nodo interno)
        y = d_v_new - z
        
        # Verificar que todas las distancias sean positivas (con tolerancia pequeña)
        tolerance = 1e-6
        if x < -tolerance or y < -tolerance or z < -tolerance:
            # Calcular error de validación
            error = abs(min(0, x)) + abs(min(0, y)) + abs(min(0, z))
            return None, error
        
        # Asegurar que las distancias no sean negativas (ajustar a 0 si son muy pequeñas)
        x = max(0, x)
        y = max(0, y)
        z = max(0, z)
        
        # Calcular error de validación (verificar que se cumplan las ecuaciones)
        error1 = abs((x + z) - d_u_new)
        error2 = abs((y + z) - d_v_new)
        error3 = abs((x + y) - d_u_v)
        total_error = error1 + error2 + error3
        
        return [x, y, z], total_error
    
    def _get_node_label(self, node: int) -> str:
        """
        Obtiene la etiqueta de un nodo, manejando nodos hoja e internos.
        
        Args:
            node: Índice del nodo
            
        Returns:
            Etiqueta del nodo
        """
        # Si es un nodo hoja, usar la lista de labels
        if node < self.n:
            return self.labels[node]
        
        # Si es un nodo interno, obtener el label del árbol
        if node in self.tree.nodes():
            return self.tree.nodes[node].get('label', f'U{node}')
        
        # Fallback
        return f'U{node}'
    
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
        # Calcular distancias (u y v pueden ser internos)
        d_u_new = self._get_distance_to_leaf(u, new_node)
        d_v_new = self._get_distance_to_leaf(v, new_node)
        # Usar la distancia en el árbol, no la de la matriz
        d_u_v = self._get_tree_distance(u, v)
        
        # Obtener labels de manera segura
        label_u = self._get_node_label(u)
        label_v = self._get_node_label(v)
        label_new = self._get_node_label(new_node)
        label_internal = f'U{internal_node}'
        
        equations = [
            f"d({label_u}, {label_new}) = d({label_u}, {label_internal}) + d({label_internal}, {label_new})",
            f"  → {d_u_new:.2f} = {x:.2f} + {z:.2f}",
            f"",
            f"d({label_v}, {label_new}) = d({label_v}, {label_internal}) + d({label_internal}, {label_new})",
            f"  → {d_v_new:.2f} = {y:.2f} + {z:.2f}",
            f"",
            f"d({label_u}, {label_v}) = d({label_u}, {label_internal}) + d({label_v}, {label_internal})",
            f"  → {d_u_v:.2f} = {x:.2f} + {y:.2f}",
            f"",
            f"Solución:",
            f"  d({label_u}, {label_internal}) = {x:.2f}",
            f"  d({label_v}, {label_internal}) = {y:.2f}",
            f"  d({label_internal}, {label_new}) = {z:.2f}"
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

