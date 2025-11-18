"""
Ejemplos predefinidos de matrices simétricas para pruebas.
"""
import numpy as np
from typing import Dict, Tuple, List


# Ejemplos para Árbol Aditivo (Mh y Ml)
ADDITIVE_EXAMPLES = {
    "Ejemplo 1 (3x3)": {
        "description": "Matriz pequeña de 3 nodos",
        "Mh": np.array([
            [0, 2, 3],
            [2, 0, 4],
            [3, 4, 0]
        ]),
        "Ml": np.array([
            [0, 5, 7],
            [5, 0, 9],
            [7, 9, 0]
        ]),
        "labels": ["A", "B", "C"]
    },
    "Ejemplo 2 (4x4)": {
        "description": "Matriz mediana de 4 nodos",
        "Mh": np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ]),
        "Ml": np.array([
            [0, 2, 4, 6],
            [2, 0, 8, 10],
            [4, 8, 0, 12],
            [6, 10, 12, 0]
        ]),
        "labels": ["Nodo1", "Nodo2", "Nodo3", "Nodo4"]
    },
    "Ejemplo 3 (4x4) - Complejo": {
        "description": "Matriz 4x4 con valores más variados",
        "Mh": np.array([
            [0, 3, 1, 5],
            [3, 0, 2, 4],
            [1, 2, 0, 6],
            [5, 4, 6, 0]
        ]),
        "Ml": np.array([
            [0, 7, 3, 11],
            [7, 0, 5, 9],
            [3, 5, 0, 13],
            [11, 9, 13, 0]
        ]),
        "labels": ["X", "Y", "Z", "W"]
    },
    "Ejemplo 4 (5x5)": {
        "description": "Matriz grande de 5 nodos",
        "Mh": np.array([
            [0, 5, 7, 8, 7],
            [5, 0, 7, 7, 8],
            [7, 7, 0, 6, 6],
            [8, 7, 6, 0, 4],
            [7, 8, 6, 4, 0]
        ]),
        "Ml": np.array([
            [0, 1, 4, 5, 3],
            [1, 0, 4, 5, 4],
            [4, 4, 0, 3, 2],
            [5, 5, 3, 0, 1],
            [3, 4, 2, 1, 0]
        ]),
        "labels": ["A", "B", "C", "D", "E"]
    }
}

# Ejemplos para Árbol Ultramétrico (Neighbor-Joining)
ULTRAMETRIC_EXAMPLES = {
    "Ejemplo 1 (3x3)": {
        "description": "Matriz pequeña de 3 especies",
        "matrix": np.array([
            [0, 5, 9],
            [5, 0, 10],
            [9, 10, 0]
        ]),
        "labels": ["Humano", "Chimpancé", "Gorila"]
    },
    "Ejemplo 2 (4x4)": {
        "description": "Matriz de 4 especies",
        "matrix": np.array([
            [0, 2, 4, 6],
            [2, 0, 4, 6],
            [4, 4, 0, 6],
            [6, 6, 6, 0]
        ]),
        "labels": ["A", "B", "C", "D"]
    },
    "Ejemplo 3 (4x4) - Biológico": {
        "description": "Distancias entre 4 especies (ejemplo biológico)",
        "matrix": np.array([
            [0, 3, 7, 9],
            [3, 0, 6, 8],
            [7, 6, 0, 4],
            [9, 8, 4, 0]
        ]),
        "labels": ["Especie1", "Especie2", "Especie3", "Especie4"]
    },
    "Ejemplo 4 (5x5)": {
        "description": "Matriz grande de 5 especies",
        "matrix": np.array([
            [0, 2, 4, 6, 8],
            [2, 0, 4, 6, 8],
            [4, 4, 0, 4, 6],
            [6, 6, 4, 0, 4],
            [8, 8, 6, 4, 0]
        ]),
        "labels": ["A", "B", "C", "D", "E"]
    },
    "Ejemplo 5 (5x5) - Variado": {
        "description": "Matriz 5x5 con valores variados",
        "matrix": np.array([
            [0, 3, 5, 7, 9],
            [3, 0, 4, 6, 8],
            [5, 4, 0, 4, 6],
            [7, 6, 4, 0, 5],
            [9, 8, 6, 5, 0]
        ]),
        "labels": ["N1", "N2", "N3", "N4", "N5"]
    }
}


def get_additive_examples() -> Dict:
    """Retorna los ejemplos para árbol aditivo."""
    return ADDITIVE_EXAMPLES


def get_ultrametric_examples() -> Dict:
    """Retorna los ejemplos para árbol ultramétrico."""
    return ULTRAMETRIC_EXAMPLES


def get_example_names(algorithm_type: str) -> List[str]:
    """Retorna los nombres de los ejemplos disponibles."""
    if algorithm_type == "additive":
        return list(ADDITIVE_EXAMPLES.keys())
    else:
        return list(ULTRAMETRIC_EXAMPLES.keys())

