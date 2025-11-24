"""
Ejemplos predefinidos de matrices simétricas para pruebas.
"""
import numpy as np
from typing import Dict


# Ejemplos para Árbol Ultramétrico (Mh y Ml)
# NOTA: Mh es distancia, Ml es peso
ULTRAMETRIC_EXAMPLES = {
    "Ejemplo 1 (3x3)": {
        "description": "Matriz pequeña de 3 nodos",
        "Mh": np.array([
            [0, 5, 7],
            [5, 0, 9],
            [7, 9, 0]
        ]),
        "Ml": np.array([
            [0, 2, 3],
            [2, 0, 4],
            [3, 4, 0]
        ]),
        "labels": ["A", "B", "C"]
    },
    "Ejemplo 2 (4x4)": {
        "description": "Matriz mediana de 4 nodos",
        "Mh": np.array([
            [0, 2, 4, 6],
            [2, 0, 8, 10],
            [4, 8, 0, 12],
            [6, 10, 12, 0]
        ]),
        "Ml": np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ]),
        "labels": ["Nodo1", "Nodo2", "Nodo3", "Nodo4"]
    },
    "Ejemplo 3 (5x5)": {
        "description": "Matriz grande de 5 nodos",
        "Mh": np.array([
            [0, 3, 6, 9, 12],
            [3, 0, 5, 8, 11],
            [6, 5, 0, 7, 10],
            [9, 8, 7, 0, 13],
            [12, 11, 10, 13, 0]
        ]),
        "Ml": np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 1.5, 2.5, 3.5],
            [2, 1.5, 0, 2, 3],
            [3, 2.5, 2, 0, 4.5],
            [4, 3.5, 3, 4.5, 0]
        ]),
        "labels": ["X", "Y", "Z", "W", "V"]
    },
    "Ejemplo 4 (3x3) - Simple": {
        "description": "Ejemplo simple de 3 nodos con valores pequeños",
        "Mh": np.array([
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ]),
        "Ml": np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ]),
        "labels": ["1", "2", "3"]
    }
}

# Ejemplos para Árbol Aditivo (1 matriz, sistemas de ecuaciones)
ADDITIVE_EXAMPLES = {
    "Ejemplo 1 (3x3)": {
        "description": "Matriz pequeña de 3 nodos",
        "matrix": np.array([
            [0, 5, 9],
            [5, 0, 10],
            [9, 10, 0]
        ]),
        "labels": ["A", "B", "C"]
    },
    "Ejemplo 2 (4x4)": {
        "description": "Matriz mediana de 4 nodos",
        "matrix": np.array([
            [0, 2, 4, 6],
            [2, 0, 4, 6],
            [4, 4, 0, 6],
            [6, 6, 6, 0]
        ]),
        "labels": ["Nodo1", "Nodo2", "Nodo3", "Nodo4"]
    },
    "Ejemplo 3 (5x5)": {
        "description": "Matriz grande de 5 nodos",
        "matrix": np.array([
            [0, 3, 6, 9, 12],
            [3, 0, 5, 8, 11],
            [6, 5, 0, 7, 10],
            [9, 8, 7, 0, 13],
            [12, 11, 10, 13, 0]
        ]),
        "labels": ["X", "Y", "Z", "W", "V"]
    },
    "Ejemplo 4 (3x3) - Aditivo Simple": {
        "description": "Ejemplo simple de 3 nodos",
        "matrix": np.array([
            [0, 7, 11],
            [7, 0, 8],
            [11, 8, 0]
        ]),
        "labels": ["1", "2", "3"]
    }
}


def get_ultrametric_examples() -> Dict:
    """Retorna los ejemplos para árbol ultramétrico (Mh y Ml)."""
    return ULTRAMETRIC_EXAMPLES


def get_additive_examples() -> Dict:
    """Retorna los ejemplos para árbol aditivo (1 matriz, sistemas de ecuaciones)."""
    return ADDITIVE_EXAMPLES

