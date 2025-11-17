"""
Utilidades para carga y validación de matrices.
"""
import numpy as np
import csv
from typing import Tuple, Optional, List


def load_matrix_from_file(filepath: str) -> np.ndarray:
    """
    Carga una matriz desde un archivo CSV o TXT.
    
    Args:
        filepath: Ruta al archivo
        
    Returns:
        Matriz numpy
    """
    try:
        # Intentar cargar como CSV
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Filtrar filas vacías y convertir a float
                row_data = [float(x.strip()) for x in row if x.strip()]
                if row_data:
                    data.append(row_data)
        
        matrix = np.array(data)
        return matrix
    except Exception as e:
        raise ValueError(f"Error al cargar matriz desde archivo: {e}")


def validate_matrix(matrix: np.ndarray, symmetric: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Valida que una matriz tenga el formato correcto.
    
    Args:
        matrix: Matriz a validar
        symmetric: Si debe ser simétrica
        
    Returns:
        Tupla (es_válida, mensaje_error)
    """
    if matrix is None or matrix.size == 0:
        return False, "La matriz está vacía"
    
    if len(matrix.shape) != 2:
        return False, "La matriz debe ser bidimensional"
    
    if matrix.shape[0] != matrix.shape[1]:
        return False, "La matriz debe ser cuadrada"
    
    if symmetric:
        if not np.allclose(matrix, matrix.T):
            return False, "La matriz debe ser simétrica"
    
    # Verificar que no tenga NaN o infinitos
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        return False, "La matriz contiene valores NaN o infinitos"
    
    return True, None


def parse_matrix_from_text(text: str) -> np.ndarray:
    """
    Parsea una matriz desde texto (formato separado por espacios o comas).
    
    Args:
        text: Texto con la matriz
        
    Returns:
        Matriz numpy
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    data = []
    
    for line in lines:
        # Intentar separar por espacios o comas
        if ',' in line:
            values = [float(x.strip()) for x in line.split(',') if x.strip()]
        else:
            values = [float(x.strip()) for x in line.split() if x.strip()]
        
        if values:
            data.append(values)
    
    if not data:
        raise ValueError("No se pudo parsear la matriz desde el texto")
    
    return np.array(data)


def matrix_to_string(matrix: np.ndarray, precision: int = 3) -> str:
    """
    Convierte una matriz a string formateado.
    
    Args:
        matrix: Matriz a convertir
        precision: Número de decimales
        
    Returns:
        String formateado
    """
    return '\n'.join([' '.join([f'{val:.{precision}f}' for val in row]) for row in matrix])

