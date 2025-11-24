# Aplicación de Árboles Filogenéticos - Proyecto Pepito

Aplicación Python con interfaz gráfica (Tkinter) para visualizar la construcción paso a paso de árboles filogenéticos usando dos algoritmos.

## Estructura del Proyecto

### Módulos

1. **`ultrametric_tree.py`** - Árbol Ultramétrico
   - **Entrada**: 2 matrices (Mh: pesos, Ml: distancias)
   - **Algoritmo**: Spanning tree, cálculo de Cw, construcción del árbol
   - **Clase**: `UltrametricTreeBuilder`

2. **`additive_tree.py`** - Árbol Aditivo
   - **Entrada**: 1 matriz de distancias
   - **Algoritmo**: Construcción nodo por nodo resolviendo sistemas de ecuaciones
   - **Clase**: `AdditiveTreeBuilder`

### Archivos de Soporte

- `main.py` - Aplicación principal con interfaz Tkinter
- `matrix_utils.py` - Utilidades para carga y validación de matrices
- `visualization.py` - Funciones de visualización con matplotlib
- `matrix_examples.py` - Ejemplos predefinidos
- `requirements.txt` - Dependencias del proyecto

## Requisitos

- Python 3.8 o superior
- Dependencias:
  - numpy
  - matplotlib
  - networkx

## Instalación

1. Crear entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecutar la aplicación:
```bash
source activar_entorno.sh
python main.py
```

### Árbol Ultramétrico

1. Seleccionar "Árbol Ultramétrico (Mh, Ml)"
2. Cargar dos matrices:
   - **Mh**: Matriz de pesos (pesos de los arcos)
   - **Ml**: Matriz de distancias (distancias entre nodos)
3. Hacer clic en "Ejecutar Algoritmo"
4. Navegar por los pasos usando los controles

### Árbol Aditivo

1. Seleccionar "Árbol Aditivo (1 matriz, sistemas)"
2. Cargar una matriz de distancias
3. Hacer clic en "Ejecutar Algoritmo"
4. Navegar por los pasos usando los controles
5. Ver los sistemas de ecuaciones resueltos para cada nodo

## Características

- **Visualización Interactiva**:
  - Navegación paso a paso (anterior/siguiente)
  - Reproducción automática
  - Pausa y reanudación
  - Visualización de grafos, árboles y matrices

- **Entrada de Datos**:
  - Carga desde archivos CSV/TXT
  - Entrada manual de matrices
  - Ejemplos predefinidos

## Notas

- El árbol ultramétrico usa el algoritmo con spanning tree y cálculo de Cw
- El árbol aditivo resuelve sistemas de ecuaciones para cada nodo interno
- Todos los pasos intermedios se capturan para visualización

