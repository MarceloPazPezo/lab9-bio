# Aplicación de Árboles Filogenéticos

Aplicación Python con interfaz gráfica (Tkinter) para visualizar la construcción paso a paso de árboles filogenéticos usando algoritmos de árbol aditivo y árbol ultramétrico (Neighbor-Joining).

## Características

- **Árbol Aditivo**: Implementación paso a paso del algoritmo de árbol aditivo
  - Grafo inicial desde matriz Mh (pesos)
  - Spanning tree sin ciclos
  - Identificación de arcos de mayor valor
  - Cálculo de Cw usando matriz Ml (distancias)
  - Construcción de matriz Cw
  - Árbol final

- **Árbol Ultramétrico (Neighbor-Joining)**: Implementación paso a paso del algoritmo NJ
  - Visualización de cada iteración
  - Cálculo de matriz Q
  - Identificación de pares cercanos
  - Construcción progresiva del árbol

- **Visualización Interactiva**:
  - Navegación paso a paso (anterior/siguiente)
  - Reproducción automática
  - Pausa y reanudación
  - Visualización de grafos y matrices

- **Entrada de Datos**:
  - Carga desde archivos CSV/TXT
  - Entrada manual de matrices

## Requisitos

- Python 3.8 o superior
- Dependencias (ver `requirements.txt`):
  - numpy
  - matplotlib
  - networkx

## Instalación

1. Clonar o descargar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecutar la aplicación:
```bash
python main.py
```

### Árbol Aditivo

1. Seleccionar "Árbol Aditivo"
2. Cargar dos matrices:
   - **Mh**: Matriz de pesos (pesos de los arcos)
   - **Ml**: Matriz de distancias (distancias entre nodos)
3. Hacer clic en "Ejecutar Algoritmo"
4. Navegar por los pasos usando los controles

### Árbol Ultramétrico (Neighbor-Joining)

1. Seleccionar "Árbol Ultramétrico (NJ)"
2. Cargar una matriz de distancias
3. Hacer clic en "Ejecutar Algoritmo"
4. Navegar por los pasos usando los controles

### Formato de Matrices

Las matrices deben ser:
- Cuadradas
- Simétricas
- Sin valores NaN o infinitos

Ejemplo de formato CSV:
```
0,1,2,3
1,0,4,5
2,4,0,6
3,5,6,0
```

O formato de texto (separado por espacios):
```
0 1 2 3
1 0 4 5
2 4 0 6
3 5 6 0
```

## Estructura del Proyecto

```
lab9-bio/
├── main.py                 # Aplicación principal Tkinter
├── additive_tree.py        # Módulo de árbol aditivo
├── ultrametric_tree.py     # Módulo de árbol ultramétrico (Neighbor-Joining)
├── visualization.py         # Utilidades de visualización
├── matrix_utils.py         # Utilidades para matrices
├── requirements.txt        # Dependencias
└── README.md               # Este archivo
```

## Controles

- **◄ Anterior**: Retrocede al paso anterior
- **Siguiente ►**: Avanza al siguiente paso
- **▶ Automático**: Reproduce todos los pasos automáticamente
- **⏸ Pausa**: Pausa la reproducción automática

## Notas

- Los algoritmos capturan todos los pasos intermedios para visualización
- La visualización muestra grafos, árboles y matrices según corresponda
- El panel de información muestra detalles adicionales de cada paso

