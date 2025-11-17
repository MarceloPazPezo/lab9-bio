"""
Aplicación principal para visualización paso a paso de árboles filogenéticos.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import threading
import time

from additive_tree import AdditiveTreeBuilder
from ultrametric_tree import UltrametricTreeBuilder
from matrix_utils import load_matrix_from_file, validate_matrix, parse_matrix_from_text, matrix_to_string
from matrix_examples import get_additive_examples, get_ultrametric_examples, get_example_names
from visualization import draw_graph, draw_tree, draw_matrix, draw_calculation_text, create_canvas


class PhylogeneticTreeApp:
    """Aplicación principal para árboles filogenéticos."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Árboles Filogenéticos - Visualización Paso a Paso")
        self.root.geometry("1700x900")
        
        # Variables
        self.algorithm_type = tk.StringVar(value="additive")
        self.steps = []
        self.current_step_index = 0
        self.auto_playing = False
        self.auto_thread = None
        
        # Matrices
        self.Mh = None
        self.Ml = None
        self.distance_matrix = None
        
        # Etiquetas de nodos
        self.node_labels = None
        self.node_labels_mh = None
        self.node_labels_ml = None
        
        self.setup_ui()
        # Configurar protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Maneja el cierre de la aplicación."""
        # Detener reproducción automática si está activa
        self.stop_auto()
        # Esperar un momento para que el hilo termine
        if self.auto_thread and self.auto_thread.is_alive():
            import time
            time.sleep(0.1)
        # Cerrar la aplicación
        self.root.destroy()
    
    def setup_ui(self):
        """Configura la interfaz de usuario."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Panel izquierdo: Controles
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_panel.configure(width=350)
        
        # Panel central: Visualización
        center_panel = ttk.Frame(main_frame, padding="10")
        center_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        center_panel.columnconfigure(0, weight=1)
        center_panel.rowconfigure(0, weight=1)
        
        # Panel derecho: Matrices de entrada
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.grid(row=0, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.configure(width=400)
        right_panel.columnconfigure(0, weight=1)
        
        # Panel inferior: Información
        info_panel = ttk.Frame(main_frame, padding="10")
        info_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_panel.columnconfigure(0, weight=1)
        info_panel.rowconfigure(0, weight=1)
        
        self.setup_control_panel(left_panel)
        self.setup_visualization_panel(center_panel)
        self.setup_info_panel(info_panel)
        self.setup_matrices_panel(right_panel)
    
    def setup_control_panel(self, parent):
        """Configura el panel de controles."""
        # Título
        title_label = ttk.Label(parent, text="Configuración", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Selección de algoritmo
        algo_frame = ttk.LabelFrame(parent, text="Algoritmo", padding="10")
        algo_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(algo_frame, text="Árbol Aditivo", variable=self.algorithm_type,
                       value="additive", command=self.on_algorithm_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(algo_frame, text="Árbol Ultramétrico (NJ)", variable=self.algorithm_type,
                       value="ultrametric", command=self.on_algorithm_change).grid(row=1, column=0, sticky=tk.W)
        
        # Entrada de matrices
        matrix_frame = ttk.LabelFrame(parent, text="Entrada de Matrices", padding="10")
        matrix_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Botones de carga
        ttk.Button(matrix_frame, text="Cargar desde Archivo", 
                  command=self.load_from_file).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(matrix_frame, text="Entrada Manual", 
                  command=self.open_manual_input).grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(matrix_frame, text="Cargar Ejemplo", 
                  command=self.load_example).grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Etiquetas de estado
        self.matrix_status_label = ttk.Label(matrix_frame, text="No hay matrices cargadas", 
                                            foreground="red")
        self.matrix_status_label.grid(row=3, column=0, pady=5)
        
        # Botón de ejecución
        self.execute_button = ttk.Button(parent, text="Ejecutar Algoritmo", 
                                        command=self.execute_algorithm, state=tk.DISABLED)
        self.execute_button.grid(row=4, column=0, pady=20, sticky=(tk.W, tk.E))
        
        # Controles de navegación
        nav_frame = ttk.LabelFrame(parent, text="Navegación", padding="10")
        nav_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.grid(row=0, column=0, pady=5)
        
        self.prev_button = ttk.Button(nav_buttons_frame, text="◄ Anterior", 
                                     command=self.prev_step, state=tk.DISABLED)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        self.next_button = ttk.Button(nav_buttons_frame, text="Siguiente ►", 
                                     command=self.next_step, state=tk.DISABLED)
        self.next_button.grid(row=0, column=1, padx=5)
        
        self.auto_button = ttk.Button(nav_buttons_frame, text="▶ Automático", 
                                     command=self.toggle_auto, state=tk.DISABLED)
        self.auto_button.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.pause_button = ttk.Button(nav_buttons_frame, text="⏸ Pausa", 
                                      command=self.pause_auto, state=tk.DISABLED)
        self.pause_button.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Información del paso
        self.step_info_label = ttk.Label(nav_frame, text="Paso: 0 / 0", font=("Arial", 10))
        self.step_info_label.grid(row=1, column=0, pady=10)
    
    def setup_visualization_panel(self, parent):
        """Configura el panel de visualización."""
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.text(0.5, 0.5, "Seleccione un algoritmo y cargue las matrices", 
                    ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def setup_info_panel(self, parent):
        """Configura el panel de información."""
        info_label = ttk.Label(parent, text="Información del Paso", font=("Arial", 12, "bold"))
        info_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(parent, height=8, wrap=tk.WORD)
        self.info_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
    
    def setup_matrices_panel(self, parent):
        """Configura el panel de matrices de entrada."""
        title_label = ttk.Label(parent, text="📊 Matrices de Entrada", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 15), sticky=tk.W)
        
        # Frame para las matrices con scroll
        matrices_frame = ttk.Frame(parent)
        matrices_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(matrices_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(matrices_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.matrices_canvas = canvas
        self.matrices_frame = scrollable_frame
        self.matrices_container = scrollable_frame
        
        # Inicializar con mensaje
        self.update_matrices_display()
    
    def on_algorithm_change(self):
        """Se llama cuando cambia el algoritmo seleccionado."""
        self.reset_state()
        self.update_matrices_display()
    
    def load_from_file(self):
        """Carga matrices desde archivo."""
        if self.algorithm_type.get() == "additive":
            # Necesita dos archivos: Mh y Ml
            filepath = filedialog.askopenfilename(
                title="Seleccionar matriz Mh (pesos)",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filepath:
                try:
                    self.Mh = load_matrix_from_file(filepath)
                    is_valid, error_msg = validate_matrix(self.Mh, symmetric=True)
                    if not is_valid:
                        messagebox.showerror("Error", f"Matriz Mh inválida: {error_msg}")
                        self.Mh = None
                        return
                    
                    filepath2 = filedialog.askopenfilename(
                        title="Seleccionar matriz Ml (distancias)",
                        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
                    )
                    if filepath2:
                        self.Ml = load_matrix_from_file(filepath2)
                        is_valid, error_msg = validate_matrix(self.Ml, symmetric=True)
                        if not is_valid:
                            messagebox.showerror("Error", f"Matriz Ml inválida: {error_msg}")
                            self.Ml = None
                            return
                        
                        if self.Mh.shape != self.Ml.shape:
                            messagebox.showerror("Error", "Las matrices Mh y Ml deben tener el mismo tamaño")
                            self.Mh = None
                            self.Ml = None
                            return
                        
                        self.matrix_status_label.config(
                            text=f"Matrices cargadas: {self.Mh.shape[0]}x{self.Mh.shape[1]}",
                            foreground="green"
                        )
                        self.execute_button.config(state=tk.NORMAL)
                        self.update_matrices_display()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al cargar archivo: {e}")
        else:
            # Solo necesita una matriz de distancias
            filepath = filedialog.askopenfilename(
                title="Seleccionar matriz de distancias",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filepath:
                try:
                    self.distance_matrix = load_matrix_from_file(filepath)
                    is_valid, error_msg = validate_matrix(self.distance_matrix, symmetric=True)
                    if not is_valid:
                        messagebox.showerror("Error", f"Matriz inválida: {error_msg}")
                        self.distance_matrix = None
                        return
                    
                    self.matrix_status_label.config(
                        text=f"Matriz cargada: {self.distance_matrix.shape[0]}x{self.distance_matrix.shape[1]}",
                        foreground="green"
                    )
                    self.execute_button.config(state=tk.NORMAL)
                    self.update_matrices_display()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al cargar archivo: {e}")
    
    def load_example(self):
        """Carga un ejemplo predefinido."""
        window = tk.Toplevel(self.root)
        window.title("Seleccionar Ejemplo")
        window.geometry("500x400")
        
        # Frame principal
        main_frame = ttk.Frame(window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Seleccione un ejemplo:", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Lista de ejemplos
        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Obtener ejemplos según el algoritmo
        if self.algorithm_type.get() == "additive":
            examples = get_additive_examples()
        else:
            examples = get_ultrametric_examples()
        
        # Llenar lista
        for name, example in examples.items():
            listbox.insert(tk.END, f"{name} - {example['description']}")
        
        # Frame de información
        info_frame = ttk.LabelFrame(main_frame, text="Información", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        info_label = ttk.Label(info_frame, text="", wraplength=400)
        info_label.pack()
        
        def on_select(event):
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                example_name = list(examples.keys())[idx]
                example = examples[example_name]
                info_label.config(text=f"Descripción: {example['description']}")
        
        listbox.bind('<<ListboxSelect>>', on_select)
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        def load_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Advertencia", "Por favor seleccione un ejemplo")
                return
            
            idx = selection[0]
            example_name = list(examples.keys())[idx]
            example = examples[example_name]
            
            try:
                if self.algorithm_type.get() == "additive":
                    self.Mh = example['Mh'].copy()
                    self.Ml = example['Ml'].copy()
                    self.node_labels_mh = example.get('labels', None)
                    self.node_labels_ml = example.get('labels', None)
                    
                    self.matrix_status_label.config(
                        text=f"Ejemplo cargado: {example_name} ({self.Mh.shape[0]}x{self.Mh.shape[1]})",
                        foreground="green"
                    )
                else:
                    self.distance_matrix = example['matrix'].copy()
                    self.node_labels = example.get('labels', None)
                    
                    self.matrix_status_label.config(
                        text=f"Ejemplo cargado: {example_name} ({self.distance_matrix.shape[0]}x{self.distance_matrix.shape[1]})",
                        foreground="green"
                    )
                
                self.execute_button.config(state=tk.NORMAL)
                self.update_matrices_display()
                window.destroy()
                messagebox.showinfo("Éxito", f"Ejemplo '{example_name}' cargado correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar ejemplo: {e}")
        
        ttk.Button(button_frame, text="Cargar", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=window.destroy).pack(side=tk.LEFT, padx=5)
    
    def open_manual_input(self):
        """Abre ventana para entrada manual visual de matrices."""
        window = tk.Toplevel(self.root)
        window.title("Entrada Visual de Matriz")
        window.geometry("800x700")
        
        if self.algorithm_type.get() == "additive":
            # Dos matrices con tabs
            notebook = ttk.Notebook(window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Tab para Mh
            frame_mh = ttk.Frame(notebook, padding="10")
            notebook.add(frame_mh, text="Matriz Mh (Pesos)")
            self._create_matrix_input_ui(frame_mh, "Mh")
            
            # Tab para Ml
            frame_ml = ttk.Frame(notebook, padding="10")
            notebook.add(frame_ml, text="Matriz Ml (Distancias)")
            self._create_matrix_input_ui(frame_ml, "Ml")
            
            def save_matrices():
                try:
                    # Obtener datos de ambas matrices
                    data_mh, labels_mh = self._get_matrix_data(frame_mh)
                    data_ml, labels_ml = self._get_matrix_data(frame_ml)
                    
                    if data_mh is None or data_ml is None:
                        return
                    
                    self.Mh = np.array(data_mh, dtype=float)
                    self.Ml = np.array(data_ml, dtype=float)
                    
                    is_valid_mh, error_mh = validate_matrix(self.Mh, symmetric=True)
                    is_valid_ml, error_ml = validate_matrix(self.Ml, symmetric=True)
                    
                    if not is_valid_mh:
                        messagebox.showerror("Error", f"Matriz Mh inválida: {error_mh}")
                        return
                    if not is_valid_ml:
                        messagebox.showerror("Error", f"Matriz Ml inválida: {error_ml}")
                        return
                    if self.Mh.shape != self.Ml.shape:
                        messagebox.showerror("Error", "Las matrices deben tener el mismo tamaño")
                        return
                    
                    # Guardar etiquetas si se proporcionaron
                    if labels_mh and all(labels_mh):
                        self.node_labels_mh = labels_mh
                    if labels_ml and all(labels_ml):
                        self.node_labels_ml = labels_ml
                    
                    self.matrix_status_label.config(
                        text=f"Matrices cargadas: {self.Mh.shape[0]}x{self.Mh.shape[1]}",
                        foreground="green"
                    )
                    self.execute_button.config(state=tk.NORMAL)
                    self.update_matrices_display()
                    window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al guardar matrices: {e}")
            
            button_frame = ttk.Frame(window)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Guardar Matrices", command=save_matrices).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancelar", command=window.destroy).pack(side=tk.LEFT, padx=5)
        else:
            # Una matriz
            frame = ttk.Frame(window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            self._create_matrix_input_ui(frame, "distance")
            
            def save_matrix():
                try:
                    data, labels = self._get_matrix_data(frame)
                    if data is None:
                        return
                    
                    self.distance_matrix = np.array(data, dtype=float)
                    is_valid, error_msg = validate_matrix(self.distance_matrix, symmetric=True)
                    
                    if not is_valid:
                        messagebox.showerror("Error", f"Matriz inválida: {error_msg}")
                        return
                    
                    # Guardar etiquetas si se proporcionaron
                    if labels and all(labels):
                        self.node_labels = labels
                    
                    self.matrix_status_label.config(
                        text=f"Matriz cargada: {self.distance_matrix.shape[0]}x{self.distance_matrix.shape[1]}",
                        foreground="green"
                    )
                    self.execute_button.config(state=tk.NORMAL)
                    self.update_matrices_display()
                    window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error al guardar matriz: {e}")
            
            button_frame = ttk.Frame(window)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Guardar Matriz", command=save_matrix).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancelar", command=window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _create_matrix_input_ui(self, parent, matrix_type):
        """Crea la interfaz visual para entrada de matriz."""
        # Frame superior: tamaño y nombres de nodos
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Selector de tamaño
        size_frame = ttk.Frame(top_frame)
        size_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(size_frame, text="Tamaño de matriz:").pack(side=tk.LEFT, padx=5)
        size_var = tk.IntVar(value=3)
        size_spinbox = ttk.Spinbox(size_frame, from_=2, to=10, textvariable=size_var, width=5,
                                   command=lambda: self._update_matrix_grid(parent, size_var.get(), matrix_type))
        size_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(size_frame, text="Crear Matriz", 
                  command=lambda: self._update_matrix_grid(parent, size_var.get(), matrix_type)).pack(side=tk.LEFT, padx=5)
        
        # Frame para nombres de nodos
        names_frame = ttk.LabelFrame(parent, text="Nombres de Nodos (opcional)", padding="5")
        names_frame.pack(fill=tk.X, pady=5)
        self.node_name_entries = {}
        
        # Frame para la matriz (con scrollbar)
        matrix_frame = ttk.LabelFrame(parent, text="Matriz", padding="15")
        matrix_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas con scrollbar para matrices grandes
        canvas = tk.Canvas(matrix_frame, bg="#F5F5F5", highlightthickness=0)
        scrollbar_v = ttk.Scrollbar(matrix_frame, orient="vertical", command=canvas.yview)
        scrollbar_h = ttk.Scrollbar(matrix_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg="#F5F5F5")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        
        # Almacenar referencias
        parent._matrix_canvas = canvas
        parent._matrix_frame = scrollable_frame
        parent._size_var = size_var
        parent._matrix_type = matrix_type
        parent._matrix_entries = {}
        
        # Crear matriz inicial
        self._update_matrix_grid(parent, 3, matrix_type)
    
    def _update_matrix_grid(self, parent, size, matrix_type):
        """Actualiza la grilla de la matriz."""
        # Limpiar frame anterior
        for widget in parent._matrix_frame.winfo_children():
            widget.destroy()
        parent._matrix_entries = {}
        
        # Buscar frame de nombres de nodos
        names_frame = None
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                try:
                    if "Nombres" in widget.cget("text"):
                        names_frame = widget
                        # Limpiar contenido anterior
                        for child in widget.winfo_children():
                            child.destroy()
                        break
                except:
                    pass
        
        if names_frame:
            # Crear campos de nombres
            names_grid = ttk.Frame(names_frame)
            names_grid.pack(fill=tk.X, padx=5, pady=5)
            parent._node_name_entries = {}
            
            for i in range(size):
                ttk.Label(names_grid, text=f"Nodo {i+1}:").grid(row=0, column=i*2, padx=5, pady=2)
                entry = ttk.Entry(names_grid, width=12)
                entry.grid(row=0, column=i*2+1, padx=5, pady=2)
                entry.insert(0, f"Node {i+1}")
                # Actualizar encabezados cuando cambie el nombre
                def update_headers(idx=i, entry_widget=entry):
                    def on_change(event=None):
                        self._update_matrix_headers(parent)
                    return on_change
                entry.bind('<KeyRelease>', update_headers())
                parent._node_name_entries[i] = entry
        
        # Obtener nombres de nodos si están disponibles
        node_names = {}
        if hasattr(parent, '_node_name_entries'):
            for i in range(size):
                if i in parent._node_name_entries:
                    name = parent._node_name_entries[i].get().strip()
                    node_names[i] = name if name else f"Node {i+1}"
                else:
                    node_names[i] = f"Node {i+1}"
        else:
            node_names = {i: f"Node {i+1}" for i in range(size)}
        
        # Crear tabla de matriz con estilo visual mejorado
        table_frame = tk.Frame(parent._matrix_frame, bg="#F5F5F5")
        table_frame.grid(row=0, column=0, pady=10)
        
        # Celda vacía en esquina superior izquierda
        corner_label = tk.Label(table_frame, text="", width=12, height=2, 
                               relief=tk.RAISED, bg="#2E7D32", borderwidth=2)
        corner_label.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        
        # Encabezados de columnas con nombres de nodos
        for j in range(size):
            node_name = node_names.get(j, f"Node {j+1}")
            header_label = tk.Label(table_frame, text=node_name[:10], width=12, height=2,
                                   relief=tk.RAISED, bg="#4CAF50", fg="white", 
                                   font=("Arial", 10, "bold"), borderwidth=2)
            header_label.grid(row=0, column=j+1, padx=1, pady=1, sticky="nsew")
        
        # Crear filas de la matriz
        for i in range(size):
            # Encabezado de fila con nombre de nodo
            node_name = node_names.get(i, f"Node {i+1}")
            row_header = tk.Label(table_frame, text=node_name[:10], width=12, height=2,
                                 relief=tk.RAISED, bg="#4CAF50", fg="white",
                                 font=("Arial", 10, "bold"), borderwidth=2)
            row_header.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            
            # Celdas de la matriz
            for j in range(size):
                # Crear frame para la celda con borde
                cell_frame = tk.Frame(table_frame, bg="#424242", bd=1)
                cell_frame.grid(row=i+1, column=j+1, padx=1, pady=1, sticky="nsew")
                
                entry = tk.Entry(cell_frame, width=10, justify=tk.CENTER, 
                               font=("Arial", 11), relief=tk.FLAT, bd=0,
                               bg="white", fg="black")
                entry.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
                
                # Si es la diagonal, poner 0 y deshabilitar
                if i == j:
                    entry.insert(0, "0")
                    entry.config(state=tk.DISABLED, bg="#E8E8E8", fg="#666666",
                               font=("Arial", 11, "italic"))
                else:
                    entry.insert(0, "0")
                    # Hacer que la matriz sea simétrica automáticamente
                    def make_symmetric(row=i, col=j, entry_widget=entry):
                        def on_change(event=None):
                            try:
                                val = entry_widget.get()
                                if val.strip():
                                    # Actualizar celda simétrica
                                    if (col, row) in parent._matrix_entries:
                                        parent._matrix_entries[(col, row)].delete(0, tk.END)
                                        parent._matrix_entries[(col, row)].insert(0, val)
                            except:
                                pass
                        return on_change
                    entry.bind('<KeyRelease>', make_symmetric())
                    entry.bind('<FocusIn>', lambda e, ef=entry: ef.config(bg="#E3F2FD"))
                    entry.bind('<FocusOut>', lambda e, ef=entry: ef.config(bg="white"))
                
                parent._matrix_entries[(i, j)] = entry
        
        # Configurar pesos de grid para que las celdas se expandan uniformemente
        for i in range(size + 1):
            table_frame.grid_columnconfigure(i, weight=1, uniform="col")
        for i in range(size + 1):
            table_frame.grid_rowconfigure(i, weight=1, uniform="row")
    
    def _update_matrix_headers(self, parent):
        """Actualiza los encabezados de la matriz con los nombres de nodos."""
        if not hasattr(parent, '_matrix_frame') or not hasattr(parent, '_node_name_entries'):
            return
        
        # Buscar el frame de la tabla
        widgets = parent._matrix_frame.winfo_children()
        if not widgets:
            return
        
        # Buscar el table_frame (primer Frame que contiene la tabla)
        table_frame = None
        for widget in widgets:
            if isinstance(widget, tk.Frame):
                # Verificar si tiene el grid configurado (es nuestra tabla)
                try:
                    grid_info = widget.grid_info()
                    if grid_info:
                        table_frame = widget
                        break
                except:
                    pass
        
        if table_frame:
            # Actualizar encabezados de columnas (fila 0, columnas 1+)
            for j in range(len(parent._node_name_entries)):
                widget = table_frame.grid_slaves(row=0, column=j+1)
                if widget and isinstance(widget[0], tk.Label):
                    if j in parent._node_name_entries:
                        name = parent._node_name_entries[j].get().strip()
                        widget[0].config(text=name[:10] if name else f"Node {j+1}")
            
            # Actualizar encabezados de filas (columna 0, filas 1+)
            for i in range(len(parent._node_name_entries)):
                widget = table_frame.grid_slaves(row=i+1, column=0)
                if widget and isinstance(widget[0], tk.Label):
                    if i in parent._node_name_entries:
                        name = parent._node_name_entries[i].get().strip()
                        widget[0].config(text=name[:10] if name else f"Node {i+1}")
    
    def _get_matrix_data(self, parent):
        """Obtiene los datos de la matriz desde la interfaz."""
        try:
            size = parent._size_var.get()
            data = []
            labels = []
            
            # Obtener nombres de nodos si existen
            if hasattr(parent, '_node_name_entries'):
                for i in range(size):
                    if i in parent._node_name_entries:
                        name = parent._node_name_entries[i].get().strip()
                        labels.append(name if name else f"Node {i+1}")
                    else:
                        labels.append(f"Node {i+1}")
            else:
                labels = [f"Node {i+1}" for i in range(size)]
            
            # Obtener valores de la matriz
            for i in range(size):
                row = []
                for j in range(size):
                    if (i, j) in parent._matrix_entries:
                        val = parent._matrix_entries[(i, j)].get().strip()
                        if not val:
                            val = "0"
                        row.append(float(val))
                    else:
                        row.append(0.0)
                data.append(row)
            
            return data, labels
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer matriz: {e}")
            return None, None
    
    def execute_algorithm(self):
        """Ejecuta el algoritmo seleccionado."""
        try:
            if self.algorithm_type.get() == "additive":
                if self.Mh is None or self.Ml is None:
                    messagebox.showerror("Error", "Debe cargar las matrices Mh y Ml")
                    return
                
                # Usar etiquetas de Mh si están disponibles
                labels = self.node_labels_mh if self.node_labels_mh else None
                builder = AdditiveTreeBuilder(self.Mh, self.Ml, labels=labels)
                self.steps = builder.build()
            else:
                if self.distance_matrix is None:
                    messagebox.showerror("Error", "Debe cargar la matriz de distancias")
                    return
                
                # Usar etiquetas personalizadas si están disponibles
                labels = self.node_labels if self.node_labels else None
                builder = UltrametricTreeBuilder(self.distance_matrix, labels=labels)
                self.steps = builder.build()
            
            self.current_step_index = 0
            self.update_navigation_buttons()
            self.show_step(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar algoritmo: {e}")
            import traceback
            traceback.print_exc()
    
    def _safe_show_step(self, step_index: int):
        """Versión segura de show_step que verifica que los widgets existan."""
        try:
            if self._widget_exists(self.root):
                self.show_step(step_index)
        except (tk.TclError, AttributeError):
            pass
    
    def show_step(self, step_index: int):
        """Muestra un paso específico."""
        try:
            if not self.steps or step_index < 0 or step_index >= len(self.steps):
                return
        except (AttributeError, IndexError):
            return
        
        step = self.steps[step_index]
        self.current_step_index = step_index
        
        # Limpiar axes
        self.ax.clear()
        
        # Mostrar según el tipo de paso (priorizar grafo/árbol sobre matriz)
        # Para paso 3 y 4, siempre mostrar la matriz aunque haya grafo
        if step.get('step') == 3 and 'path_matrix' in step:
            # Paso 3: mostrar matriz de arcos
            matrix = step['path_matrix']
            if self.algorithm_type.get() == "additive" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                labels = self.node_labels_mh[:matrix.shape[0]]
            else:
                labels = [f'Node {i+1}' for i in range(matrix.shape[0])]
            draw_matrix(matrix, self.ax, step.get('title', ''), labels)
        elif step.get('step') == 4 and 'calculation_texts' in step:
            # Paso 4: mostrar cálculos como texto formateado
            calculation_texts = step['calculation_texts']
            draw_calculation_text(calculation_texts, self.ax, step.get('title', ''))
        else:
            graph = step.get('graph') or step.get('tree')
            if graph:
                # Preparar etiquetas - usar labels personalizados si están disponibles
                node_labels = {}
                for node in graph.nodes():
                    # Primero intentar obtener del grafo
                    label = graph.nodes[node].get('label', None)
                    if label:
                        node_labels[node] = str(label)
                    else:
                        # Si no hay label en el grafo, usar etiquetas personalizadas de la app
                        if self.algorithm_type.get() == "additive" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                            if node < len(self.node_labels_mh):
                                node_labels[node] = str(self.node_labels_mh[node])
                            else:
                                node_labels[node] = str(node)
                        elif self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels') and self.node_labels:
                            if node < len(self.node_labels):
                                node_labels[node] = str(self.node_labels[node])
                            else:
                                node_labels[node] = str(node)
                        else:
                            node_labels[node] = str(node)
                
                edge_labels = {}
                for u, v in graph.edges():
                    if 'distance' in graph[u][v]:
                        edge_labels[(u, v)] = f"{graph[u][v]['distance']:.3f}"
                    elif 'weight' in graph[u][v]:
                        edge_labels[(u, v)] = f"{graph[u][v]['weight']:.3f}"
                
                draw_tree(graph, self.ax, step.get('title', ''), 
                         node_labels, edge_labels)
            elif 'matrix' in step:
                # Solo mostrar matriz si no hay grafo
                matrix = step['matrix']
                # Usar etiquetas personalizadas si están disponibles
                if self.algorithm_type.get() == "additive" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                    labels = self.node_labels_mh[:matrix.shape[0]]
                elif self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels') and self.node_labels:
                    labels = self.node_labels[:matrix.shape[0]]
                else:
                    labels = [f'Node {i+1}' for i in range(matrix.shape[0])]
                draw_matrix(matrix, self.ax, step.get('title', ''), labels)
        
        # Actualizar información
        info_text = f"Paso {step.get('step', step_index)}: {step.get('title', '')}\n\n"
        info_text += f"{step.get('description', '')}\n\n"
        
        if 'edges_info' in step:
            info_text += "Información de arcos:\n"
            for edge_info in step['edges_info']:
                if 'edge_label' in edge_info:
                    edge_label = edge_info['edge_label']
                    info_text += f"  Arco {edge_label}: Peso Mh={edge_info['weight_Mh']:.3f}, Cw={edge_info['Cw']:.3f}"
                    if 'pairs_count' in edge_info:
                        info_text += f" (usado por {edge_info['pairs_count']} pares)\n"
                        if 'pairs_str' in edge_info:
                            info_text += f"    Pares: {edge_info['pairs_str']}\n"
                    else:
                        info_text += "\n"
                else:
                    u, v = edge_info['edge']
                    info_text += f"  Arco ({u+1}, {v+1}): Peso Mh={edge_info['weight_Mh']:.3f}, Cw={edge_info['Cw']:.3f}\n"
        
        if 'path_matrix' in step:
            info_text += "\nMatriz de arcos (Paso 3):\n"
            info_text += "Cada celda muestra el arco de mayor peso en el camino entre los nodos.\n"
        
        if 'Q_matrix' in step:
            info_text += "\nMatriz Q:\n"
            info_text += matrix_to_string(step['Q_matrix'], precision=3)
            info_text += "\n\n"
            if 'min_pair' in step:
                i, j = step['min_pair']
                info_text += f"Par mínimo encontrado: ({i+1}, {j+1})\n"
        
        if 'distances' in step:
            info_text += f"\nDistancias:\n"
            info_text += f"  Distancia i->u: {step['distances']['i_to_u']:.3f}\n"
            info_text += f"  Distancia j->u: {step['distances']['j_to_u']:.3f}\n"
        
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", info_text)
        
        # Actualizar canvas
        self.canvas.draw()
        self.update_step_info()
    
    def prev_step(self):
        """Retrocede al paso anterior."""
        if self.current_step_index > 0:
            self.show_step(self.current_step_index - 1)
    
    def next_step(self):
        """Avanza al siguiente paso."""
        if self.current_step_index < len(self.steps) - 1:
            self.show_step(self.current_step_index + 1)
    
    def toggle_auto(self):
        """Inicia o detiene la reproducción automática."""
        if not self.auto_playing:
            self.start_auto()
        else:
            self.stop_auto()
    
    def start_auto(self):
        """Inicia la reproducción automática."""
        if not self.steps:
            return
        
        try:
            self.auto_playing = True
            if hasattr(self, 'auto_button') and self._widget_exists(self.auto_button):
                self.auto_button.config(text="⏸ Detener", state=tk.NORMAL)
            if hasattr(self, 'pause_button') and self._widget_exists(self.pause_button):
                self.pause_button.config(state=tk.NORMAL)
            if hasattr(self, 'prev_button') and self._widget_exists(self.prev_button):
                self.prev_button.config(state=tk.DISABLED)
            if hasattr(self, 'next_button') and self._widget_exists(self.next_button):
                self.next_button.config(state=tk.DISABLED)
        except (tk.TclError, AttributeError):
            # Los widgets ya no existen
            self.auto_playing = False
            return
        
        def auto_play():
            try:
                while self.auto_playing and self.current_step_index < len(self.steps) - 1:
                    time.sleep(2)  # Esperar 2 segundos entre pasos
                    if self.auto_playing:
                        try:
                            # Verificar que la ventana aún existe
                            if self._widget_exists(self.root):
                                self.root.after(0, lambda idx=self.current_step_index + 1: self._safe_show_step(idx))
                            else:
                                break
                        except (tk.TclError, AttributeError):
                            # La ventana fue cerrada
                            break
            except (tk.TclError, AttributeError):
                # La aplicación fue cerrada
                pass
        
        self.auto_thread = threading.Thread(target=auto_play, daemon=True)
        self.auto_thread.start()
    
    def stop_auto(self):
        """Detiene la reproducción automática."""
        self.auto_playing = False
        try:
            if hasattr(self, 'auto_button') and self._widget_exists(self.auto_button):
                self.auto_button.config(text="▶ Automático", state=tk.NORMAL)
            if hasattr(self, 'pause_button') and self._widget_exists(self.pause_button):
                self.pause_button.config(state=tk.DISABLED)
            self.update_navigation_buttons()
        except (tk.TclError, AttributeError):
            # Los widgets ya no existen, ignorar
            pass
    
    def pause_auto(self):
        """Pausa la reproducción automática."""
        self.stop_auto()
    
    def _widget_exists(self, widget):
        """Verifica si un widget aún existe."""
        try:
            if widget is None:
                return False
            widget.winfo_exists()
            return True
        except (tk.TclError, AttributeError):
            return False
    
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación."""
        try:
            has_steps = len(self.steps) > 0
            if hasattr(self, 'prev_button') and self._widget_exists(self.prev_button):
                self.prev_button.config(state=tk.NORMAL if has_steps and self.current_step_index > 0 else tk.DISABLED)
            if hasattr(self, 'next_button') and self._widget_exists(self.next_button):
                self.next_button.config(state=tk.NORMAL if has_steps and self.current_step_index < len(self.steps) - 1 else tk.DISABLED)
            if hasattr(self, 'auto_button') and self._widget_exists(self.auto_button):
                self.auto_button.config(state=tk.NORMAL if has_steps else tk.DISABLED)
        except (tk.TclError, AttributeError):
            # Los widgets ya no existen, ignorar
            pass
    
    def update_step_info(self):
        """Actualiza la información del paso actual."""
        try:
            total = len(self.steps)
            current = self.current_step_index + 1
            if hasattr(self, 'step_info_label') and self._widget_exists(self.step_info_label):
                self.step_info_label.config(text=f"Paso: {current} / {total}")
            self.update_navigation_buttons()
        except (tk.TclError, AttributeError):
            # Los widgets ya no existen, ignorar
            pass
    
    def update_matrices_display(self):
        """Actualiza el panel de matrices de entrada."""
        # Limpiar frame anterior
        for widget in self.matrices_container.winfo_children():
            widget.destroy()
        
        if self.algorithm_type.get() == "additive":
            if self.Mh is not None and self.Ml is not None:
                # Mostrar Mh con mejor formato visual
                mh_frame = ttk.LabelFrame(self.matrices_container, text="Matriz Mh (Pesos)", padding="5")
                mh_frame.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
                
                mh_table = self._create_matrix_table(mh_frame, self.Mh, self.node_labels_mh)
                mh_table.pack(fill=tk.BOTH, expand=True)
                
                # Mostrar Ml con mejor formato visual
                ml_frame = ttk.LabelFrame(self.matrices_container, text="Matriz Ml (Distancias)", padding="5")
                ml_frame.grid(row=1, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
                
                ml_table = self._create_matrix_table(ml_frame, self.Ml, self.node_labels_ml)
                ml_table.pack(fill=tk.BOTH, expand=True)
                
                self.matrices_container.columnconfigure(0, weight=1)
            else:
                no_data_label = ttk.Label(self.matrices_container, 
                                         text="No hay matrices cargadas", 
                                         foreground="gray", font=("Arial", 10))
                no_data_label.grid(row=0, column=0, pady=20)
        else:
            if self.distance_matrix is not None:
                # Mostrar matriz de distancias con mejor formato visual
                dist_frame = ttk.LabelFrame(self.matrices_container, text="Matriz de Distancias", padding="5")
                dist_frame.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
                
                dist_table = self._create_matrix_table(dist_frame, self.distance_matrix, self.node_labels)
                dist_table.pack(fill=tk.BOTH, expand=True)
                
                self.matrices_container.columnconfigure(0, weight=1)
            else:
                no_data_label = ttk.Label(self.matrices_container, 
                                         text="No hay matriz cargada", 
                                         foreground="gray", font=("Arial", 10))
                no_data_label.grid(row=0, column=0, pady=20)
        
        # Actualizar scroll region
        self.matrices_container.update_idletasks()
        self.matrices_canvas.configure(scrollregion=self.matrices_canvas.bbox("all"))
    
    def _create_matrix_table(self, parent, matrix, labels=None):
        """Crea una tabla visual mejorada para mostrar la matriz."""
        if labels is None:
            labels = [f"Node {i+1}" for i in range(matrix.shape[0])]
        
        # Frame para la tabla con scroll horizontal si es necesario
        outer_frame = tk.Frame(parent)
        outer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas con scrollbar horizontal
        canvas = tk.Canvas(outer_frame, bg="white", highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(outer_frame, orient="horizontal", command=canvas.xview)
        table_frame = tk.Frame(canvas, bg="white")
        
        canvas.create_window((0, 0), window=table_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set)
        
        canvas.pack(side="top", fill="both", expand=True)
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Crear encabezados de columnas
        header_frame = tk.Frame(table_frame, bg="#2E7D32")
        header_frame.pack(fill=tk.X)
        
        # Celda vacía en esquina
        corner = tk.Label(header_frame, text="", width=12, height=2, 
                         bg="#1B5E20", fg="white", font=("Arial", 10, "bold"))
        corner.pack(side=tk.LEFT, padx=1, pady=1)
        
        # Encabezados de columnas
        for j, label in enumerate(labels):
            header = tk.Label(header_frame, text=str(label)[:10], width=10, height=2,
                            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                            relief=tk.RAISED, bd=2)
            header.pack(side=tk.LEFT, padx=1, pady=1)
        
        # Crear filas de la matriz
        for i, label in enumerate(labels):
            row_frame = tk.Frame(table_frame, bg="white")
            row_frame.pack(fill=tk.X)
            
            # Encabezado de fila
            row_header = tk.Label(row_frame, text=str(label)[:10], width=12, height=2,
                                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                                 relief=tk.RAISED, bd=2)
            row_header.pack(side=tk.LEFT, padx=1, pady=1)
            
            # Celdas de la matriz
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if isinstance(val, (int, float)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)[:8]
                
                # Color alternado para mejor legibilidad
                if i == j:
                    bg_color = "#E8E8E8"  # Diagonal
                    fg_color = "#666666"
                elif (i + j) % 2 == 0:
                    bg_color = "#FFFFFF"
                    fg_color = "#000000"
                else:
                    bg_color = "#F5F5F5"
                    fg_color = "#000000"
                
                cell = tk.Label(row_frame, text=val_str, width=10, height=2,
                              bg=bg_color, fg=fg_color, font=("Courier", 10, "bold"),
                              relief=tk.SUNKEN, bd=1, anchor=tk.CENTER)
                cell.pack(side=tk.LEFT, padx=1, pady=1)
        
        # Actualizar scroll region
        def update_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        table_frame.bind("<Configure>", update_scroll)
        
        return outer_frame
    
    def _create_matrix_text(self, matrix, labels=None):
        """Crea una representación en texto de la matriz."""
        if labels is None:
            labels = [f"Node {i+1}" for i in range(matrix.shape[0])]
        
        # Calcular ancho de columna
        max_label_len = max(len(str(l)) for l in labels)
        col_width = max(max_label_len + 2, 8)
        
        # Crear encabezado
        text = " " * (col_width + 2)  # Espacio para etiqueta de fila
        for label in labels:
            text += f"{str(label):>{col_width}} "
        text += "\n"
        
        # Crear filas
        for i, label in enumerate(labels):
            text += f"{str(label):>{col_width}} "
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if isinstance(val, (int, float)):
                    text += f"{val:>{col_width}.3f} "
                else:
                    text += f"{str(val):>{col_width}} "
            text += "\n"
        
        return text
    
    def reset_state(self):
        """Resetea el estado de la aplicación."""
        self.steps = []
        self.current_step_index = 0
        self.stop_auto()
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Seleccione un algoritmo y cargue las matrices", 
                    ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()
        self.info_text.delete("1.0", tk.END)
        self.update_step_info()
        self.execute_button.config(state=tk.DISABLED)
        self.update_matrices_display()


def main():
    """Función principal."""
    root = tk.Tk()
    app = PhylogeneticTreeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

