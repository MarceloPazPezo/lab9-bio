"""
Aplicación principal para visualización paso a paso de árboles filogenéticos.
Versión simplificada con 2 algoritmos: Ultramétrico (Mh, Ml) y Aditivo (1 matriz, sistemas de ecuaciones).
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import threading
import time
import sys

from ultrametric_tree import UltrametricTreeBuilder
from additive_tree import AdditiveTreeBuilder
from matrix_utils import load_matrix_from_file, validate_matrix, parse_matrix_from_text, matrix_to_string
from visualization import draw_graph, draw_tree, draw_matrix, draw_calculation_text, create_canvas, draw_additive_tree
from matrix_examples import get_ultrametric_examples, get_additive_examples


class TreeApp:
    """Aplicación principal para árboles filogenéticos."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Árboles Filogenéticos - Visualización Paso a Paso")
        
        # Iniciar en pantalla completa (compatible con Windows y Linux)
        try:
            if sys.platform == 'win32':
                self.root.state('zoomed')
            else:
                self.root.attributes('-zoomed', True)
        except:
            self.root.update_idletasks()
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.root.geometry(f'{width}x{height}+0+0')
        
        # Variables
        self.algorithm_type = tk.StringVar(value="ultrametric")
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
    
    def on_closing(self):
        """Maneja el cierre de la aplicación."""
        self.auto_playing = False
        if hasattr(self, 'auto_thread') and self.auto_thread and self.auto_thread.is_alive():
            time.sleep(0.2)
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                try:
                    widget.destroy()
                except:
                    pass
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def setup_ui(self):
        """Configura la interfaz de usuario."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        right_panel.configure(width=550)
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
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_control_panel(self, parent):
        """Configura el panel de controles."""
        title_label = ttk.Label(parent, text="Configuración", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Selección de algoritmo
        algo_frame = ttk.LabelFrame(parent, text="Algoritmo", padding="10")
        algo_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(algo_frame, text="Árbol Ultramétrico (Mh, Ml)", variable=self.algorithm_type,
                       value="ultrametric", command=self.on_algorithm_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(algo_frame, text="Árbol Aditivo (1 matriz, sistemas)", variable=self.algorithm_type,
                       value="additive", command=self.on_algorithm_change).grid(row=1, column=0, sticky=tk.W)
        
        # Entrada de matrices
        matrix_frame = ttk.LabelFrame(parent, text="Entrada de Matrices", padding="10")
        matrix_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(matrix_frame, text="Cargar desde Archivo", 
                  command=self.load_from_file).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(matrix_frame, text="Entrada Manual", 
                  command=self.open_manual_input).grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(matrix_frame, text="Cargar Ejemplo", 
                  command=self.load_example).grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
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
        
        self.step_info_label = ttk.Label(nav_frame, text="Paso: 0 / 0", font=("Arial", 10))
        self.step_info_label.grid(row=1, column=0, pady=10)
    
    def setup_visualization_panel(self, parent):
        """Configura el panel de visualización."""
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
        
        self.info_text = scrolledtext.ScrolledText(parent, height=12, wrap=tk.WORD, font=("Consolas", 9))
        self.info_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
    
    def setup_matrices_panel(self, parent):
        """Configura el panel de matrices de entrada."""
        title_label = ttk.Label(parent, text="📊 Matrices de Entrada", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 15), sticky=tk.W)
        
        matrices_frame = ttk.Frame(parent)
        matrices_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
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
        
        self.update_matrices_display()
    
    def on_algorithm_change(self):
        """Se llama cuando cambia el algoritmo seleccionado."""
        self.reset_state()
        self.update_matrices_display()
    
    def load_from_file(self):
        """Carga matrices desde archivo según el algoritmo seleccionado."""
        algo_type = self.algorithm_type.get()
        
        if algo_type == "ultrametric":
            # Necesita dos archivos: Mh y Ml
            filepath = filedialog.askopenfilename(
                title="Seleccionar matriz Mh (distancias)",
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
                        title="Seleccionar matriz Ml (pesos)",
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
        else:  # additive
            # Solo una matriz de distancias
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
        """Carga un ejemplo predefinido desde un diálogo de selección."""
        algo_type = self.algorithm_type.get()
        
        # Crear ventana de selección
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Seleccionar Ejemplo")
        selection_window.geometry("500x400")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        # Centrar ventana
        selection_window.update_idletasks()
        x = (selection_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (selection_window.winfo_screenheight() // 2) - (400 // 2)
        selection_window.geometry(f"500x400+{x}+{y}")
        
        if algo_type == "ultrametric":
            examples = get_ultrametric_examples()
            title_text = "Seleccionar Ejemplo - Árbol Ultramétrico"
        else:  # additive
            examples = get_additive_examples()
            title_text = "Seleccionar Ejemplo - Árbol Aditivo"
        
        # Título
        title_label = ttk.Label(selection_window, text=title_text, font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # Frame para la lista
        list_frame = ttk.Frame(selection_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Lista de ejemplos
        example_listbox = tk.Listbox(list_frame, font=("Arial", 10), height=12)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=example_listbox.yview)
        example_listbox.configure(yscrollcommand=scrollbar.set)
        
        example_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Llenar lista
        example_keys = list(examples.keys())
        for key in example_keys:
            example_listbox.insert(tk.END, key)
        
        # Descripción
        desc_label = ttk.Label(selection_window, text="", font=("Arial", 9), foreground="gray", wraplength=450)
        desc_label.pack(pady=5)
        
        def on_select(event):
            selection = example_listbox.curselection()
            if selection:
                idx = selection[0]
                example_key = example_keys[idx]
                example = examples[example_key]
                desc_label.config(text=f"Descripción: {example.get('description', 'Sin descripción')}")
        
        example_listbox.bind('<<ListboxSelect>>', on_select)
        
        # Botones
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        def load_selected():
            selection = example_listbox.curselection()
            if not selection:
                messagebox.showwarning("Advertencia", "Por favor seleccione un ejemplo")
                return
            
            idx = selection[0]
            example_key = example_keys[idx]
            example = examples[example_key]
            
            try:
                if algo_type == "ultrametric":
                    self.Mh = example["Mh"].copy()
                    self.Ml = example["Ml"].copy()
                    self.node_labels_mh = example.get("labels", None)
                    self.node_labels_ml = example.get("labels", None)
                    self.matrix_status_label.config(
                        text=f"Ejemplo cargado: {example_key} ({self.Mh.shape[0]}x{self.Mh.shape[1]})",
                        foreground="green"
                    )
                else:  # additive
                    self.distance_matrix = example["matrix"].copy()
                    self.node_labels = example.get("labels", None)
                    self.matrix_status_label.config(
                        text=f"Ejemplo cargado: {example_key} ({self.distance_matrix.shape[0]}x{self.distance_matrix.shape[1]})",
                        foreground="green"
                    )
                
                self.execute_button.config(state=tk.NORMAL)
                self.update_matrices_display()
                selection_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar ejemplo: {e}")
        
        ttk.Button(button_frame, text="Cargar", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=selection_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Seleccionar primer elemento por defecto
        if example_keys:
            example_listbox.selection_set(0)
            example_listbox.see(0)
            on_select(None)
    
    def open_manual_input(self):
        """Abre ventana para entrada manual visual de matrices."""
        algo_type = self.algorithm_type.get()
        window = tk.Toplevel(self.root)
        window.title("Entrada Visual de Matriz")
        window.geometry("800x700")
        
        if algo_type == "ultrametric":
            # Dos matrices con tabs
            notebook = ttk.Notebook(window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            frame_mh = ttk.Frame(notebook, padding="10")
            notebook.add(frame_mh, text="Matriz Mh (Distancias)")
            self._create_matrix_input_ui(frame_mh, "Mh")
            
            frame_ml = ttk.Frame(notebook, padding="10")
            notebook.add(frame_ml, text="Matriz Ml (Pesos)")
            self._create_matrix_input_ui(frame_ml, "Ml")
            
            def save_matrices():
                try:
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
        else:  # additive
            # Solo una matriz
            frame = ttk.Frame(window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
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
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=5)
        
        size_frame = ttk.Frame(top_frame)
        size_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(size_frame, text="Tamaño de matriz:").pack(side=tk.LEFT, padx=5)
        size_var = tk.IntVar(value=3)
        size_spinbox = ttk.Spinbox(size_frame, from_=2, to=10, textvariable=size_var, width=5,
                                   command=lambda: self._update_matrix_grid(parent, size_var.get(), matrix_type))
        size_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(size_frame, text="Crear Matriz", 
                  command=lambda: self._update_matrix_grid(parent, size_var.get(), matrix_type)).pack(side=tk.LEFT, padx=5)
        
        names_frame = ttk.LabelFrame(parent, text="Nombres de Nodos (opcional)", padding="5")
        names_frame.pack(fill=tk.X, pady=5)
        self.node_name_entries = {}
        
        matrix_frame = ttk.LabelFrame(parent, text="Matriz", padding="15")
        matrix_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
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
        
        parent._matrix_canvas = canvas
        parent._matrix_frame = scrollable_frame
        parent._size_var = size_var
        parent._matrix_type = matrix_type
        parent._matrix_entries = {}
        
        self._update_matrix_grid(parent, 3, matrix_type)
    
    def _update_matrix_grid(self, parent, size, matrix_type):
        """Actualiza la grilla de la matriz."""
        for widget in parent._matrix_frame.winfo_children():
            widget.destroy()
        parent._matrix_entries = {}
        
        names_frame = None
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                try:
                    if "Nombres" in widget.cget("text"):
                        names_frame = widget
                        for child in widget.winfo_children():
                            child.destroy()
                        break
                except:
                    pass
        
        if names_frame:
            names_grid = ttk.Frame(names_frame)
            names_grid.pack(fill=tk.X, padx=5, pady=5)
            parent._node_name_entries = {}
            
            for i in range(size):
                ttk.Label(names_grid, text=f"Nodo {i+1}:").grid(row=0, column=i*2, padx=5, pady=2)
                entry = ttk.Entry(names_grid, width=12)
                entry.grid(row=0, column=i*2+1, padx=5, pady=2)
                entry.insert(0, f"Node {i+1}")
                def update_headers(idx=i, entry_widget=entry):
                    def on_change(event=None):
                        self._update_matrix_headers(parent)
                    return on_change
                entry.bind('<KeyRelease>', update_headers())
                parent._node_name_entries[i] = entry
        
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
        
        table_frame = tk.Frame(parent._matrix_frame, bg="#F5F5F5")
        table_frame.grid(row=0, column=0, pady=10)
        
        corner_label = tk.Label(table_frame, text="", width=12, height=2, 
                               relief=tk.RAISED, bg="#2E7D32", borderwidth=2)
        corner_label.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        
        for j in range(size):
            node_name = node_names.get(j, f"Node {j+1}")
            header_label = tk.Label(table_frame, text=node_name[:10], width=12, height=2,
                                   relief=tk.RAISED, bg="#4CAF50", fg="white", 
                                   font=("Arial", 10, "bold"), borderwidth=2)
            header_label.grid(row=0, column=j+1, padx=1, pady=1, sticky="nsew")
        
        for i in range(size):
            node_name = node_names.get(i, f"Node {i+1}")
            row_header = tk.Label(table_frame, text=node_name[:10], width=12, height=2,
                                 relief=tk.RAISED, bg="#4CAF50", fg="white",
                                 font=("Arial", 10, "bold"), borderwidth=2)
            row_header.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            
            for j in range(size):
                cell_frame = tk.Frame(table_frame, bg="#424242", bd=1)
                cell_frame.grid(row=i+1, column=j+1, padx=1, pady=1, sticky="nsew")
                
                entry = tk.Entry(cell_frame, width=10, justify=tk.CENTER, 
                               font=("Arial", 11), relief=tk.FLAT, bd=0,
                               bg="white", fg="black")
                entry.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
                
                if i == j:
                    entry.insert(0, "0")
                    entry.config(state=tk.DISABLED, bg="#E8E8E8", fg="#666666",
                               font=("Arial", 11, "italic"))
                else:
                    entry.insert(0, "0")
                    def make_symmetric(row=i, col=j, entry_widget=entry):
                        def on_change(event=None):
                            try:
                                val = entry_widget.get()
                                if val.strip():
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
        
        for i in range(size + 1):
            table_frame.grid_columnconfigure(i, weight=1, uniform="col")
        for i in range(size + 1):
            table_frame.grid_rowconfigure(i, weight=1, uniform="row")
    
    def _update_matrix_headers(self, parent):
        """Actualiza los encabezados de la matriz con los nombres de nodos."""
        if not hasattr(parent, '_matrix_frame') or not hasattr(parent, '_node_name_entries'):
            return
        
        widgets = parent._matrix_frame.winfo_children()
        if not widgets:
            return
        
        table_frame = None
        for widget in widgets:
            if isinstance(widget, tk.Frame):
                try:
                    grid_info = widget.grid_info()
                    if grid_info:
                        table_frame = widget
                        break
                except:
                    pass
        
        if table_frame:
            for j in range(len(parent._node_name_entries)):
                widget = table_frame.grid_slaves(row=0, column=j+1)
                if widget and isinstance(widget[0], tk.Label):
                    if j in parent._node_name_entries:
                        name = parent._node_name_entries[j].get().strip()
                        widget[0].config(text=name[:10] if name else f"Node {j+1}")
            
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
            
            if hasattr(parent, '_node_name_entries'):
                for i in range(size):
                    if i in parent._node_name_entries:
                        name = parent._node_name_entries[i].get().strip()
                        labels.append(name if name else f"Node {i+1}")
                    else:
                        labels.append(f"Node {i+1}")
            else:
                labels = [f"Node {i+1}" for i in range(size)]
            
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
            algo_type = self.algorithm_type.get()
            
            if algo_type == "ultrametric":
                if self.Mh is None or self.Ml is None:
                    messagebox.showerror("Error", "Debe cargar las matrices Mh y Ml")
                    return
                labels = self.node_labels_mh if self.node_labels_mh else None
                builder = UltrametricTreeBuilder(self.Mh, self.Ml, labels=labels)
                self.steps = builder.build()
                
            elif algo_type == "additive":
                if self.distance_matrix is None:
                    messagebox.showerror("Error", "Debe cargar una matriz de distancias")
                    return
                labels = self.node_labels if self.node_labels else None
                builder = AdditiveTreeBuilder(self.distance_matrix, labels=labels)
                self.steps = builder.build()
            else:
                messagebox.showerror("Error", "Algoritmo no reconocido")
                return
            
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
        
        self.ax.clear()
        
        # Mostrar según el tipo de paso
        if step.get('step') == 3 and 'path_matrix' in step:
            # Paso 3: mostrar matriz de arcos (ultramétrico)
            matrix = step['path_matrix']
            if self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                labels = self.node_labels_mh[:matrix.shape[0]]
            else:
                labels = [f'Node {i+1}' for i in range(matrix.shape[0])]
            draw_matrix(matrix, self.ax, step.get('title', ''), labels)
        elif step.get('step') == 4 and 'calculation_texts' in step:
            # Paso 4: mostrar cálculos como texto formateado (ultramétrico)
            calculation_texts = step['calculation_texts']
            draw_calculation_text(calculation_texts, self.ax, step.get('title', ''))
        elif step.get('step') == 5 and 'cw_result_matrix' in step:
            # Paso 5: mostrar matriz de resultados de Cw (ultramétrico)
            matrix = step['cw_result_matrix']
            if self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                labels = self.node_labels_mh[:matrix.shape[0]]
            else:
                labels = [f'Node {i+1}' for i in range(matrix.shape[0])]
            draw_matrix(matrix, self.ax, step.get('title', ''), labels)
        else:
            graph = step.get('graph') or step.get('tree')
            if graph:
                node_labels = {}
                for node in graph.nodes():
                    label = graph.nodes[node].get('label', None)
                    if label:
                        node_labels[node] = str(label)
                    else:
                        if self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                            if node < len(self.node_labels_mh):
                                node_labels[node] = str(self.node_labels_mh[node])
                            else:
                                node_labels[node] = str(node)
                        elif self.algorithm_type.get() == "additive" and hasattr(self, 'node_labels') and self.node_labels:
                            if node < len(self.node_labels):
                                node_labels[node] = str(self.node_labels[node])
                            else:
                                node_labels[node] = str(node)
                        else:
                            node_labels[node] = str(node)
                
                edge_labels = {}
                for u, v in graph.edges():
                    if 'distance' in graph[u][v]:
                        edge_labels[(u, v)] = f"{graph[u][v]['distance']:.2f}"
                    elif 'weight' in graph[u][v]:
                        edge_labels[(u, v)] = f"{graph[u][v]['weight']:.2f}"
                
                # Para árboles aditivos, siempre usar draw_additive_tree
                # Solo usar draw_additive_tree para árboles ultramétricos
                # Los árboles aditivos usan la visualización normal de grafo
                if step.get('is_ultrametric_tree', False):
                    draw_additive_tree(graph, self.ax, step.get('title', ''), 
                                      node_labels, edge_labels)
                else:
                    # Visualización normal de grafo para árboles aditivos
                    draw_tree(graph, self.ax, step.get('title', ''), 
                             node_labels, edge_labels)
            elif 'matrix' in step:
                matrix = step['matrix']
                if self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                    labels = self.node_labels_mh[:matrix.shape[0]]
                elif self.algorithm_type.get() == "additive" and hasattr(self, 'node_labels') and self.node_labels:
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
                    weight_key = 'weight_Ml' if 'weight_Ml' in edge_info else 'weight_Mh'
                    weight_value = edge_info.get(weight_key, 0)
                    info_text += f"  Arco {edge_label}: Peso Ml={weight_value:.2f}, Cw={edge_info['Cw']:.2f}"
                    if 'pairs_count' in edge_info:
                        info_text += f" (usado por {edge_info['pairs_count']} pares)\n"
                        if 'pairs_str' in edge_info:
                            info_text += f"    Pares: {edge_info['pairs_str']}\n"
                    else:
                        info_text += "\n"
                else:
                    u, v = edge_info['edge']
                    weight_key = 'weight_Ml' if 'weight_Ml' in edge_info else 'weight_Mh'
                    weight_value = edge_info.get(weight_key, 0)
                    info_text += f"  Arco ({u+1}, {v+1}): Peso Ml={weight_value:.2f}, Cw={edge_info['Cw']:.2f}\n"
        
        if 'path_matrix' in step:
            info_text += "\nMatriz de arcos (Paso 3):\n"
            info_text += "Cada celda muestra el arco de mayor peso en el camino entre los nodos.\n"
        
        if 'cw_values' in step and step.get('step') == 5:
            info_text += "\nValores de Cw calculados:\n"
            for edge_key, cw_value in sorted(step['cw_values'].items(), key=lambda x: x[1], reverse=True):
                u, v = edge_key
                if self.algorithm_type.get() == "ultrametric" and hasattr(self, 'node_labels_mh') and self.node_labels_mh:
                    label_u = self.node_labels_mh[u] if u < len(self.node_labels_mh) else f"Node {u+1}"
                    label_v = self.node_labels_mh[v] if v < len(self.node_labels_mh) else f"Node {v+1}"
                else:
                    label_u = f"Node {u+1}"
                    label_v = f"Node {v+1}"
                info_text += f"  ({label_u}{label_v}): {cw_value:.2f}\n"
        
        if 'equation_info' in step:
            eq_info = step['equation_info']
            if 'union_info' in eq_info:
                union = eq_info['union_info']
                info_text += "\n📊 Información de la Unión:\n"
                info_text += f"  Nuevo nodo agregado: {union['new_node']}\n"
                info_text += f"  Arco donde se inserta: {union['edge'][0]} ↔ {union['edge'][1]}\n"
                info_text += f"  Nodo interno creado: {union['internal_node']}\n"
                info_text += "\n  Distancias calculadas:\n"
                for key, dist in union['distances'].items():
                    info_text += f"    {key}: {dist:.2f}\n"
            if 'equations' in eq_info:
                info_text += "\n📐 Sistema de Ecuaciones Resuelto:\n"
                for eq in eq_info['equations']:
                    info_text += f"  {eq}\n"
            elif 'equation' in eq_info:
                info_text += f"\nEcuación: {eq_info['equation']}\n"
        
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", info_text)
        
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
            self.auto_playing = False
            return
        
        def auto_play():
            try:
                while self.auto_playing and self.current_step_index < len(self.steps) - 1:
                    time.sleep(2)
                    if self.auto_playing:
                        try:
                            if self._widget_exists(self.root):
                                self.root.after(0, lambda idx=self.current_step_index + 1: self._safe_show_step(idx))
                            else:
                                break
                        except (tk.TclError, AttributeError):
                            break
            except (tk.TclError, AttributeError):
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
            pass
    
    def update_matrices_display(self):
        """Actualiza el panel de matrices de entrada."""
        for widget in self.matrices_container.winfo_children():
            widget.destroy()
        
        algo_type = self.algorithm_type.get()
        
        if algo_type == "ultrametric":
            if self.Mh is not None and self.Ml is not None:
                mh_frame = ttk.LabelFrame(self.matrices_container, text="Matriz Mh (Distancias)", padding="5")
                mh_frame.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
                
                mh_table = self._create_matrix_table(mh_frame, self.Mh, self.node_labels_mh)
                mh_table.pack(fill=tk.BOTH, expand=True)
                
                ml_frame = ttk.LabelFrame(self.matrices_container, text="Matriz Ml (Pesos)", padding="5")
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
                matrix_frame = ttk.LabelFrame(self.matrices_container, text="Matriz de Distancias", padding="5")
                matrix_frame.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
                
                matrix_table = self._create_matrix_table(matrix_frame, self.distance_matrix, self.node_labels)
                matrix_table.pack(fill=tk.BOTH, expand=True)
                
                self.matrices_container.columnconfigure(0, weight=1)
            else:
                no_data_label = ttk.Label(self.matrices_container, 
                                         text="No hay matriz cargada", 
                                         foreground="gray", font=("Arial", 10))
                no_data_label.grid(row=0, column=0, pady=20)
        
        self.matrices_container.update_idletasks()
        self.matrices_canvas.configure(scrollregion=self.matrices_canvas.bbox("all"))
    
    def _create_matrix_table(self, parent, matrix, labels=None):
        """Crea una tabla visual mejorada para mostrar la matriz."""
        if labels is None:
            labels = [f"Node {i+1}" for i in range(matrix.shape[0])]
        
        outer_frame = tk.Frame(parent)
        outer_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(outer_frame, bg="white", highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(outer_frame, orient="horizontal", command=canvas.xview)
        table_frame = tk.Frame(canvas, bg="white")
        
        canvas.create_window((0, 0), window=table_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set)
        
        canvas.pack(side="top", fill="both", expand=True)
        h_scrollbar.pack(side="bottom", fill="x")
        
        header_frame = tk.Frame(table_frame, bg="#2E7D32")
        header_frame.pack(fill=tk.X)
        
        corner = tk.Label(header_frame, text="", width=6, height=1, 
                         bg="#1B5E20", fg="white", font=("Arial", 8, "bold"))
        corner.pack(side=tk.LEFT, padx=1, pady=1)
        
        for j, label in enumerate(labels):
            header = tk.Label(header_frame, text=str(label)[:6], width=6, height=1,
                            bg="#4CAF50", fg="white", font=("Arial", 8, "bold"),
                            relief=tk.RAISED, bd=1)
            header.pack(side=tk.LEFT, padx=1, pady=1)
        
        for i, label in enumerate(labels):
            row_frame = tk.Frame(table_frame, bg="white")
            row_frame.pack(fill=tk.X)
            
            row_header = tk.Label(row_frame, text=str(label)[:6], width=6, height=1,
                                 bg="#4CAF50", fg="white", font=("Arial", 8, "bold"),
                                 relief=tk.RAISED, bd=1)
            row_header.pack(side=tk.LEFT, padx=1, pady=1)
            
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if isinstance(val, (int, float)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)[:6]
                
                if i == j:
                    bg_color = "#E8E8E8"
                    fg_color = "#666666"
                elif (i + j) % 2 == 0:
                    bg_color = "#FFFFFF"
                    fg_color = "#000000"
                else:
                    bg_color = "#F5F5F5"
                    fg_color = "#000000"
                
                cell = tk.Label(row_frame, text=val_str, width=6, height=1,
                              bg=bg_color, fg=fg_color, font=("Courier", 8),
                              relief=tk.SUNKEN, bd=1, anchor=tk.CENTER)
                cell.pack(side=tk.LEFT, padx=1, pady=1)
        
        def update_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        table_frame.bind("<Configure>", update_scroll)
        
        return outer_frame
    
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
    app = TreeApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nAplicación cerrada.")
    finally:
        if hasattr(app, 'auto_playing'):
            app.auto_playing = False


if __name__ == "__main__":
    main()

