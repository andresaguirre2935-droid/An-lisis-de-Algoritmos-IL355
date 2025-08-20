import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Algoritmos de búsqueda
def busqueda_lineal(lista, x):
    for i, elem in enumerate(lista):
        if elem == x:
            return i
    return -1

def busqueda_binaria(lista, x):
    izquierda, derecha = 0, len(lista) - 1
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        if lista[medio] == x:
            return medio
        elif lista[medio] < x:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    return -1

# Generación de datos
def generar_datos(size):
    lista = sorted(random.sample(range(1, size * 10), size))
    return lista

# Medición de tiempos promedio
def medir_tiempos():
    sizes = [100, 1000, 10000, 100000]
    resultados = {"Lineal": [], "Binaria": []}

    for size in sizes:
        lista = generar_datos(size)
        x = random.choice(lista)  # garantizamos que existe

        # Búsqueda lineal
        tiempos = []
        for _ in range(5):
            inicio = time.perf_counter()
            busqueda_lineal(lista, x)
            fin = time.perf_counter()
            tiempos.append((fin - inicio) * 1000)
        resultados["Lineal"].append(sum(tiempos) / len(tiempos))

        # Búsqueda binaria
        tiempos = []
        for _ in range(5):
            inicio = time.perf_counter()
            busqueda_binaria(lista, x)
            fin = time.perf_counter()
            tiempos.append((fin - inicio) * 1000)
        resultados["Binaria"].append(sum(tiempos) / len(tiempos))

    return sizes, resultados

# GUI principal
class BusquedaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparación de Búsqueda Lineal y Binaria")

        self.lista = []

        # Frame de configuración
        frame_config = ttk.LabelFrame(root, text="Configuración de Datos")
        frame_config.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_config, text="Tamaño de lista:").pack(side="left", padx=5)
        self.entry_size = ttk.Entry(frame_config, width=10)
        self.entry_size.pack(side="left", padx=5)
        self.entry_size.insert(0, "100")

        self.btn_generar = ttk.Button(frame_config, text="Generar datos", command=self.generar_lista)
        self.btn_generar.pack(side="left", padx=5)

        # Frame de búsqueda
        frame_busqueda = ttk.LabelFrame(root, text="Búsqueda")
        frame_busqueda.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_busqueda, text="Valor a buscar:").pack(side="left", padx=5)
        self.entry_valor = ttk.Entry(frame_busqueda, width=10)
        self.entry_valor.pack(side="left", padx=5)

        self.btn_lineal = ttk.Button(frame_busqueda, text="Búsqueda Lineal", command=self.buscar_lineal)
        self.btn_lineal.pack(side="left", padx=5)

        self.btn_binaria = ttk.Button(frame_busqueda, text="Búsqueda Binaria", command=self.buscar_binaria)
        self.btn_binaria.pack(side="left", padx=5)

        # Resultados
        self.resultado_label = ttk.Label(root, text="Resultados: ")
        self.resultado_label.pack(padx=10, pady=5)

        # Mostrar lista generada
        self.lista_label = tk.Text(root, height=5, wrap="word")
        self.lista_label.pack(fill="x", padx=10, pady=5)

        # Frame para gráfica
        frame_grafica = ttk.LabelFrame(root, text="Comparación de Tiempos Promedio")
        frame_grafica.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafica)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.btn_grafica = ttk.Button(root, text="Actualizar Gráfica", command=self.mostrar_grafica)
        self.btn_grafica.pack(pady=5)

    def generar_lista(self):
        try:
            size = int(self.entry_size.get())
            if size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido para el tamaño de la lista.")
            return

        self.lista = generar_datos(size)
        self.resultado_label.config(text=f"Lista generada con {size} elementos (ordenada).")

        # Mostrar los primeros y últimos 10 elementos de la lista
        preview = self.lista[:10] + ["..."] + self.lista[-10:]
        self.lista_label.delete("1.0", tk.END)
        self.lista_label.insert(tk.END, f"Ejemplo de la lista generada:\n{preview}")

    def buscar(self, algoritmo):
        if not self.lista:
            messagebox.showwarning("Aviso", "Primero genere la lista de datos.")
            return

        try:
            x = int(self.entry_valor.get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido para buscar.")
            return

        inicio = time.perf_counter()
        indice = algoritmo(self.lista, x)
        fin = time.perf_counter()

        if indice != -1:
            resultado = f"Elemento {x} encontrado en índice {indice}."
        else:
            resultado = f"Elemento {x} no encontrado."
        tiempo = (fin - inicio) * 1000
        self.resultado_label.config(text=f"{resultado} Tiempo: {tiempo:.5f} ms")

    def buscar_lineal(self):
        self.buscar(busqueda_lineal)

    def buscar_binaria(self):
        self.buscar(busqueda_binaria)

    def mostrar_grafica(self):
        sizes, resultados = medir_tiempos()

        self.ax.clear()
        self.ax.plot(sizes, resultados["Lineal"], marker="o", label="Lineal")
        self.ax.plot(sizes, resultados["Binaria"], marker="o", label="Binaria")
        self.ax.set_xlabel("Tamaño de la lista")
        self.ax.set_ylabel("Tiempo promedio (ms)")
        self.ax.set_title("Comparación de Algoritmos de Búsqueda")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

# Punto de entrada
if __name__ == "__main__":
    root = tk.Tk()
    app = BusquedaGUI(root)
    root.mainloop()
