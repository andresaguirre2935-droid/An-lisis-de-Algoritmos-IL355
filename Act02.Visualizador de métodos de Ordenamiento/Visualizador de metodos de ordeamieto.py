import tkinter as tk
from tkinter import ttk
import random, time

# Parámetros generales
ANCHO = 800
ALTO = 300
VAL_MIN, VAL_MAX = 5, 100
RETARDO_MS = 50  # velocidad inicial de animación

# ------------------- Algoritmos paso a paso -------------------

def bubble_sort_steps(data, draw_callback):
    n = len(data)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            draw_callback(activos=[j, j+1]); yield
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                draw_callback(activos=[j, j+1]); yield
    draw_callback(activos=[])

def selection_sort_steps(data, draw_callback):
    n = len(data)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            draw_callback(activos=[i, j, min_idx]); yield
            if data[j] < data[min_idx]:
                min_idx = j
        data[i], data[min_idx] = data[min_idx], data[i]
        draw_callback(activos=[i, min_idx]); yield
    draw_callback(activos=[])

def merge_sort_steps(data, draw_callback):
    aux = data.copy()

    def _merge(lo, mid, hi):
        for k in range(lo, hi + 1):
            aux[k] = data[k]
        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            draw_callback(activos=list(range(lo, hi+1))); yield
            if i > mid:
                data[k] = aux[j]; j += 1
            elif j > hi:
                data[k] = aux[i]; i += 1
            elif aux[j] < aux[i]:
                data[k] = aux[j]; j += 1
            else:
                data[k] = aux[i]; i += 1
            draw_callback(activos=[k]); yield

    def _sort(lo, hi):
        if lo >= hi: return
        mid = (lo + hi) // 2
        yield from _sort(lo, mid)
        yield from _sort(mid+1, hi)
        yield from _merge(lo, mid, hi)

    yield from _sort(0, len(data)-1)
    draw_callback(activos=[])

def quick_sort_steps(data, draw_callback):
    def _partition(lo, hi):
        pivot = data[hi]
        i = lo
        for j in range(lo, hi):
            draw_callback(activos=[j, hi]); yield
            if data[j] <= pivot:
                data[i], data[j] = data[j], data[i]
                draw_callback(activos=[i, j]); yield
                i += 1
        data[i], data[hi] = data[hi], data[i]
        draw_callback(activos=[i]); yield
        return i

    def _qsort(lo, hi):
        if lo < hi:
            p = yield from _partition(lo, hi)
            yield from _qsort(lo, p-1)
            yield from _qsort(p+1, hi)

    yield from _qsort(0, len(data)-1)
    draw_callback(activos=[])

# ------------------- Función de dibujo -------------------

def dibujar_barras(canvas, datos, activos=None):
    canvas.delete("all")
    if not datos: return
    n = len(datos)
    margen = 10
    ancho_disp = ANCHO - 2 * margen
    alto_disp = ALTO - 2 * margen
    w = ancho_disp / n
    esc = alto_disp / max(datos)
    
    for i, v in enumerate(datos):
        x0 = margen + i * w
        x1 = x0 + w * 0.9
        h = v * esc
        y0 = ALTO - margen - h
        y1 = ALTO - margen
        color = "#4e79a7"
        if activos and i in activos:
            color = "#f28e2b"

        # Dibujar barra
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # ---- NUEVO: mostrar valor encima de la barra ----
        canvas.create_text(
            (x0 + x1) / 2, y0 - 8,  # centrado horizontalmente y un poco arriba
            text=str(v),
            fill="black",
            font=("Arial", 8)
        )

    canvas.create_text(6, 6, anchor="nw", text=f"n={len(datos)}", fill="#666")


# ------------------- Aplicación principal -------------------

root = tk.Tk()
root.title("Visualizador de Algoritmos de Ordenamiento")

canvas = tk.Canvas(root, width=ANCHO, height=ALTO, bg="white")
canvas.pack(padx=10, pady=10)

# Estado global
datos = []
algoritmos = {
    "Bubble Sort": bubble_sort_steps,
    "Selection Sort": selection_sort_steps,
    "Merge Sort": merge_sort_steps,
    "Quick Sort": quick_sort_steps
}

# Dropdown de algoritmos
alg_var = tk.StringVar(value="Bubble Sort")
alg_menu = ttk.Combobox(root, textvariable=alg_var, values=list(algoritmos.keys()), state="readonly")
alg_menu.pack(pady=5)

# Entrada para N
frame_n = tk.Frame(root)
frame_n.pack(pady=5)
tk.Label(frame_n, text="N barras:").pack(side="left")
n_var = tk.StringVar(value="40")
n_entry = tk.Entry(frame_n, textvariable=n_var, width=5)
n_entry.pack(side="left", padx=5)

# Scale para velocidad
speed_var = tk.IntVar(value=RETARDO_MS)
tk.Label(root, text="Velocidad (ms):").pack()
speed_scale = tk.Scale(root, from_=0, to=200, orient="horizontal", variable=speed_var)
speed_scale.pack(pady=5)

# Funciones de control
def generar():
    global datos
    try:
        n = int(n_var.get())
    except ValueError:
        n = 40
    n = max(5, min(200, n))
    datos = [random.randint(VAL_MIN, VAL_MAX) for _ in range(n)]
    dibujar_barras(canvas, datos)

def mezclar():
    random.shuffle(datos)
    dibujar_barras(canvas, datos)

def limpiar():
    dibujar_barras(canvas, datos, activos=[])

def ordenar():
    if not datos: return
    algoritmo = algoritmos[alg_var.get()]
    gen = algoritmo(datos, lambda activos=None: dibujar_barras(canvas, datos, activos))
    def paso():
        try:
            next(gen)
            root.after(speed_var.get(), paso)
        except StopIteration:
            pass
    paso()

# Botones
panel = tk.Frame(root)
panel.pack(pady=6)
tk.Button(panel, text="Generar", command=generar).pack(side="left", padx=5)
tk.Button(panel, text="Ordenar", command=ordenar).pack(side="left", padx=5)
tk.Button(panel, text="Mezclar", command=mezclar).pack(side="left", padx=5)
tk.Button(panel, text="Limpiar", command=limpiar).pack(side="left", padx=5)

generar()
root.mainloop()
