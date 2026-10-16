"""Funciones del modelo de transformación de histogramas de 8 bits.

La implementación recoge las operaciones centrales de los notebooks
01_exploracion.ipynb y 02_modelo.ipynb, adaptadas para recibir una imagen
como matriz NumPy en lugar de depender de Google Drive/Colab.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline


NIVELES = np.arange(256, dtype=np.int16)


def cargar_imagen_grises(archivo) -> np.ndarray:
    """Convierte una imagen cargada a una matriz de grises uint8 (0–255)."""
    imagen = Image.open(archivo).convert("L")
    return np.asarray(imagen, dtype=np.uint8)


def calcular_histograma(imagen: np.ndarray) -> np.ndarray:
    """Calcula el histograma de 256 niveles de una imagen de grises."""
    imagen = np.asarray(imagen)
    if imagen.ndim != 2:
        raise ValueError("La imagen debe ser una matriz en escala de grises.")
    return np.bincount(imagen.ravel(), minlength=256).astype(float)


def seleccionar_puntos_uniformes(histograma: np.ndarray, numero_puntos: int):
    """Selecciona puntos uniformes sobre el intervalo de intensidades [0,255]."""
    if not 2 <= int(numero_puntos) <= 256:
        raise ValueError("El número de puntos debe estar entre 2 y 256.")
    x = np.round(np.linspace(0, 255, int(numero_puntos))).astype(int)
    x = np.unique(x)
    return x, np.asarray(histograma, dtype=float)[x]


def construir_histograma_interpolado(
    histograma: np.ndarray,
    numero_puntos: int,
    metodo: str,
):
    """Interpola el histograma, recorta valores negativos y conserva su suma.

    Devuelve (histograma_objetivo, x_puntos, y_puntos).
    """
    h = np.asarray(histograma, dtype=float)
    if h.shape != (256,):
        raise ValueError("El histograma debe tener exactamente 256 valores.")
    if not np.all(np.isfinite(h)) or np.any(h < 0) or h.sum() <= 0:
        raise ValueError("El histograma contiene frecuencias no válidas.")

    x_puntos, y_puntos = seleccionar_puntos_uniformes(h, numero_puntos)
    x_eval = np.arange(256)

    if metodo == "Lineal":
        interpolado = np.interp(x_eval, x_puntos, y_puntos)
    elif metodo == "Cúbica":
        spline = CubicSpline(x_puntos, y_puntos)
        interpolado = spline(x_eval)
    else:
        raise ValueError("El método debe ser 'Lineal' o 'Cúbica'.")

    interpolado = np.nan_to_num(interpolado, nan=0.0, posinf=0.0, neginf=0.0)
    interpolado = np.clip(interpolado, 0.0, None)
    suma = interpolado.sum()
    if suma <= 0:
        raise ValueError("La interpolación produjo un histograma vacío.")
    interpolado *= h.sum() / suma
    return interpolado, x_puntos, y_puntos


def histograma_potencia(histograma: np.ndarray, k: float) -> np.ndarray:
    """Aplica H(r)^k mediante normalización probabilística."""
    h = np.asarray(histograma, dtype=float)
    if k <= 0:
        raise ValueError("El parámetro k debe ser mayor que 0.")
    if h.sum() <= 0:
        raise ValueError("El histograma está vacío.")
    p = h / h.sum()
    p_k = np.power(p, float(k))
    suma = p_k.sum()
    if suma <= 0:
        raise ValueError("La potencia produjo un histograma vacío.")
    p_k /= suma
    return p_k * h.sum()


def funcion_peso_dos_intervalos(
    niveles: np.ndarray,
    a1: int,
    b1: int,
    alpha: float,
    a2: int,
    b2: int,
    beta: float,
) -> np.ndarray:
    """Construye W(r) con dos intervalos separados y peso 1 fuera de ellos."""
    if not (0 <= a1 <= b1 <= 255):
        raise ValueError("El primer intervalo debe cumplir 0 ≤ a1 ≤ b1 ≤ 255.")
    if not (0 <= a2 <= b2 <= 255):
        raise ValueError("El segundo intervalo debe cumplir 0 ≤ a2 ≤ b2 ≤ 255.")
    if not b1 < a2:
        raise ValueError("Los intervalos deben estar separados: b1 debe ser menor que a2.")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Los pesos alpha y beta deben ser mayores que cero.")

    W = np.ones_like(niveles, dtype=float)
    W[(niveles >= a1) & (niveles <= b1)] = float(alpha)
    W[(niveles >= a2) & (niveles <= b2)] = float(beta)
    return W


def histograma_k_y_peso(histograma: np.ndarray, W: np.ndarray, k: float) -> np.ndarray:
    """Aplica H_{k,W}(r) proporcional a H(r)^k W(r) y conserva la suma."""
    h = np.asarray(histograma, dtype=float)
    W = np.asarray(W, dtype=float)
    if h.shape != (256,) or W.shape != (256,):
        raise ValueError("El histograma y W deben tener 256 valores.")
    if k <= 0:
        raise ValueError("El parámetro k debe ser mayor que 0.")
    p = h / h.sum()
    p_modificada = np.power(p, float(k)) * W
    suma = p_modificada.sum()
    if suma <= 0:
        raise ValueError("La combinación de k y W produjo un histograma vacío.")
    return (p_modificada / suma) * h.sum()


def funcion_g_lambda(niveles: np.ndarray, lambda_param: float) -> np.ndarray:
    """Función global g_lambda(r) usada para deformar el eje de intensidades."""
    if lambda_param <= 0:
        raise ValueError("lambda debe ser mayor que cero.")
    u = np.asarray(niveles, dtype=float) / 255.0
    numerador = u ** float(lambda_param)
    denominador = numerador + (1.0 - u) ** float(lambda_param)
    return 255.0 * numerador / denominador


def deformar_histograma_eje(histograma_base: np.ndarray, lambda_param: float):
    """Deforma globalmente las posiciones y reparte masa entre niveles vecinos."""
    h = np.asarray(histograma_base, dtype=float)
    if h.shape != (256,):
        raise ValueError("El histograma debe tener exactamente 256 valores.")
    posiciones = funcion_g_lambda(NIVELES, lambda_param)
    nuevo = np.zeros(256, dtype=float)
    for r, frecuencia in enumerate(h):
        posicion = posiciones[r]
        inferior = int(np.floor(posicion))
        superior = min(inferior + 1, 255)
        fraccion = posicion - inferior
        nuevo[inferior] += frecuencia * (1.0 - fraccion)
        nuevo[superior] += frecuencia * fraccion
    if not np.isclose(nuevo.sum(), h.sum()):
        raise RuntimeError("La deformación no conservó la frecuencia total.")
    return posiciones, nuevo


def construir_transformacion_cdf_inversa(
    histograma_original: np.ndarray,
    histograma_objetivo: np.ndarray,
) -> np.ndarray:
    """Correspondencia entre la CDF original y la CDF objetivo (searchsorted)."""
    h0 = np.asarray(histograma_original, dtype=float)
    ht = np.asarray(histograma_objetivo, dtype=float)
    if h0.sum() <= 0 or ht.sum() <= 0:
        raise ValueError("Los histogramas deben tener suma positiva.")
    cdf_original = np.cumsum(h0) / h0.sum()
    cdf_objetivo = np.cumsum(ht) / ht.sum()
    T = np.searchsorted(cdf_objetivo, cdf_original, side="left")
    return np.clip(T, 0, 255).astype(np.uint8)


def construir_transformacion_desde_histograma(histograma_objetivo: np.ndarray) -> np.ndarray:
    """Construye T(r)=redondeo(255·CDF_objetivo(r)), como en el notebook."""
    h = np.asarray(histograma_objetivo, dtype=float)
    if h.sum() <= 0:
        raise ValueError("El histograma objetivo está vacío.")
    cdf = np.cumsum(h) / h.sum()
    return np.clip(np.rint(255.0 * cdf), 0, 255).astype(np.uint8)


def ecualizacion_clasica(imagen: np.ndarray):
    """Ecualización clásica por CDF con resta del primer valor acumulado no nulo."""
    h = calcular_histograma(imagen)
    cdf = np.cumsum(h) / h.sum()
    presentes = np.flatnonzero(h)
    if presentes.size == 0:
        T = np.arange(256, dtype=np.uint8)
    else:
        cdf_min = cdf[presentes[0]]
        denominador = 1.0 - cdf_min
        if denominador <= 0:
            T = np.arange(256, dtype=np.uint8)
        else:
            T = np.rint((cdf - cdf_min) / denominador * 255.0)
            T = np.clip(T, 0, 255).astype(np.uint8)
    return T[imagen], T


def calcular_metricas(imagen: np.ndarray) -> dict[str, float | int]:
    """Métricas descriptivas utilizadas en la exploración inicial."""
    img = np.asarray(imagen, dtype=np.uint8)
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / img.size
    entropia = float(-np.sum(p * np.log2(p))) if p.size else 0.0
    return {
        "Media": float(np.mean(img)),
        "Desviación estándar": float(np.std(img)),
        "Rango dinámico": int(img.max()) - int(img.min()),
        "Entropía (bits)": entropia,
        "Mínimo": int(img.min()),
        "Máximo": int(img.max()),
        "Píxeles": int(img.size),
    }


def procesar_imagen(
    imagen: np.ndarray,
    metodo: str,
    numero_puntos: int,
    modo: str,
    k: float,
    lambda_param: float,
    a1: int,
    b1: int,
    alpha: float,
    a2: int,
    b2: int,
    beta: float,
) -> dict:
    """Ejecuta el flujo seleccionado y devuelve la imagen y datos gráficos."""
    hist_original = calcular_histograma(imagen)
    hist_interpolado, x_puntos, y_puntos = construir_histograma_interpolado(
        hist_original, numero_puntos, metodo
    )
    hist_objetivo = hist_interpolado.copy()
    W = funcion_peso_dos_intervalos(NIVELES, a1, b1, alpha, a2, b2, beta)

    if modo == "Interpolación base (CDF inversa)":
        # Correspondiente a construir_transformacion() en el notebook.
        T = construir_transformacion_cdf_inversa(hist_original, hist_objetivo)
    elif modo == "Interpolación + k":
        hist_objetivo = histograma_potencia(hist_interpolado, k)
        # El notebook construye aquí T directamente desde la CDF objetivo.
        T = construir_transformacion_desde_histograma(hist_objetivo)
    elif modo == "Interpolación + W(r)":
        hist_objetivo = hist_interpolado * W
        hist_objetivo *= hist_original.sum() / hist_objetivo.sum()
        T = construir_transformacion_cdf_inversa(
            hist_original,
            hist_objetivo
        )
    elif modo == "Interpolación + k + W(r)":
        hist_objetivo = histograma_k_y_peso(hist_interpolado, W, k)
        T = construir_transformacion_desde_histograma(hist_objetivo)
    elif modo == "Deformación global del eje con λ":
        _, hist_objetivo = deformar_histograma_eje(hist_interpolado, lambda_param)
        # En el notebook, el experimento global lambda utiliza la CDF inversa.
        T = construir_transformacion_cdf_inversa(hist_original, hist_objetivo)
    else:
        raise ValueError("Selecciona un modo de transformación válido.")

    imagen_resultado = T[imagen]
    hist_resultado_real = calcular_histograma(imagen_resultado)
    imagen_he, T_he = ecualizacion_clasica(imagen)

    return {
        "imagen_original": imagen,
        "imagen_resultado": imagen_resultado,
        "imagen_he": imagen_he,
        "hist_original": hist_original,
        "hist_interpolado": hist_interpolado,
        "hist_objetivo": hist_objetivo,
        "hist_resultado_real": hist_resultado_real,
        "T": T,
        "T_he": T_he,
        "hist_he_real": calcular_histograma(imagen_he),
        "x_puntos": x_puntos,
        "y_puntos": y_puntos,
        "metricas_original": calcular_metricas(imagen),
        "metricas_resultado": calcular_metricas(imagen_resultado),
        "metricas_he": calcular_metricas(imagen_he),
    }
