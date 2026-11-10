"""Inferencia con la CNN Kather exportada a TFLite.

Imagen de entrada: RGB, 150x150, float32 con píxeles 0..255.
La normalización 1/255 está dentro del modelo; NO se repite aquí.
"""
from __future__ import annotations

from pathlib import Path
import json
import threading

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model_v1_float16.tflite"
CLASS_FILE = BASE / "class_names.json"
IMG_SIZE = (150, 150)

ETIQUETAS = {
    "ADIPOSE": "Adiposo",
    "COMPLEX": "Complejo",
    "DEBRIS": "Detritos",
    "EMPTY": "Fondo vacío",
    "LYMPHO": "Linfocitos",
    "MUCOSA": "Mucosa",
    "STROMA": "Estroma",
    "TUMOR": "Tumor",
}

INTERPRETACION = {
    "ADIPOSE": "El modelo reconoce características visuales asociadas con tejido adiposo.",
    "COMPLEX": "El modelo encuentra patrones compatibles con la categoría de tejido complejo del conjunto Kather.",
    "DEBRIS": "El modelo reconoce características visuales correspondientes a la categoría de detritos.",
    "EMPTY": "El modelo encuentra características asociadas a fondo o espacio sin tejido relevante.",
    "LYMPHO": "El modelo reconoce patrones visuales relacionados con infiltrados de linfocitos.",
    "MUCOSA": "El modelo encuentra patrones visuales de mucosa colorrectal.",
    "STROMA": "El modelo reconoce características de tejido estromal o de soporte.",
    "TUMOR": "El modelo reconoce patrones visuales asociados con tejido tumoral en este dataset.",
}


def clases():
    datos = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    codigos = datos["class_names"]
    nombres = datos["labels_es"]
    if len(codigos) != 8 or len(nombres) != 8:
        raise ValueError("El archivo de clases no contiene exactamente 8 categorías.")
    return codigos, nombres


def preparar_imagen(imagen: Image.Image) -> np.ndarray:
    """Coincide con el preprocesamiento del script original de Colab."""
    rgb = imagen.convert("RGB")
    if rgb.size != IMG_SIZE:
        rgb = rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(rgb, dtype=np.float32), axis=0)


def _convertir_entrada(x: np.ndarray, det: dict) -> np.ndarray:
    dtype = det["dtype"]
    if np.issubdtype(dtype, np.floating):
        return x.astype(dtype)
    if np.issubdtype(dtype, np.integer):
        escala, cero = det["quantization"]
        if escala <= 0:
            raise ValueError("Entrada cuantizada sin parámetros de cuantización.")
        limites = np.iinfo(dtype)
        return np.clip(np.round(x / escala + cero), limites.min, limites.max).astype(dtype)
    raise ValueError(f"Tipo de entrada TFLite no compatible: {dtype}")


class Clasificador:
    def __init__(self):
        # Import diferido para mostrar errores claros si falló la instalación.
        from ai_edge_litert.interpreter import Interpreter
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontró {MODEL_PATH.name} en el repositorio.")
        self.interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.codigos, self.nombres = clases()
        self.lock = threading.Lock()
        entrada = tuple(int(v) for v in self.input["shape"])
        salida = tuple(int(v) for v in self.output["shape"])
        if entrada != (1, 150, 150, 3) or salida != (1, 8):
            raise ValueError(f"Forma inesperada del modelo: entrada {entrada}, salida {salida}")

    def predecir(self, imagen: Image.Image) -> dict:
        x = _convertir_entrada(preparar_imagen(imagen), self.input)
        # Streamlit comparte recursos en caché entre sesiones. Interpreter no
        # debe usarse simultáneamente sin sincronización.
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], x)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output["index"]).copy()
        p = pred.reshape(-1).astype(np.float64)
        if np.issubdtype(self.output["dtype"], np.integer):
            escala, cero = self.output["quantization"]
            if escala <= 0:
                raise ValueError("Salida cuantizada sin parámetros.")
            p = (p - cero) * escala
        if len(p) != len(self.codigos) or not np.all(np.isfinite(p)):
            raise ValueError("El modelo devolvió una salida no válida.")
        # TFLite exporta las probabilidades softmax; normalizamos únicamente
        # por una pequeña posible deriva numérica de la exportación float16.
        p = np.maximum(p, 0)
        suma = float(p.sum())
        if suma <= 0:
            raise ValueError("Las probabilidades del modelo son inválidas.")
        p = p / suma
        orden = np.argsort(p)[::-1]
        ganador = int(orden[0])
        return {
            "codigo": self.codigos[ganador],
            "nombre": self.nombres[ganador],
            "probabilidad": float(p[ganador]),
            "explicacion": INTERPRETACION.get(self.codigos[ganador], "Categoría del dataset Kather."),
            "clases": [
                {"codigo":self.codigos[int(i)], "nombre":self.nombres[int(i)], "probabilidad":float(p[int(i)])}
                for i in orden
            ],
        }
