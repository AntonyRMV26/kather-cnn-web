# Kather CNN — Aplicación web de clasificación histológica

Aplicación educativa de una página para clasificar parches histológicos
colorrectales en ocho categorías con la **CNN Optimizada** del proyecto.
No es una herramienta clínica.

## Archivos del repositorio

- `app.py`: interfaz de Streamlit.
- `inferencia.py`: preprocesamiento, ejecución de TFLite e interpretación.
- `model_v1_float16.tflite`: modelo **real** exportado desde Colab.
- `class_names.json`: etiquetas originales en su orden exacto.
- `metadata.json`: metadatos originales del modelo y sus resultados.
- `requirements.txt`: dependencias mínimas, sin instalar TensorFlow completo.

## Publicar gratis en Streamlit Community Cloud

1. En GitHub, abre el repositorio público `kather-cnn-web`.
2. Pulsa **Add file > Upload files**. Descomprime primero el ZIP recibido.
   Sube los siete archivos de esta carpeta a la **raíz** del repositorio.
   No subas el `.zip` como único archivo ni una carpeta adicional.
3. Haz **Commit changes** en la rama `main`.
4. En https://share.streamlit.io, elige **Deploy a public app from GitHub**.
5. Repositorio: `AntonyRMV26/kather-cnn-web` (ajustar si tu usuario difiere).
   Rama: `main`. Main file path: `app.py`.
6. En **Advanced settings**, elige **Python 3.12**.
7. Pulsa **Deploy** y espera la instalación de dependencias.
8. Prueba una imagen histológica de 150x150 px. Verifica los 8 nombres
   y que coincida la predicción con el notebook al usar la misma imagen.

## Ejecutar localmente

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Preprocesamiento

Imagen JPG o PNG → RGB → redimensionamiento bilineal a 150x150 → float32
en escala 0..255. **El reescalado 1/255 ya está integrado en el modelo**.

## Procedencia, alcance y limitaciones

El conjunto Kather tiene ocho categorías: ADIPOSE, COMPLEX, DEBRIS,
EMPTY, LYMPHO, MUCOSA, STROMA y TUMOR. El conjunto de prueba de 750
recortes fue separado por imagen, no por paciente o lámina. La métrica
Macro F1 ≈0.9227 corresponde a esa evaluación interna y no justifica
uso clínico. No usar imágenes con información identificable de pacientes.

Código web preparado a partir de la entrega real del grupo; no se han
modificado ni vuelto a entrenar los pesos de la CNN.
