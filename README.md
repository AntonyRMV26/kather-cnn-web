# 🔬 Kather CNN — Clasificador de Tejidos Histológicos

Aplicación web académica que utiliza una **Red Neuronal Convolucional (CNN)** para clasificar imágenes microscópicas de tejidos colorrectales en ocho categorías.

### 🌐 Acceder a la aplicación

**[Abrir Kather CNN](https://kather-cnn-web.streamlit.app/)**

### 📌 ¿Cómo utilizarla?

1. Ingresa al enlace de la aplicación.
2. Pulsa **Upload** y selecciona una imagen histológica en formato JPG o PNG.
3. Haz clic en **Clasificar imagen**.
4. Observa la categoría predicha, su probabilidad estimada y una breve interpretación.

### 🧬 Categorías reconocidas

- Adiposo
- Complejo
- Detritos
- Fondo vacío
- Linfocitos
- Mucosa
- Estroma
- Tumor

### 📊 Modelo y rendimiento

Modelo CNN entrenado con el dataset **Kather (5,000 imágenes, ocho clases balanceadas)** y desplegado mediante TensorFlow Lite y Streamlit.

**Accuracy en el conjunto de prueba: 92.27%.**

### ⚠️ Limitaciones importantes

- Utiliza únicamente imágenes histológicas colorrectales similares a las del dataset Kather.
- No está diseñado para fotografías convencionales, otros tejidos ni imágenes médicas de modalidades diferentes.
- Una probabilidad elevada no garantiza que la clasificación sea correcta.
- El rendimiento indicado corresponde al conjunto de prueba utilizado; no representa una validación clínica con pacientes independientes.
- **Esta aplicación es exclusivamente educativa y no constituye una herramienta de diagnóstico médico.** No subas imágenes con datos personales o clínicos identificables.

---

**Proyecto académico — Ingeniería del Conocimiento | Universidad Nacional del Centro del Perú (UNCP)**
