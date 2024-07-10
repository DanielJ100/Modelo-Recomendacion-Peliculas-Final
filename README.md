# Proyecto de Recomendación de Contenidos para Plataformas de Streaming

## Introducción

Bienvenido al repositorio del proyecto de recomendación de contenidos para plataformas de streaming. 
Como Data Scientist en una start-up que provee servicios de agregación de plataformas de streaming, 
hemos desarrollado un modelo de recomendación que ofrece métricas prometedoras. 

Sin embargo, enfrentamos un desafío importante: los datos que utilizamos no están normalizados, 
lo que requiere una tarea exhaustiva de maduración de datos. 
Este proyecto aborda todo el ciclo de vida de un proyecto de Machine Learning, desde la recolección y 
tratamiento de datos hasta el entrenamiento y mantenimiento del modelo de ML a medida que llegan nuevos datos.

## Problema

Aunque nuestro modelo de recomendación inicial ha mostrado buenas métricas, los datos de entrada no están 
en un formato uniforme y normalizado.
Esto afecta la calidad de las recomendaciones y la eficiencia del modelo. 
La falta de normalización en los datos se traduce en inconsistencias y ruidos que 
dificultan el proceso de entrenamiento y evaluación del modelo.

## Pasos a Seguir

### 1. Recolección de Datos
- Identificar y recolectar datos de diversas plataformas de streaming.
- Asegurar la calidad y la integridad de los datos recolectados.

### 2. Maduración y Normalización de Datos
- Limpieza de datos: Eliminar valores nulos y duplicados.
- Transformación de datos: Estandarizar los formatos de fecha, texto y numéricos.
- Normalización de características: Escalar valores numéricos para uniformidad.

### 3. Análisis Exploratorio de Datos (EDA)
- Visualización de datos: Crear histogramas, gráficos de barras y diagramas de dispersión.
- Identificación de patrones y tendencias en los datos.

### 4. Preparación de Datos para el Modelado
- Creación de características combinadas relevantes para el modelo de recomendación.
- División del dataset en conjuntos de entrenamiento y prueba.

### 5. Entrenamiento del Modelo de Recomendación
- Implementar técnicas de vectorización de texto (TF-IDF).
- Calcular similitudes coseno entre los contenidos.
- Entrenar el modelo utilizando técnicas de aprendizaje supervisado o no supervisado.

### 6. Evaluación del Modelo
- Medir el rendimiento del modelo utilizando métricas apropiadas (precisión, recall, F1-score).
- Realizar validación cruzada para asegurar la robustez del modelo.

### 7. Implementación y Mantenimiento del Modelo
- Desplegar el modelo en un entorno de producción.
- Establecer un pipeline de actualización para incorporar nuevos datos.
- Monitorear el rendimiento del modelo y realizar ajustes necesarios.

## Solución a Entregar

La solución final incluirá un modelo de recomendación optimizado y un pipeline de datos 
bien estructurado que asegure la normalización y la calidad continua de los datos. 
Se proporcionará un entorno de producción donde el modelo puede recibir nuevos datos 
y actualizarse automáticamente, manteniendo la eficiencia y precisión de las recomendaciones.

---
