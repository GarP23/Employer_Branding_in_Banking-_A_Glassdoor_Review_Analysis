# 🏦 Bank of America Employer Branding Analysis: Uncovering Turnover Drivers with NLP & AI

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![BERTopic](https://img.shields.io/badge/BERTopic-Topic%20Modeling-00A8E8?style=for-the-badge)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Sentiment%20QC-FF6F61?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Procesamiento del Lenguaje Natural (NLP) aplicado a analítica de personas (*People Analytics*): Transformando 5,000 reseñas de texto libre en Glassdoor en palancas estratégicas de retención de talento.**

---

## 🚀 LinkedIn Executive Pitch (Hook + Solution + Impact)

🔥 **¿Por qué renuncia realmente el talento en la gran banca internacional?** La mayoría de las empresas asume que el salario es el principal factor de rotación. Sin embargo, los datos cuentan una historia diferente.

💡 En este proyecto de **Data Science y NLP**, analizamos un corpus de **5,000 reseñas de Glassdoor (2018–2023)** comparando la marca empleadora de **Bank of America** frente a sus competidores clave (**Citi, HSBC, J.P. Morgan y Wells Fargo**). 

🎯 **El Resultado:** A través de modelos *Transformers* (**DistilBERT**) y modelado de tópicos (*Topic Modeling*) con **BERTopic**, identificamos que el salario actúa únicamente como un factor higiénico, mientras que la **presión por objetivos comerciales en la red** y la **sobrecarga horaria** son los verdaderos detonantes que disparan la fuga de talento voluntaria en un **+4.29 pp** en empleados salientes.

---

## 🛠️ Stack Tecnológico y Arquitectura

El pipeline fue construido íntegramente en Python utilizando tecnologías de vanguardia para NLP y Machine Learning:

| Dominio | Tecnología / Librería | Aplicación En El Proyecto |
| :--- | :--- | :--- |
| **Language Models** | `Transformers` (DistilBERT) | Auditoría y control de calidad de sentimiento en texto libre (*Pros/Cons*). |
| **Topic Modeling** | `BERTopic` + `c-TF-IDF` | Extracción semántica automática de 20 categorías de negocio comparables. |
| **Embeddings & Vector Space** | `SentenceTransformers` (`all-MiniLM-L6-v2`) | Representación vectorial denso de comentarios. |
| **Dimensionality Reduction** | `UMAP` + `HDBSCAN` | Reducción espacial y clustering jerárquico no supervisado. |
| **Data Processing** | `Pandas`, `NumPy` | Normalización, estructuración y cruce estadístico multidimensional. |
| **Visualization** | `Matplotlib`, `Seaborn` | Generación automatizada de gráficos ejecutivos de alta resolución (300 DPI). |

---

## 🔬 Metodología Paso a Paso

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  1. EDA & POSICIONAMIENTO │ ───> │  2. CONTROL DISTILBERT  │ ───> │   3. BERTOPIC CLUSTERS  │
│  5,000 Reseñas Glassdoor│      │ Fiabilidad de Sentimiento│      │   20 Categorías de Red  │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                               │
┌─────────────────────────┐      ┌─────────────────────────┐                   │
│  5. PLAN DE ACCIÓN RRHH │ <─── │ 4. SALIDA: ACTUAL VS EX │ <─────────────────┘
│  Retención vs. Atracción│      │  Drivers de Rotación    │
└─────────────────────────┘      └─────────────────────────┘
```

1. **Análisis Exploratorio de Datos (EDA):** Evaluación del *Overall Rating*, evolutivos históricos por año (2018–2023) y diferencias de valoración según el estado laboral (*Current* vs. *Former*).
2. **Control de Calidad del Sentimiento (DistilBERT):** Auditoría mediante el modelo fine-tuned `distilbert-base-uncased-finetuned-sst-2-english` para medir la consistencia del texto libre frente a la etiqueta asignada (logrando **93.3% de alineación en Pros** y **81.4% en Cons**).
3. **Modelado Temático Multiempresa (BERTopic):** Entrenamiento conjunto para las 5 entidades financieras. Esto garantiza la alineación del espacio vectorial y permite una comparabilidad cruzada estricta entre competidores.
4. **Cruce Cuantitativo de Impacto:** Correlación entre la frecuencia de aparición de cada tópico negativo y el impacto directo en la nota global (*Overall Rating*).
5. **Aislamiento de Drivers de Salida:** Comparativa directa del perfil de quejas entre empleados actuales vs. antiguos empleados (*Former Employees*) en Bank of America.

---

## 📊 Principales Hallazgos y Métricas Clave

> [!IMPORTANT]
> **Posicionamiento Global:** Bank of America se consolida en la **Posición #2 del Ranking de Satisfacción** (Rating promedio: **3,84 / 5.0**), con una curva de mejora ascendente constante desde 2018 (3.60) hasta 2023 (3.98).

### 1. La Brecha Crítica vs. Competidores (*Gaps en pp*)
Al comparar el peso de las quejas frente a la media de sus *peers* (Citi, HSBC, JPM, Wells Fargo):
* **Presión Comercial y Relación con Clientes:** Es el mayor punto débil relativo de Bank of America, situándose **+3,94 pp por encima de sus competidores** (7,53% de las quejas).
* **Funciones de Banca Minorista:** Acumula un exceso del **+2,11 pp** en reclamaciones operativas en la red de oficinas.
* **Salario y Compensación:** Aunque representa el 10,64% de los comentarios negativos, su peso es prácticamente idéntico al promedio sectorial (10,44%), confirmando que es una **fricción estructural de la industria** y no una desventaja propia.

### 2. Destrucción de Valor en el Overall Rating
* Reseñas sin fricciones críticas: **Rating medio de 4.61 / 5.0**.
* Reseñas centradas en *Presión Comercial y Clientes*: **Rating cae a 3.47 / 5.0** (La nota más baja de toda la organización).

### 3. Anatomía de la Fuga de Talento (Actuales vs. Ex-Empleados)
| Categoría Temática | Empleados Actuales | Ex-Empleados | Delta / Incremento | Diagnóstico RRHH |
| :--- | :---: | :---: | :---: | :--- |
| **Clientes y Presión Comercial** | 6,03% | **10,32%** | **+4,29 pp** | 🔴 Driver #1 de Dimisión |
| **Jornada y Sobrecarga Horaria** | 4,48% | **6,88%** | **+2,40 pp** | 🟠 Driver #2 de Dimisión |
| **Desarrollo Profesional en Red** | 3,92% | **5,22%** | **+1,30 pp** | 🟠 Sensación de Estancamiento |
| **Salario y Compensación** | 11,28% | **9,46%** | **-1,82 pp** | ⚪ Factor No Decisivo de Salida |

---

## 🎯 Matriz de Priorización y Plan de Acción de RRHH

```text
    ┌─────────────────────────────────────────────────────────┐
    │          MATRIZ DE ACCIÓN ESTRATÉGICA PARA RRHH         │
    ├──────────────────────────┬──────────────────────────────┤
    │  🔴 RETENCIÓN (Urgente)  │  🟢 ATRACCIÓN (Branding)     │
    │  • Ajuste de métricas    │  • Promocionar mejora de     │
    │    comerciales en red.   │    satisfacción (+0.38 pts). │
    │  • Auditoría de horas    │  • Transparencia salarial    │
    │    extra no asignadas.   │    desde el reclutamiento.   │
    └──────────────────────────┴──────────────────────────────┘
```

1. **Ajuste de Objetivos Comerciales (Prioridad Crítica):** Rediseñar los esquemas de incentivos y metas de venta cruzada en la red de oficinas y centros de atención telefónica para mitigar el estrés operativo.
2. **Planes de Carrera Transparentes (Prioridad Alta):** Crear rutas de promoción interna claras para personal operativo de banca minorista, bloqueando la fuga hacia competidores.
3. **Employer Branding Externo (Atracción):** Utilizar como reclamo de talento la tendencia positiva de satisfacción laboral (+0.38 puntos en el quinquenio) y la solidez institucional.

---

## 📂 Estructura del Repositorio

```text
bank-of-america-employer-branding/
│
├── data/                               # Datasets de trabajo
│   └── glassdoor_reviews.csv           # Corpus de 5,000 reseñas anonimizadas
│
├── notebook/                           # Código del análisis
│   └── Glassdoor_NLP_Analysis.ipynb    # Jupyter Notebook modular e interactivo (Secciones 0-6)
│
├── src/                                # Scripts auxiliares
│   └── generate_slide_assets.py        # Generador automático de imágenes HD para presentaciones
│
├── .gitignore                          # Exclusión de temporales y medios pesados
├── requirements.txt                    # Dependencias y librerías del proyecto
└── README.md                           # Documentación principal
```

---

## 🔗 Recursos Externos y Entregables

Para consultar los entregables ejecutivos del proyecto (presentación visual y demostración):

* 🎬 **[Ver Video Demo / Exposición de la Presentación (Google Drive)](#)** *(Enlace al vídeo grabado)*
* 📊 **[Ver Presentación de Diapositivas en PDF / Slides (Google Drive)](#)** *(Enlace al mazo de 10 diapositivas)*

---

## 💻 Instalación y Ejecución Local

### 1. Clonar el Repositorio
```bash
git clone https://github.com/TU_USUARIO/bank-of-america-employer-branding.git
cd bank-of-america-employer-branding
```

### 2. Crear Entorno Virtual
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar el Notebook
```bash
jupyter notebook notebook/Glassdoor_NLP_Analysis.ipynb
```

---

## 👨‍💻 Autoría y Créditos

Proyecto desarrollado por el equipo de **Data Science & Talent Analytics**.

* **Contacto / LinkedIn:** [Tu Nombre y Apellidos](https://www.linkedin.com)
* **Email:** `tu_email@ejemplo.com`
* **Institución / Máster:** Análisis de Datos y Data Science

---
*Si este repositorio te ha resultado útil o interesante, ¡agradezco enormemente una ⭐️ en GitHub!*