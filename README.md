# Auditoría de equidad de modelos de aprendizaje automático sobre Saber 11

Código y resultados del artículo *Auditoría de equidad de modelos de aprendizaje
automático para predecir bajo desempeño en pruebas estandarizadas en Colombia*,
enviado a la **Revista UIS Ingenierías**.

El trabajo audita un modelo que predice bajo desempeño en la prueba Saber 11 sobre
**2.774.006 estudiantes de 11.230 colegios** (cohortes 2018-2 a 2024-2) y responde
dos preguntas: si eliminar los atributos protegidos del modelo elimina las
disparidades en sus tasas de error, y cuánto cuesta mitigarlas.

**Hallazgo principal.** La disparidad en la tasa de falsos negativos según zona del
colegio es de **0,205**, pero opera **en contra de los estudiantes urbanos**, lo que
refuta la hipótesis preregistrada. Eliminar los atributos protegidos no redujo la
disparidad: clasificadores auxiliares recuperan la zona del colegio con AUC 0,851 a
partir de las variables restantes.

---

## Estructura del repositorio

```
.
├── src/                        código del flujo de auditoría
│   ├── fairness_pipeline.py    módulo central: preprocesamiento, modelos, métricas
│   ├── extraer_cohortes.py     extrae las columnas necesarias de los archivos originales
│   ├── extraer_cohortes.sh     versión en shell, para máquinas con poca memoria
│   ├── consolidar.py           une las cohortes por partes, con techo de memoria
│   ├── run_real.py             ejecución por etapas con puntos de control en disco
│   ├── men_municipal.py        limpieza del CSV municipal del Ministerio de Educación
│   └── verificar_cifras.py     coteja las tablas del artículo contra los CSV
├── figures/
│   └── generar_figuras.py      reconstruye las cinco figuras desde los resultados
├── results/
│   ├── tables/                 CSV con todas las cifras publicadas
│   └── figures/                figuras a 300 ppp
├── paper/
│   ├── build_rui.py            arma el .docx sobre la plantilla de la revista
│   └── rui_contenido.md        texto del artículo con marcas de estructura
├── data/
│   ├── muestra_analitica.parquet   2.774.006 filas, 33 variables (31 MB)
│   └── README.md               qué incluye y cómo obtener los originales
├── requirements.txt
├── LICENSE                     MIT, para el código
└── LICENSE-DATA                CC BY 4.0, para los datos derivados y las figuras
```

---

## Qué datos hay aquí

**Sí está incluida la muestra analítica** (`data/muestra_analitica.parquet`,
31 MB): las 2.774.006 filas y 11.230 colegios que usa el análisis, con las 33
variables del modelo. Con ella se reproducen todas las tablas y figuras del
artículo sin descargar nada más.

**No están los microdatos originales.** Pesan cerca de **9,1 GB** y son propiedad
del Instituto Colombiano para la Evaluación de la Educación (Icfes), que los
publica de forma abierta con sus propios términos de uso. GitHub además rechaza
archivos de más de 100 MB y cada cohorte ronda los 420 MB. El repositorio incluye
el código que los procesa (`src/construir_muestra.py`) y las instrucciones de
descarga en [`data/README.md`](data/README.md).

> El Icfes revisa retroactivamente los archivos publicados. Los resultados de este
> repositorio corresponden a la descarga del **1 de agosto de 2026**. El registro de
> cambios de mayo de 2026 documenta el recálculo de variables en cohortes que se
> remontan a 2016, así que una descarga posterior puede arrojar cifras distintas.

---

## Reproducir los resultados

### Opción A: solo las figuras y la verificación (segundos)

No requiere descargar los microdatos.

```bash
pip install -r requirements.txt
python figures/generar_figuras.py
python src/verificar_cifras.py
```

`verificar_cifras.py` compara cada celda de las tablas del artículo contra los CSV
de `results/tables/` y comprueba la coherencia interna de las métricas. Debe
reportar **0 discrepancias**.

### Opción B: el flujo completo desde los archivos originales (horas)

```bash
# 1. descargue los microdatos siguiendo data/README.md
# 2. extraiga solo las columnas necesarias (evita cargar 8,5 GB en memoria)
python src/extraer_cohortes.py --entrada data/raw --salida data/interim

# 3. consolide las cohortes
python src/consolidar.py --entrada data/interim --salida data/processed/saber11.parquet

# 4. ejecute la auditoría por etapas
python src/run_real.py --datos data/processed/saber11.parquet --salida results/
```

`run_real.py` guarda puntos de control en disco después de cada etapa, de modo que
una interrupción no obliga a empezar de nuevo.

---

## Decisiones metodológicas que conviene conocer antes de reutilizar el código

Estas no son detalles de implementación: cambian los resultados.

**La decisión se toma por capacidad, no por umbral.** Se marca al 20% de mayor
riesgo, que es como se despliega realmente un sistema de asignación de apoyos. Un
umbral fijo de 0,5 sobre una tasa base del 20% no corresponde a ninguna decisión
real y convierte la disparidad medida en un artefacto del parámetro elegido. La
dirección del hallazgo principal depende de esta decisión.

**El remuestreo es por colegio, no por estudiante.** Los estudiantes están anidados
en colegios y sus resultados están correlacionados. Remuestrear estudiantes
individuales produce intervalos demasiado estrechos.

**Los intervalos bootstrap no sirven para declarar significancia aquí.** Los
estadísticos de disparidad del tipo máximo−mínimo son no negativos por construcción,
así que un intervalo percentil casi nunca contiene el cero y declararía
significativa cualquier disparidad. Verificamos este modo de falla sobre un control
negativo: el intervalo ingenuo excluía el cero mientras la disparidad real era
ruido. El código reporta intervalos para la **magnitud** y pruebas de **permutación
a nivel de colegio** para la **significancia**.

**La tasa histórica del colegio se calcula solo con las cohortes de entrenamiento.**
Calcularla sobre la muestra completa filtraría el resultado.

**Dos inconsistencias de codificación del proveedor están homologadas en el código**
y ninguna aparece documentada en el diccionario del Icfes: en la cohorte 2024-2 la
zona del colegio se codifica `URBANA` en 450.514 registros, mientras las cohortes
anteriores usan `URBANO`; y la variable de sexo contiene un tercer valor residual.
Sin homologar, el grupo urbano queda partido en dos justo en la cohorte de prueba.

---

## Limitación que acota todo lo demás

El examen Saber 11 se presenta al final del grado once, así que **los estudiantes
que desertaron antes nunca aparecen en los datos**, y son precisamente quienes están
en mayor riesgo educativo. Los estudiantes rurales son el 15,3%–17,4% de los
evaluados frente a cerca del 26,7% de la población en edad escolar.

La selección opera en la dirección que **atenúa** la disparidad medida, de modo que
las estimaciones de este repositorio deben leerse como una **cota inferior**.

---

## Requisitos

Python 3.10 o superior. Las versiones exactas están en `requirements.txt`.

Las dependencias principales son pandas, scikit-learn, XGBoost 3.2, Fairlearn 0.14
y matplotlib. Los valores de Shapley se calculan con
`booster.predict(dmatrix, pred_contribs=True)` y no con la biblioteca `shap`, que es
incompatible con XGBoost 3.2.

El flujo completo se ejecutó con **3 GB de memoria**, razón por la cual el
preprocesamiento va por cohortes y no en un único paso.

---

## Citación

Si utiliza este código, cite el artículo. Consulte `CITATION.cff`.

---

## Licencia

El código se distribuye bajo licencia **MIT** (`LICENSE`). Los datos derivados de
`results/` y las figuras se distribuyen bajo **CC BY 4.0** (`LICENSE-DATA`).

Los microdatos originales pertenecen al Icfes y se rigen por sus propios términos de
uso. **El Icfes no se hace responsable de los análisis ni de las conclusiones aquí
presentadas**, que son responsabilidad exclusiva del autor.

---

## Autor

Fabricio Echeverría — Corporación Universitaria Asturias, Bogotá, Colombia
ORCID: https://orcid.org/0000-0002-4921-1111

Financiado por la Corporación Universitaria Asturias.
