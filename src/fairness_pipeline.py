"""
Auditoria de equidad de modelos predictivos de bajo desempeno estandarizado
Microdatos Saber 11 - ICFES, Colombia

VERSION 2 - incorpora las correcciones de la revision critica:
  A3  bootstrap por conglomerado de colegio (los estudiantes se anidan en colegios)
  A4  umbral por capacidad institucional en lugar de 0.5
  A2  escenarios E2 (reponderacion) y E3 (gradiente exponenciado) implementados
      + SHAP por grupo + ajuste real de hiperparametros
  B3  eje y metrica primarios preregistrados; el resto es exploratorio
  B4  cohortes de pandemia excluidas del analisis principal
  B6  la ausencia diferencial de datos se analiza, no solo se tabula
  C4  imports limpios
  C5  clase positiva del detector de proxies explicita

Uso con datos reales:
    python fairness_pipeline.py --data ruta/saber11/*.txt --out resultados/

Modo verificacion (datos sinteticos; NO produce hallazgos cientificos):
    python fairness_pipeline.py --synthetic 30000 --out verificacion/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# --- Preregistro -------------------------------------------------------------
# Un solo eje y una sola metrica primarios. Todo lo demas es exploratorio y se
# reporta como tal, lo que evita el problema de comparaciones multiples.
EJE_PRIMARIO = "zona"
METRICA_PRIMARIA = "dif_fnr"
CAPACIDAD_PRINCIPAL = 0.20      # la institucion puede atender al 20% de mayor riesgo
COHORTES_PANDEMIA = {"2020", "2021"}


# ----------------------------------------------------------------------------
# 1. ESQUEMA
# ----------------------------------------------------------------------------

ORDINAL_MAPS = {
    "fami_estratovivienda": {
        "Sin Estrato": 0, "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
        "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6,
    },
    "fami_educacionmadre": {
        "Ninguno": 0, "No sabe": np.nan, "No Aplica": np.nan,
        "Primaria incompleta": 1, "Primaria completa": 2,
        "Secundaria (Bachillerato) incompleta": 3,
        "Secundaria (Bachillerato) completa": 4,
        "Tecnica o tecnologica incompleta": 5, "Técnica o tecnológica incompleta": 5,
        "Tecnica o tecnologica completa": 6, "Técnica o tecnológica completa": 6,
        "Educacion profesional incompleta": 7, "Educación profesional incompleta": 7,
        "Educacion profesional completa": 8, "Educación profesional completa": 8,
        "Postgrado": 9,
    },
    # CORREGIDO contra el Diccionario DataIcfes (abril 2026), periodo 2014-2 a 2025-2.
    # Las categorias son RANGOS, no numeros escritos. La version anterior de este
    # mapa usaba "Dos"/"Tres"/"Cuatro", que corresponden a periodos ANTERIORES a
    # 2014-2: sobre las cohortes 2018-2024 habria producido NaN en casi todas las
    # filas, la variable habria quedado imputada por la mediana y el feature
    # habria muerto en silencio.
    "fami_personashogar": {
        "1a2": 1.5, "1 a 2": 1.5, "3a4": 3.5, "3 a 4": 3.5,
        "5a6": 5.5, "5 a 6": 5.5, "7a8": 7.5, "7 a 8": 7.5,
        "9 o mas": 9.5, "9 o más": 9.5,
    },
    "fami_cuartoshogar": {   # verificado: si son numeros escritos
        "Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5,
        "Seis o mas": 6, "Seis o más": 6,
    },
    "fami_numlibros": {      # verificado: viene en MAYUSCULAS
        "0 A 10 LIBROS": 5, "11 A 25 LIBROS": 18, "26 A 100 LIBROS": 63,
        "MÁS DE 100 LIBROS": 150, "MAS DE 100 LIBROS": 150,
    },
    "estu_horassemanatrabaja": {
        "0": 0, "Menos de 10 horas": 5, "Entre 11 y 20 horas": 15,
        "Entre 21 y 30 horas": 25, "Más de 30 horas": 35, "Mas de 30 horas": 35,
    },
    "fami_situacioneconomica": {"Peor": 0, "Igual": 1, "Mejor": 2},
}
ORDINAL_MAPS["fami_educacionpadre"] = ORDINAL_MAPS["fami_educacionmadre"]

# Verificado en el diccionario: todas responden No / Si
BINARY_COLS = ["fami_tieneinternet", "fami_tienecomputador",
               "fami_tieneautomovil", "fami_tienelavadora",
               "fami_tienemotocicleta", "fami_tieneserviciotv",
               "fami_tieneconsolavideojuegos", "fami_tienehornomicroogas"]

CATEGORICAL_COLS = ["cole_calendario", "cole_caracter", "cole_jornada",
                    "cole_genero", "cole_bilingue", "cole_depto_ubicacion",
                    "fami_trabajolabormadre", "fami_trabajolaborpadre",
                    "estu_dedicacionlecturadiaria", "estu_dedicacioninternet"]

NUMERIC_COLS = ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
                "fami_personashogar", "fami_cuartoshogar", "fami_numlibros",
                "fami_situacioneconomica", "estu_horassemanatrabaja",
                "hacinamiento", "tasa_hist_colegio"] + BINARY_COLS

PROTECTED = {"zona": "cole_area_ubicacion", "naturaleza": "cole_naturaleza",
             "estrato_grupo": "estrato_grupo", "sexo": "estu_genero"}

# Clase positiva EXPLICITA del detector de proxies (correccion C5)
PROXY_POSITIVE = {"zona": "RURAL", "naturaleza": "OFICIAL", "sexo": "F"}

CLUSTER_COL = "cole_cod_dane_establecimiento"   # unidad de conglomerado
TARGET = "bajo_desempeno"


# ----------------------------------------------------------------------------
# 2. CARGA
# ----------------------------------------------------------------------------

def load_real(paths: list[str]) -> pd.DataFrame:
    frames, failed = [], []
    for p in paths:
        for sep, enc in [("|", "utf-8"), (";", "utf-8"), ("|", "latin-1"), (",", "utf-8")]:
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    df.columns = [c.strip().lower() for c in df.columns]
                    frames.append(df)
                    break
            except Exception:
                continue
        else:
            failed.append(p)
    if failed:
        print(f"[AVISO] No se pudieron leer {len(failed)} archivos: {failed}")
    if not frames:
        raise RuntimeError("Ningun archivo pudo leerse.")
    common = set.intersection(*[set(f.columns) for f in frames])
    union = set.union(*[set(f.columns) for f in frames])
    if union - common:
        print(f"[AVISO] {len(union - common)} columnas ausentes en alguna cohorte, excluidas.")
    return pd.concat([f[list(common)] for f in frames], ignore_index=True)


def make_synthetic(n: int = 30_000) -> pd.DataFrame:
    """Datos con el esquema real y una disparidad CONOCIDA inyectada.

    Controles del disenio:
      - zona, estrato y naturaleza SI tienen efecto  -> la auditoria debe detectarlo
      - sexo NO tiene efecto                          -> la auditoria NO debe detectarlo
    """
    zona = rng.choice(["URBANO", "RURAL"], n, p=[0.78, 0.22])
    rural = (zona == "RURAL").astype(int)
    estrato = np.where(rural == 1,
                       rng.choice([1, 2, 3, 4], n, p=[0.45, 0.38, 0.14, 0.03]),
                       rng.choice([1, 2, 3, 4, 5, 6], n,
                                  p=[0.16, 0.32, 0.28, 0.14, 0.06, 0.04]))
    naturaleza = np.where(rng.random(n) < np.where(rural == 1, 0.93, 0.68),
                          "OFICIAL", "NO OFICIAL")
    sexo = rng.choice(["F", "M"], n, p=[0.53, 0.47])

    # Conglomerados: ~60 estudiantes por colegio, correlacion intra-colegio real
    n_col = max(int(n / 60), 20)
    colegio = rng.integers(0, n_col, n)
    efecto_colegio = rng.normal(0, 18, n_col)[colegio]

    em = np.clip(rng.normal(4.6 - 1.5 * rural + 0.45 * estrato, 2.0, n), 0, 9).round()
    ep = np.clip(rng.normal(4.4 - 1.4 * rural + 0.42 * estrato, 2.0, n), 0, 9).round()
    per = np.clip(rng.normal(4.4 + 0.7 * rural - 0.15 * estrato, 1.5, n), 1, 12).round()
    cua = np.clip(rng.normal(3.1 - 0.6 * rural + 0.3 * estrato, 1.1, n), 1, 6).round()

    internet = np.where(rng.random(n) < np.clip(0.22 + 0.13 * estrato - 0.26 * rural, .02, .98), "Si", "No")
    compu = np.where(rng.random(n) < np.clip(0.26 + 0.12 * estrato - 0.22 * rural, .02, .98), "Si", "No")
    auto = np.where(rng.random(n) < np.clip(0.05 + 0.14 * estrato - 0.10 * rural, .01, .98), "Si", "No")
    lava = np.where(rng.random(n) < np.clip(0.40 + 0.11 * estrato - 0.20 * rural, .02, .99), "Si", "No")

    ruido = rng.normal(0, 40 + 12 * rural, n)   # heterocedastico: peor senal en rural
    punt = (228 + 9.5 * estrato + 3.6 * em + 2.9 * ep
            + 11.0 * (internet == "Si") + 8.0 * (compu == "Si")
            - 2.2 * per + 3.0 * cua - 14.0 * rural
            + 12.0 * (naturaleza == "NO OFICIAL") + efecto_colegio + ruido)

    return pd.DataFrame({
        "periodo": rng.choice(["20184", "20194", "20224", "20234", "20244"], n),
        CLUSTER_COL: colegio.astype(str),
        "cole_area_ubicacion": zona, "cole_naturaleza": naturaleza, "estu_genero": sexo,
        "cole_calendario": rng.choice(["A", "B"], n, p=[.96, .04]),
        "cole_caracter": rng.choice(["ACADÉMICO", "TÉCNICO", "TÉCNICO/ACADÉMICO", "NO APLICA"], n),
        "cole_jornada": rng.choice(["MAÑANA", "TARDE", "COMPLETA", "UNICA", "NOCHE"], n),
        "cole_genero": rng.choice(["MIXTO", "FEMENINO", "MASCULINO"], n, p=[.9, .05, .05]),
        "cole_bilingue": rng.choice(["N", "S"], n, p=[.95, .05]),
        "cole_depto_ubicacion": rng.choice(
            ["BOGOTA", "ANTIOQUIA", "VALLE", "HUILA", "CHOCO", "NARINO"], n),
        "fami_estratovivienda": estrato, "fami_educacionmadre": em,
        "fami_educacionpadre": ep, "fami_personashogar": per, "fami_cuartoshogar": cua,
        "fami_tieneinternet": internet, "fami_tienecomputador": compu,
        "fami_tieneautomovil": auto, "fami_tienelavadora": lava,
        "punt_global": np.clip(punt, 0, 500).round(),
    })


# ----------------------------------------------------------------------------
# 3. PREPROCESAMIENTO
# ----------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, quantile: float = 0.20,
               incluir_pandemia: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # B4: fuera cohortes de pandemia en el analisis principal
    anio = df["periodo"].astype(str).str[:4]
    if not incluir_pandemia:
        n0 = len(df)
        df = df[~anio.isin(COHORTES_PANDEMIA)]
        if len(df) < n0:
            print(f"[INFO] Excluidas {n0-len(df):,} filas de cohortes 2020-2021.")

    # ------------------------------------------------------------------
    # Normalizacion de etiquetas: correcciones halladas en los datos REALES,
    # ninguna de ellas documentada en el diccionario del Icfes.
    # ------------------------------------------------------------------

    # (1) En la cohorte 2024-2 la zona pasa a publicarse como "URBANA"; en todas
    #     las anteriores es "URBANO". Sin unificar, el grupo urbano se parte en
    #     dos y el eje primario del estudio queda inservible justo en la cohorte
    #     de prueba.
    if "cole_area_ubicacion" in df.columns:
        z = df["cole_area_ubicacion"].astype(str).str.strip().str.upper()
        df["cole_area_ubicacion"] = z.replace({"URBANA": "URBANO"})
        df.loc[~df["cole_area_ubicacion"].isin(["URBANO", "RURAL"]),
               "cole_area_ubicacion"] = np.nan

    # (2) estu_genero aparece como "F"/"M", como "F - Femenino" y con un tercer
    #     valor "N" residual (decenas de casos). Solo F y M son interpretables.
    if "estu_genero" in df.columns:
        g = df["estu_genero"].astype(str).str.strip().str[0].str.upper()
        df["estu_genero"] = g.where(g.isin(["F", "M"]), np.nan)

    if "cole_naturaleza" in df.columns:
        n = df["cole_naturaleza"].astype(str).str.strip().str.upper()
        df["cole_naturaleza"] = n.where(n.isin(["OFICIAL", "NO OFICIAL"]), np.nan)

    # (3) Sin colegio identificado = inscritos que presentan de forma particular.
    #     No tienen contexto escolar y el colegio es la unidad de conglomerado,
    #     asi que quedan fuera del universo del estudio. Se reporta cuantos son.
    if "cole_area_ubicacion" in df.columns:
        sin_colegio = df["cole_area_ubicacion"].isna()
        if sin_colegio.any():
            print(f"[INFO] Excluidos {sin_colegio.sum():,} inscritos sin colegio "
                  f"identificado ({sin_colegio.mean():.1%}): presentaron de forma "
                  f"individual, no tienen contexto escolar.")
            df = df[~sin_colegio].copy()

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns and df[col].dtype == object:
            crudo = df[col].astype(str).str.strip()
            mapped = crudo.map(mapping)
            # ALERTA de homologacion: si el mapa no cubre las categorias reales,
            # la variable quedaria vacia y seria imputada sin que nadie lo note.
            # Solo alerta por categorias DESCONOCIDAS. "No sabe" y "No Aplica"
            # estan en el mapa apuntando a NaN a proposito: son ausencia real.
            no_nulos = crudo.ne("nan") & crudo.ne("")
            desconocidas = set(crudo[no_nulos]) - set(mapping)
            if desconocidas:
                prop = float(crudo[no_nulos].isin(desconocidas).mean())
                print(f"[ALERTA] {col}: {prop:.1%} en categorias NO reconocidas "
                      f"{sorted(desconocidas)[:8]} -> revise el diccionario de esa cohorte")
            df[col] = mapped

    for col in BINARY_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                       .isin(["si", "sí", "s"])).astype(int)

    df["punt_global"] = pd.to_numeric(df["punt_global"], errors="coerce")
    df = df.dropna(subset=["punt_global"]).reset_index(drop=True)

    if {"fami_personashogar", "fami_cuartoshogar"}.issubset(df.columns):
        df["hacinamiento"] = df["fami_personashogar"] / df["fami_cuartoshogar"].replace(0, np.nan)

    df[TARGET] = (df.groupby("periodo")["punt_global"]
                    .transform(lambda s: (s <= s.quantile(quantile)).astype(int)))

    # "Sin Estrato" se codifica como 0 pero NO es equivalente a estrato bajo:
    # designa viviendas sin estratificacion, tipicamente rurales o informales,
    # y agruparlo con los estratos 1-2 borraria justo al grupo mas vulnerable.
    # Se mantiene como categoria propia.
    est = df["fami_estratovivienda"]
    df["estrato_grupo"] = np.select(
        [est == 0, est.between(1, 2), est.between(3, 4), est > 4],
        ["Sin estrato", "Bajo", "Medio", "Alto"], default="Desconocido")
    if CLUSTER_COL not in df.columns:
        df[CLUSTER_COL] = "sin_colegio"
    return df


def add_school_history(tr: pd.DataFrame, others: list[pd.DataFrame]) -> None:
    """Tasa historica de bajo desempeno del colegio.

    Se calcula SOLO con cohortes de entrenamiento y se propaga a validacion y
    prueba: calcularla sobre todo el conjunto seria fuga de informacion.
    """
    hist = tr.groupby(CLUSTER_COL)[TARGET].mean()
    global_rate = tr[TARGET].mean()
    for d in [tr] + others:
        d["tasa_hist_colegio"] = d[CLUSTER_COL].map(hist).fillna(global_rate)


def missingness_analysis(df: pd.DataFrame, out: Path) -> dict:
    """B6: la ausencia diferencial es un hallazgo, no un tramite."""
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    zona = PROTECTED["zona"]
    rep = df.groupby(zona)[cols].apply(lambda g: g.isna().mean()).T
    rep.columns = [f"ausencia_{c}" for c in rep.columns]
    if rep.shape[1] == 2:
        rep["diferencia"] = rep.iloc[:, 1] - rep.iloc[:, 0]
    rep.sort_values(rep.columns[-1], ascending=False).to_csv(out / "tabla_ausencia_por_zona.csv")
    return {"ausencia_media_por_zona": df.groupby(zona)[cols].apply(
        lambda g: float(g.isna().mean().mean())).to_dict()}


def build_matrix(df: pd.DataFrame, include_protected: bool):
    num = [c for c in NUMERIC_COLS if c in df.columns]
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    if include_protected:
        cat = cat + [c for c in (PROTECTED["zona"], PROTECTED["naturaleza"],
                                 PROTECTED["sexo"]) if c in df.columns]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat),
    ])
    return df[num + cat], pre


# ----------------------------------------------------------------------------
# 4. UMBRAL POR CAPACIDAD  (correccion A4)
# ----------------------------------------------------------------------------

def predict_at_capacity(proba: np.ndarray, capacity: float) -> np.ndarray:
    """Marca al `capacity` proporcion de mayor riesgo.

    Asi es como se despliega un sistema de alerta temprana: la institucion
    tiene cupos para k estudiantes. Un umbral fijo de 0.5 sobre una tasa base
    del 20% no corresponde a ninguna decision real y hace que la disparidad
    dependa de un parametro arbitrario.
    """
    k = max(int(round(capacity * len(proba))), 1)
    idx = np.argsort(-proba)[:k]
    pred = np.zeros(len(proba), dtype=int)
    pred[idx] = 1
    return pred


# ----------------------------------------------------------------------------
# 5. METRICAS DE EQUIDAD
# ----------------------------------------------------------------------------

def group_rates(y_true, y_pred, groups) -> pd.DataFrame:
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < 50:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        rows.append({"grupo": g, "n": int(m.sum()),
                     "tasa_base": float(y_true[m].mean()),
                     "tasa_seleccion": float(y_pred[m].mean()),
                     "tpr": tp / (tp + fn) if (tp + fn) else np.nan,
                     "fnr": fn / (tp + fn) if (tp + fn) else np.nan,
                     "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
                     "ppv": tp / (tp + fp) if (tp + fp) else np.nan})
    return pd.DataFrame(rows)


def disparities(tab: pd.DataFrame) -> dict:
    if len(tab) < 2:
        return {}
    return {"dif_paridad_demografica": float(tab["tasa_seleccion"].max() - tab["tasa_seleccion"].min()),
            "dif_igualdad_oportunidad": float(tab["tpr"].max() - tab["tpr"].min()),
            "dif_fnr": float(tab["fnr"].max() - tab["fnr"].min()),
            "dif_fpr": float(tab["fpr"].max() - tab["fpr"].min()),
            "dif_ppv": float(tab["ppv"].max() - tab["ppv"].min()),
            "probabilidades_igualadas": float(max(tab["tpr"].max() - tab["tpr"].min(),
                                                  tab["fpr"].max() - tab["fpr"].min()))}


def permutation_pvalue(y_true, y_pred, groups, clusters=None,
                       stat=METRICA_PRIMARIA, n_perm=500) -> float:
    """p-valor por permutacion, permutando COLEGIOS completos.

    Los estadisticos max-min son no negativos por construccion, asi que un IC
    percentil por bootstrap casi nunca incluye el cero y declararia significativa
    cualquier disparidad. La permutacion construye la distribucion nula correcta.
    Permutar a nivel de colegio preserva la estructura de conglomerado.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    obs = disparities(group_rates(y_true, y_pred, groups)).get(stat, np.nan)
    if not np.isfinite(obs):
        return float("nan")

    if clusters is not None:
        clusters = np.asarray(clusters)
        uc = pd.unique(clusters)
        # etiqueta de grupo por colegio (moda), que es lo que se permuta
        gpc = pd.Series(groups).groupby(pd.Series(clusters)).agg(
            lambda s: s.value_counts().index[0])
        null = []
        for _ in range(n_perm):
            shuffled = pd.Series(rng.permutation(gpc.values), index=gpc.index)
            gp = shuffled.reindex(clusters).values
            v = disparities(group_rates(y_true, y_pred, gp)).get(stat, np.nan)
            if np.isfinite(v):
                null.append(v)
    else:
        null = []
        for _ in range(n_perm):
            v = disparities(group_rates(y_true, y_pred, rng.permutation(groups))).get(stat, np.nan)
            if np.isfinite(v):
                null.append(v)

    if not null:
        return float("nan")
    return float((np.sum(np.asarray(null) >= obs) + 1) / (len(null) + 1))


def cluster_bootstrap(y_true, y_pred, groups, clusters, n_boot=1000,
                      alpha=0.05, n_perm=500) -> dict:
    """Bootstrap por CONGLOMERADO (correccion A3).

    Los estudiantes se anidan en colegios y sus resultados estan correlacionados.
    Remuestrear estudiantes individuales trata observaciones dependientes como
    independientes y produce intervalos demasiado estrechos. Aqui se remuestrean
    colegios completos con reemplazo.
    """
    y_true, y_pred, groups, clusters = map(np.asarray, (y_true, y_pred, groups, clusters))
    idx_por_cluster = pd.Series(np.arange(len(y_true))).groupby(pd.Series(clusters)).apply(np.array)
    claves = idx_por_cluster.index.values

    acc: dict[str, list] = {}
    for _ in range(n_boot):
        sel = rng.choice(len(claves), len(claves), replace=True)
        idx = np.concatenate([idx_por_cluster.iloc[s] for s in sel])
        d = disparities(group_rates(y_true[idx], y_pred[idx], groups[idx]))
        for k, v in d.items():
            acc.setdefault(k, []).append(v)

    primarias = {METRICA_PRIMARIA, "dif_igualdad_oportunidad", "dif_paridad_demografica"}
    out = {}
    for k, vals in acc.items():
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        entry = {"punto": float(np.nanmean(vals)), "ic_inf": float(lo), "ic_sup": float(hi)}
        if k in primarias:
            p = permutation_pvalue(y_true, y_pred, groups, clusters, stat=k, n_perm=n_perm)
            entry["p_permutacion"] = p
            entry["significativo"] = bool(np.isfinite(p) and p < alpha)
        out[k] = entry
    return out


# ----------------------------------------------------------------------------
# 6. PROXIES
# ----------------------------------------------------------------------------

def proxy_detection(df: pd.DataFrame, eje: str) -> dict:
    col, pos = PROTECTED[eje], PROXY_POSITIVE.get(eje)
    if col not in df.columns or pos is None:
        return {}
    from sklearn.model_selection import train_test_split
    X, pre = build_matrix(df, include_protected=False)
    y = (df[col].astype(str) == pos).astype(int)
    if y.nunique() < 2:
        return {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3,
                                          random_state=RANDOM_STATE, stratify=y)
    clf = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1500))]).fit(Xtr, ytr)
    return {"clase_positiva": pos,
            "auc": float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))}


# ----------------------------------------------------------------------------
# 7. MODELOS Y AJUSTE DE HIPERPARAMETROS (correccion B5)
# ----------------------------------------------------------------------------

def model_grid():
    grid = {
        "regresion_logistica": (LogisticRegression(max_iter=2000),
                                [{"C": c} for c in (0.05, 0.3, 1.0, 3.0)]),
        "random_forest": (RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
                          [{"n_estimators": 300, "min_samples_leaf": ml, "max_depth": md}
                           for ml in (10, 40) for md in (12, None)]),
    }
    try:
        from xgboost import XGBClassifier
        grid["xgboost"] = (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                          n_jobs=-1, tree_method="hist"),
            [{"n_estimators": ne, "max_depth": md, "learning_rate": lr,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}
             for ne in (300, 700) for md in (4, 8) for lr in (0.05, 0.15)])
    except ImportError:
        print("[AVISO] xgboost no disponible.")
    return grid


def tune_on_validation(tr, va, pre, Xtr, Xva, ytr, yva) -> dict:
    """Ajuste real sobre la cohorte de validacion.

    Sin esto, reportar que el modelo lineal supera a gradient boosting no es un
    hallazgo sino un artefacto de no haber ajustado nada.
    """
    best = {}
    for name, (est, params) in model_grid().items():
        mejor_auc, mejor_p = -np.inf, None
        for p in params:
            m = est.__class__(**{**est.get_params(), **p})
            try:
                pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
                auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
            except Exception:
                continue
            if auc > mejor_auc:
                mejor_auc, mejor_p = auc, p
        if mejor_p is not None:
            best[name] = {"params": mejor_p, "auc_val": float(mejor_auc)}
            print(f"  [tuning] {name:22s} AUC_val={mejor_auc:.4f}  {mejor_p}")
    return best


def build_estimator(name: str, params: dict):
    est, _ = model_grid()[name]
    return est.__class__(**{**est.get_params(), **params})


def evaluate(y_true, proba, pred) -> dict:
    return {"auc_roc": float(roc_auc_score(y_true, proba)),
            "auc_pr": float(average_precision_score(y_true, proba)),
            "brier": float(brier_score_loss(y_true, proba)),
            "sensibilidad": float(((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
            "especificidad": float(((pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))}


# ----------------------------------------------------------------------------
# 8. MITIGACION
# ----------------------------------------------------------------------------

class WeightedPipeline(BaseEstimator, ClassifierMixin):
    """Adaptador que enruta `sample_weight` al paso final del Pipeline.

    Necesario para E3: ExponentiatedGradient reentrena el estimador con pesos
    en cada iteracion y llama fit(X, y, sample_weight=...), firma que
    sklearn.Pipeline no acepta directamente.
    """

    def __init__(self, pre=None, estimator=None):
        self.pre = pre
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        from sklearn.base import clone
        self.pre_ = clone(self.pre)
        Xt = self.pre_.fit_transform(X, y)
        self.estimator_ = clone(self.estimator)
        if sample_weight is not None:
            self.estimator_.fit(Xt, y, sample_weight=sample_weight)
        else:
            self.estimator_.fit(Xt, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(self.pre_.transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.pre_.transform(X))


def reweighing_weights(y, a) -> np.ndarray:
    """Reponderacion de Kamiran-Calders (E2, preprocesamiento).

    w(a,y) = P(A=a) P(Y=y) / P(A=a, Y=y). Elimina la dependencia entre atributo
    protegido y etiqueta en la distribucion ponderada de entrenamiento.
    """
    y, a = np.asarray(y), np.asarray(a).astype(str)
    n = len(y)
    w = np.ones(n, float)
    for av in np.unique(a):
        for yv in np.unique(y):
            m = (a == av) & (y == yv)
            if m.sum() == 0:
                continue
            w[m] = ((a == av).mean() * (y == yv).mean()) / (m.sum() / n)
    return w


# ----------------------------------------------------------------------------
# 9. EJECUCION
# ----------------------------------------------------------------------------

def audit(y, pred, df_te, out: Path, n_boot: int, n_perm: int, etiqueta: str) -> dict:
    res = {}
    clusters = df_te[CLUSTER_COL].astype(str).values
    for eje, col in PROTECTED.items():
        if col not in df_te.columns:
            continue
        g = df_te[col].astype(str).values
        tab = group_rates(y, pred, g)
        if tab.empty:
            continue
        tab.to_csv(out / f"tasas_{etiqueta}_{eje}.csv", index=False)
        entrada = {"primario": eje == EJE_PRIMARIO,
                   "tabla": tab.to_dict("records"),
                   "disparidades": disparities(tab)}
        if eje == EJE_PRIMARIO:
            entrada["inferencia_cluster"] = cluster_bootstrap(
                y, pred, g, clusters, n_boot=n_boot, n_perm=n_perm)
        else:
            entrada["nota"] = "exploratorio: sin prueba de significancia (ver preregistro)"
        res[eje] = entrada
    return res


def run(df: pd.DataFrame, out: Path, n_boot: int, n_perm: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {"n_total": int(len(df)), "tasa_base": float(df[TARGET].mean()),
               "preregistro": {"eje_primario": EJE_PRIMARIO,
                               "metrica_primaria": METRICA_PRIMARIA,
                               "capacidad": CAPACIDAD_PRINCIPAL}}

    periodos = sorted(df["periodo"].astype(str).unique())
    if len(periodos) < 3:
        raise RuntimeError("Se requieren >=3 cohortes para la particion temporal.")
    test_p, val_p = periodos[-1], periodos[-2]
    tr = df[~df["periodo"].astype(str).isin([test_p, val_p])].copy()
    va = df[df["periodo"].astype(str) == val_p].copy()
    te = df[df["periodo"].astype(str) == test_p].copy()
    add_school_history(tr, [va, te])
    R["particion"] = {"tipo": "temporal", "train": periodos[:-2], "val": val_p, "test": test_p,
                      "n": [len(tr), len(va), len(te)]}
    ytr, yva, yte = tr[TARGET].values, va[TARGET].values, te[TARGET].values

    # --- E0 ---
    Xtr, pre = build_matrix(tr, True)
    Xva, _ = build_matrix(va, True)
    Xte, _ = build_matrix(te, True)
    R["tuning"] = tune_on_validation(tr, va, pre, Xtr, Xva, ytr, yva)

    perf, probas = {}, {}
    for name, info in R["tuning"].items():
        pipe = Pipeline([("pre", pre), ("m", build_estimator(name, info["params"]))]).fit(Xtr, ytr)
        pr = pipe.predict_proba(Xte)[:, 1]
        pd_ = predict_at_capacity(pr, CAPACIDAD_PRINCIPAL)
        perf[name] = evaluate(yte, pr, pd_)
        probas[name] = pr
        print(f"  {name:22s} AUC={perf[name]['auc_roc']:.4f} AP={perf[name]['auc_pr']:.4f}")
    R["desempeno_E0"] = perf

    best = max(perf, key=lambda k: perf[k]["auc_roc"])
    R["mejor_modelo"] = best
    pred0 = predict_at_capacity(probas[best], CAPACIDAD_PRINCIPAL)
    R["auditoria_E0"] = audit(yte, pred0, te, out, n_boot, n_perm, "E0")

    # interseccional
    inter = (te[PROTECTED["zona"]].astype(str) + " x " + te["estrato_grupo"].astype(str)).values
    tabi = group_rates(yte, pred0, inter)
    tabi.to_csv(out / "tasas_interseccional.csv", index=False)
    R["auditoria_interseccional"] = {"tabla": tabi.to_dict("records"),
                                     "disparidades": disparities(tabi)}

    # --- E1: ceguera al atributo ---
    Xtr1, pre1 = build_matrix(tr, False)
    Xte1, _ = build_matrix(te, False)
    est_best = build_estimator(best, R["tuning"][best]["params"])
    p1 = Pipeline([("pre", pre1), ("m", est_best)]).fit(Xtr1, ytr)
    pr1 = p1.predict_proba(Xte1)[:, 1]
    pred1 = predict_at_capacity(pr1, CAPACIDAD_PRINCIPAL)
    gz = te[PROTECTED["zona"]].astype(str).values
    R["E1_ceguera"] = {"desempeno": evaluate(yte, pr1, pred1),
                       "disparidades_zona": disparities(group_rates(yte, pred1, gz))}
    R["proxies"] = {e: proxy_detection(te, e) for e in PROTECTED}

    # --- E2: reponderacion ---
    try:
        w = reweighing_weights(ytr, tr[PROTECTED["zona"]].astype(str).values)
        p2 = Pipeline([("pre", pre1), ("m", build_estimator(best, R["tuning"][best]["params"]))])
        p2.fit(Xtr1, ytr, **{"m__sample_weight": w})
        pr2 = p2.predict_proba(Xte1)[:, 1]
        pred2 = predict_at_capacity(pr2, CAPACIDAD_PRINCIPAL)
        R["E2_reponderacion"] = {"desempeno": evaluate(yte, pr2, pred2),
                                 "disparidades_zona": disparities(group_rates(yte, pred2, gz))}
    except Exception as e:
        R["E2_reponderacion"] = {"error": str(e)}

    # --- E3: gradiente exponenciado ---
    try:
        from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
        sub = rng.choice(len(Xtr1), min(len(Xtr1), 40_000), replace=False)
        base = WeightedPipeline(pre=pre1, estimator=LogisticRegression(max_iter=1000))
        eg = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds(), max_iter=8)
        eg.fit(Xtr1.iloc[sub], ytr[sub],
               sensitive_features=tr[PROTECTED["zona"]].astype(str).values[sub])
        pred3 = np.asarray(eg.predict(Xte1)).astype(int)
        R["E3_gradiente_exponenciado"] = {
            "tasa_seleccion": float(pred3.mean()),
            "disparidades_zona": disparities(group_rates(yte, pred3, gz))}
    except Exception as e:
        R["E3_gradiente_exponenciado"] = {"error": str(e)}

    # --- E4: umbrales por grupo ---
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        topt = ThresholdOptimizer(
            estimator=Pipeline([("pre", pre1), ("m", LogisticRegression(max_iter=1000))]),
            constraints="equalized_odds", predict_method="predict_proba", prefit=False)
        topt.fit(Xtr1, ytr, sensitive_features=tr[PROTECTED["zona"]].astype(str).values)
        pred4 = np.asarray(topt.predict(Xte1, sensitive_features=gz)).astype(int)
        R["E4_umbral_por_grupo"] = {
            "tasa_seleccion": float(pred4.mean()),
            "disparidades_zona": disparities(group_rates(yte, pred4, gz))}
    except Exception as e:
        R["E4_umbral_por_grupo"] = {"error": str(e)}

    # --- Sensibilidad: capacidad y umbral del outcome ---
    R["sensibilidad_capacidad"] = {
        f"{c:.2f}": disparities(group_rates(yte, predict_at_capacity(probas[best], c), gz))
        for c in (0.10, 0.20, 0.30)}

    # --- SHAP por grupo ---
    try:
        import shap
        Xs = pre.fit(Xtr).transform(Xte)
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else Xs
        mdl = build_estimator(best, R["tuning"][best]["params"]).fit(pre.transform(Xtr), ytr)
        sub = rng.choice(len(Xs), min(1500, len(Xs)), replace=False)
        expl = (shap.TreeExplainer(mdl) if hasattr(mdl, "feature_importances_")
                else shap.LinearExplainer(mdl, pre.transform(Xtr)[:800]))
        sv = np.asarray(expl.shap_values(Xs[sub]))
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        names = pre.get_feature_names_out()
        gsub = gz[sub]
        R["shap_por_zona"] = {
            str(g): dict(sorted(zip(names, np.abs(sv[gsub == g]).mean(0)),
                                key=lambda t: -t[1])[:10])
            for g in np.unique(gsub) if (gsub == g).sum() > 30}
    except Exception as e:
        R["shap_por_zona"] = {"error": str(e)}

    # --- Figura principal ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [("E0", perf[best]["auc_roc"], R["auditoria_E0"][EJE_PRIMARIO]["disparidades"]["dif_fnr"]),
               ("E1 ceguera", R["E1_ceguera"]["desempeno"]["auc_roc"],
                R["E1_ceguera"]["disparidades_zona"]["dif_fnr"])]
        for key, lab in [("E2_reponderacion", "E2 reponderación"),
                         ("E3_gradiente_exponenciado", "E3 restricción"),
                         ("E4_umbral_por_grupo", "E4 umbral/grupo")]:
            d = R.get(key, {})
            if "disparidades_zona" in d:
                auc = d.get("desempeno", {}).get("auc_roc", perf[best]["auc_roc"])
                pts.append((lab, auc, d["disparidades_zona"]["dif_fnr"]))
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        for lab, auc, fnr in pts:
            ax.scatter(fnr, auc, s=95)
            ax.annotate(lab, (fnr, auc), textcoords="offset points", xytext=(7, 5), fontsize=9)
        ax.set_xlabel("Disparidad de tasa de falsos negativos (urbano vs. rural)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(f"Costo en exactitud de reducir la inequidad\n(capacidad = {CAPACIDAD_PRINCIPAL:.0%})")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "fig_curva_costo_equidad.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        R["figura_error"] = str(e)

    (out / "resultados.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--quantile", type=float, default=0.20)
    ap.add_argument("--incluir-pandemia", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        print(f"[MODO VERIFICACION] {a.synthetic} filas sinteticas. NO son hallazgos cientificos.")
        df = make_synthetic(a.synthetic)
    elif a.data:
        df = load_real(a.data)
    else:
        raise SystemExit("Indique --data o --synthetic")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = preprocess(df, quantile=a.quantile, incluir_pandemia=a.incluir_pandemia)
    print(f"Filas: {len(df):,} | tasa base: {df[TARGET].mean():.3f} | "
          f"colegios: {df[CLUSTER_COL].nunique():,}")
    miss = missingness_analysis(df, out)
    print(f"Ausencia media de predictores por zona: {miss['ausencia_media_por_zona']}")
    R = run(df, out, n_boot=a.boot, n_perm=a.perm)
    print(f"\nMejor modelo: {R['mejor_modelo']}")
    print(f"Resultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
