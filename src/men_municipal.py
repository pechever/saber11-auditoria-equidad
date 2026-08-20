"""
Carga y limpieza de las estadisticas municipales del MEN
(preescolar, basica y media, 2011-2024).

El archivo tal como se descarga tiene cuatro defectos que rompen el cruce con
Saber 11 en silencio. Este modulo los corrige y deja constancia de cada uno.

Uso:
    from men_municipal import load_men
    men = load_men("MEN_ESTADISTICAS...csv")
    saber = saber.merge(men, left_on=["cole_cod_mcpio_ubicacion","anio"],
                        right_on=["cod_municipio","anio"], how="left")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PCT_COLS = [
    "TASA_MATRICULACIÓN_5_16", "COBERTURA_NETA", "COBERTURA_NETA_TRANSICIÓN",
    "COBERTURA_NETA_PRIMARIA", "COBERTURA_NETA_SECUNDARIA", "COBERTURA_NETA_MEDIA",
    "COBERTURA_BRUTA", "COBERTURA_BRUTA_TRANSICIÓN", "COBERTURA_BRUTA_PRIMARIA",
    "COBERTURA_BRUTA_SECUNDARIA", "COBERTURA_BRUTA_MEDIA",
    "DESERCIÓN", "DESERCIÓN_TRANSICIÓN", "DESERCIÓN_PRIMARIA",
    "DESERCIÓN_SECUNDARIA", "DESERCIÓN_MEDIA",
    "APROBACIÓN", "APROBACIÓN_TRANSICIÓN", "APROBACIÓN_PRIMARIA",
    "APROBACIÓN_SECUNDARIA", "APROBACIÓN_MEDIA",
    "REPROBACIÓN", "REPROBACIÓN_TRANSICIÓN", "REPROBACIÓN_PRIMARIA",
    "REPROBACIÓN_SECUNDARIA", "REPROBACIÓN_MEDIA",
    "REPITENCIA", "REPITENCIA_TRANSICIÓN", "REPITENCIA_PRIMARIA",
    "REPITENCIA_SECUNDARIA", "REPITENCIA_MEDIA",
]

# Tasas donde un 0 exacto es implausible y casi con seguridad significa
# "sin dato". No se aplica a cobertura, donde 0 puede ser real (municipio
# sin oferta del nivel).
ZERO_IS_MISSING = [
    "DESERCIÓN", "DESERCIÓN_TRANSICIÓN", "DESERCIÓN_PRIMARIA",
    "DESERCIÓN_SECUNDARIA", "DESERCIÓN_MEDIA",
    "APROBACIÓN", "APROBACIÓN_PRIMARIA", "APROBACIÓN_SECUNDARIA", "APROBACIÓN_MEDIA",
    "REPROBACIÓN", "REPROBACIÓN_PRIMARIA", "REPROBACIÓN_SECUNDARIA", "REPROBACIÓN_MEDIA",
    "REPITENCIA", "REPITENCIA_PRIMARIA", "REPITENCIA_SECUNDARIA", "REPITENCIA_MEDIA",
]


def load_men(path: str, verbose: bool = True,
             zero_is_missing: list[str] | None = None) -> pd.DataFrame:
    """
    zero_is_missing: columnas donde un 0 exacto se recodifica como NaN.
      Por defecto se aplica a desercion, aprobacion, reprobacion y repitencia.
      OJO: es una decision discutible. En municipios pequenos un 0 real es
      plausible para repitencia y reprobacion (pocos estudiantes, ninguno
      repite). Para desercion en media el 0 es mucho mas sospechoso.
      Revise el log y ajuste la lista segun su criterio; documente la eleccion
      en el manuscrito.
    """
    zero_cols = ZERO_IS_MISSING if zero_is_missing is None else zero_is_missing
    df = pd.read_csv(path, dtype=str)
    log = []

    # --- Defecto 1: filas de agregado nacional mezcladas con municipios ---
    nac = df["CÓDIGO_MUNICIPIO"].str.strip().isin(["0", ""]) | (
        df["MUNICIPIO"].str.upper().str.strip() == "NACIONAL")
    if nac.any():
        log.append(f"eliminadas {nac.sum()} filas de agregado NACIONAL")
        df = df[~nac].copy()

    # --- Defecto 2: cero inicial perdido en Antioquia (05) y Atlantico (08) ---
    antes = df["CÓDIGO_MUNICIPIO"].nunique()
    df["cod_municipio"] = df["CÓDIGO_MUNICIPIO"].str.strip().str.zfill(5)
    df["cod_departamento"] = df["CÓDIGO_DEPARTAMENTO"].str.strip().str.zfill(2)
    despues = df["cod_municipio"].nunique()
    if antes != despues:
        log.append(f"zfill(5) unifico codigos duplicados: {antes} -> {despues} municipios "
                   f"(afectaba a Antioquia y Atlantico)")

    # --- Defecto 3: porcentajes almacenados como texto con '%' ---
    for c in PCT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.rstrip("%").str.replace(",", "."),
                                  errors="coerce")

    df["anio"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["poblacion_5_16"] = pd.to_numeric(df["POBLACIÓN_5_16"], errors="coerce")

    # --- Defecto 4: ausencia codificada como 0 ---
    for c in zero_cols:
        if c in df.columns:
            n = int((df[c] == 0).sum())
            if n:
                df.loc[df[c] == 0, c] = np.nan
                log.append(f"{c}: {n} ceros recodificados como NaN")

    # Indicador de cobertura bruta implausible (>100%): matricula extraedad
    # o estudiantes que cruzan frontera municipal. No se corrige, se marca.
    if "COBERTURA_BRUTA_MEDIA" in df.columns:
        df["flag_cobertura_bruta_media_gt100"] = df["COBERTURA_BRUTA_MEDIA"] > 100

    df = df.rename(columns={
        "COBERTURA_NETA_MEDIA": "cobertura_neta_media",
        "COBERTURA_BRUTA_MEDIA": "cobertura_bruta_media",
        "DESERCIÓN_MEDIA": "desercion_media",
        "REPROBACIÓN_MEDIA": "reprobacion_media",
        "REPITENCIA_MEDIA": "repitencia_media",
        "APROBACIÓN_MEDIA": "aprobacion_media",
        "MUNICIPIO": "municipio", "DEPARTAMENTO": "departamento",
    })

    if verbose:
        print("[MEN] limpieza aplicada:")
        for l in log:
            print("   -", l)
        print(f"[MEN] {len(df):,} filas | {df['cod_municipio'].nunique():,} municipios | "
              f"{int(df['anio'].min())}-{int(df['anio'].max())}")

    return df


def add_rurality(men: pd.DataFrame, rural_csv: str | None = None) -> pd.DataFrame:
    """El dataset NO trae clasificacion urbano/rural.

    Para desagregar por ruralidad hay que cruzar una fuente externa, p. ej. las
    categorias de ruralidad del DNP (ciudades y aglomeraciones / intermedio /
    rural / rural disperso) por codigo DIVIPOLA. Sin ese cruce, cualquier
    afirmacion 'rural vs urbano' basada solo en este archivo es infundada.
    """
    if rural_csv is None:
        raise NotImplementedError(
            "Se requiere un archivo externo con categorias de ruralidad por "
            "codigo DIVIPOLA (DNP). Este dataset no las contiene.")
    rur = pd.read_csv(rural_csv, dtype=str)
    rur["cod_municipio"] = rur.iloc[:, 0].str.strip().str.zfill(5)
    return men.merge(rur, on="cod_municipio", how="left")


if __name__ == "__main__":
    import sys
    m = load_men(sys.argv[1])
    print()
    print(m.groupby("anio")["cobertura_neta_media"]
           .agg(["count", "median"]).tail(6).to_string())
