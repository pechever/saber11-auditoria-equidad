"""
Consolida las cohortes ya extraidas en un unico marco listo para la auditoria.

Restriccion: 3 GB de RAM para 3 millones de filas. Cargar todo como `object`
desborda la memoria, asi que se procesa una cohorte a la vez, se convierte a
tipos compactos y solo entonces se concatena.
"""
import gc
import glob
import os
import sys

import numpy as np
import pandas as pd

import fairness_pipeline as fp

SLIM = "/tmp/saber/slim"
os.makedirs(SLIM, exist_ok=True)

# Columnas que sobreviven al preprocesamiento
MANTENER = (["periodo", fp.CLUSTER_COL, fp.TARGET, "punt_global", "estrato_grupo",
             "cole_cod_mcpio_ubicacion", "estu_inse_individual"]
            + [v for v in fp.PROTECTED.values()]
            + fp.NUMERIC_COLS + fp.CATEGORICAL_COLS)


def procesar(path: str) -> str:
    coh = os.path.basename(path).replace("c", "").replace(".pkl", "")
    dst = f"{SLIM}/{coh}.pkl"
    if os.path.exists(dst):
        return f"{coh} ya estaba"

    df = pd.read_pickle(path)
    # Solo las columnas que requieren mapeo pasan a texto; el resto se queda
    # como category para no reventar la memoria.
    for c in list(fp.ORDINAL_MAPS) + fp.BINARY_COLS + ["estu_genero"]:
        if c in df.columns:
            df[c] = df[c].astype(object)

    d = fp.preprocess(df, incluir_pandemia=True)   # el filtro ya se hizo al elegir cohortes
    del df
    gc.collect()

    cols = [c for c in dict.fromkeys(MANTENER) if c in d.columns]
    d = d[cols]
    for c in d.columns:                            # compactar
        if d[c].dtype == "float64":
            d[c] = d[c].astype("float32")
        elif d[c].dtype == "int64":
            d[c] = d[c].astype("int32")
        elif d[c].dtype == object and d[c].nunique() < 3000:
            d[c] = d[c].astype("category")

    d.to_pickle(dst)
    n, mb = len(d), os.path.getsize(dst) / 1e6
    del d
    gc.collect()
    return f"{coh}: {n:,} filas -> {mb:.0f} MB"


if __name__ == "__main__":
    if sys.argv[1:] == ["unir"]:
        fs = sorted(glob.glob(f"{SLIM}/*.pkl"))
        d = pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True)
        d.to_pickle("/tmp/saber/consolidado.pkl")
        print(f"consolidado: {len(d):,} filas x {d.shape[1]} cols | "
              f"{d.memory_usage(deep=True).sum()/1e6:.0f} MB en memoria")
        print(f"tasa base: {d[fp.TARGET].mean():.4f} | colegios: {d[fp.CLUSTER_COL].nunique():,}")
    else:
        for p in sys.argv[1:]:
            print(procesar(p), flush=True)
