"""
Extrae del zip de DataIcfes solo las cohortes necesarias, se queda con las
columnas del estudio y guarda en formato comprimido.

El zip pesa 9,1 GB y el disco disponible es ~4,8 GB: por eso se procesa una
cohorte a la vez y se borra el .txt crudo en cuanto queda guardado.
"""
import gc
import os
import subprocess
import sys
import time

import pandas as pd

ZIP = "/sessions/keen-epic-mccarthy/mnt/uploads/OneDrive_1_1-8-2026.zip"
OUT = "/sessions/keen-epic-mccarthy/mnt/outputs/data"
COHORTES = ["20182", "20192", "20222", "20232", "20242"]  # segundos semestres, sin pandemia

KEEP = [
    "periodo", "estu_consecutivo",
    # colegio
    "cole_area_ubicacion", "cole_naturaleza", "cole_calendario", "cole_caracter",
    "cole_jornada", "cole_genero", "cole_bilingue", "cole_depto_ubicacion",
    "cole_cod_dane_establecimiento", "cole_cod_mcpio_ubicacion",
    # estudiante
    "estu_genero", "estu_horassemanatrabaja", "estu_dedicacionlecturadiaria",
    "estu_dedicacioninternet", "estu_etnia", "estu_discapacidad", "estu_repite",
    "estu_inse_individual", "estu_nse_individual", "estu_nse_establecimiento",
    # hogar
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_personashogar", "fami_cuartoshogar", "fami_numlibros",
    "fami_situacioneconomica", "fami_trabajolabormadre", "fami_trabajolaborpadre",
    "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet",
    "fami_tienelavadora", "fami_tienemotocicleta", "fami_tieneserviciotv",
    "fami_tieneconsolavideojuegos", "fami_tienehornomicroogas",
    "fami_comecarnepescadohuevo", "fami_comecerealfrutoslegumbre",
    "fami_comelechederivados",
    # resultado
    "punt_global",
]


def procesar(coh: str) -> dict:
    txt = f"{OUT}/Examen_Saber_11_{coh}.txt"
    dst = f"{OUT}/saber11_{coh}.pkl.gz"
    if os.path.exists(dst):
        return {"cohorte": coh, "estado": "ya existia"}

    t0 = time.time()
    if not os.path.exists(txt):
        subprocess.run(["unzip", "-o", "-j", ZIP, f"Examen_Saber_11_{coh}.txt", "-d", OUT],
                       check=True, capture_output=True)

    cols = pd.read_csv(txt, sep=";", nrows=0).columns.tolist()
    usar = [c for c in KEEP if c in cols]
    faltan = [c for c in KEEP if c not in cols]

    df = pd.read_csv(txt, sep=";", usecols=usar, dtype=str, low_memory=False)
    df.to_pickle(dst, compression="gzip")

    info = {"cohorte": coh, "filas": len(df), "cols_guardadas": len(usar),
            "cols_faltantes": faltan,
            "mb_pkl": round(os.path.getsize(dst) / 1e6, 1),
            "seg": round(time.time() - t0, 1)}
    os.remove(txt)                      # liberar disco de inmediato
    del df
    gc.collect()
    return info


if __name__ == "__main__":
    objetivo = sys.argv[1:] or COHORTES
    for c in objetivo:
        print(procesar(c), flush=True)
