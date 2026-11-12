"""Construye la muestra analitica a partir de los microdatos originales del Icfes.

Toma los cinco archivos de cohorte, conserva solo las variables que usa el
analisis, aplica la homologacion de etiquetas documentada y excluye a los
inscritos que presentaron el examen de forma individual (sin colegio, y por
tanto sin conglomerado). El resultado es un unico Parquet de unos 31 MB que
reproduce todas las cifras del articulo.

Uso:
    python src/construir_muestra.py --entrada data/raw --salida data/muestra_analitica.parquet

Los archivos de entrada se esperan como Examen_Saber_11_<periodo>.txt, con
separador ';' y codificacion latin-1. Vease data/README.md.
"""
import argparse
from pathlib import Path

import pandas as pd

COHORTES = [20182, 20192, 20222, 20232, 20242]

# Solo lo que usa el analisis. Las variables derivadas (hacinamiento y tasa
# historica del colegio) las calcula fairness_pipeline.py, no se guardan aqui.
COLUMNAS = [
    "periodo", "estu_estudiante", "cole_cod_dane_establecimiento",
    # atributos protegidos
    "cole_area_ubicacion", "cole_naturaleza", "estu_genero",
    # resultado del que se deriva la variable objetivo
    "punt_global",
    # caracteristicas del colegio
    "cole_calendario", "cole_caracter", "cole_jornada", "cole_genero",
    "cole_bilingue", "cole_depto_ubicacion",
    # contexto del hogar y uso del tiempo
    "fami_trabajolabormadre", "fami_trabajolaborpadre",
    "estu_dedicacionlecturadiaria", "estu_dedicacioninternet",
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_personashogar", "fami_cuartoshogar", "fami_numlibros",
    "fami_situacioneconomica", "estu_horassemanatrabaja",
    # tenencia de bienes
    "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
    "fami_tienelavadora", "fami_tienemotocicleta", "fami_tieneserviciotv",
    "fami_tieneconsolavideojuegos", "fami_tienehornomicroogas",
]

# Cifras publicadas en la Tabla 3, para comprobar la construccion.
ESPERADO = {20182: 555_319, 20192: 554_769, 20222: 543_706,
            20232: 562_914, 20242: 557_298}


def leer_cohorte(ruta: Path) -> pd.DataFrame:
    # Los archivos del Icfes son UTF-8. Leerlos como latin-1 no falla, pero
    # corrompe las tildes ("Educacion" -> "EducaciA3n") y deja fuera del mapeo
    # ordinal a una cuarta parte de fami_educacionmadre, que acaba imputada.
    df = pd.read_csv(ruta, sep=';', encoding='utf-8', low_memory=False,
                     usecols=lambda c: c in COLUMNAS,
                     dtype={'cole_cod_dane_establecimiento': 'str'})

    # Homologacion NO documentada por el Icfes: en la cohorte 2024-2 la zona
    # del colegio se codifica URBANA en 450.514 registros, mientras todas las
    # cohortes anteriores usan URBANO. Sin esto el grupo urbano queda partido
    # en dos justo en la cohorte de prueba.
    df['cole_area_ubicacion'] = df['cole_area_ubicacion'].replace({'URBANA': 'URBANO'})

    df['punt_global'] = pd.to_numeric(df['punt_global'], errors='coerce')

    # Exclusion de inscritos individuales: sin colegio no hay conglomerado, y
    # el bootstrap y las pruebas de permutacion operan a nivel de colegio.
    cod = df['cole_cod_dane_establecimiento']
    df = df[cod.notna() & cod.astype(str).str.strip().ne('')]
    df = df[df['punt_global'].notna()]

    for c in df.columns:
        if df[c].dtype == object and c != 'cole_cod_dane_establecimiento':
            df[c] = df[c].astype('category')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entrada', default='data/raw')
    ap.add_argument('--salida', default='data/muestra_analitica.parquet')
    args = ap.parse_args()

    entrada, salida = Path(args.entrada), Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    partes, problemas = [], []
    for c in COHORTES:
        ruta = entrada / f'Examen_Saber_11_{c}.txt'
        if not ruta.exists():
            raise SystemExit(f'falta {ruta}. Vease data/README.md')
        d = leer_cohorte(ruta)
        marca = 'OK' if len(d) == ESPERADO[c] else f'DIFIERE (esperado {ESPERADO[c]:,})'
        if len(d) != ESPERADO[c]:
            problemas.append(c)
        print(f'  {c}: {len(d):,} filas  {marca}')
        partes.append(d)

    df = pd.concat(partes, ignore_index=True)
    df.to_parquet(salida, compression='zstd', index=False)

    mb = salida.stat().st_size / 1e6
    print(f'\n{salida}: {len(df):,} filas, '
          f'{df.cole_cod_dane_establecimiento.nunique():,} colegios, {mb:.1f} MB')
    if problemas:
        print('\nAVISO: las cohortes', problemas, 'no coinciden con la Tabla 3.')
        print('El Icfes revisa retroactivamente los archivos publicados; '
              'compruebe la fecha de su descarga.')


if __name__ == '__main__':
    main()
