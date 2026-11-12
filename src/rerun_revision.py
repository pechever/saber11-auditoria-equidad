"""Re-ejecucion para la revision: AUC por escenario, variantes a capacidad fija
e inferencia con parametros registrados.

Responde a tres observaciones del evaluador:

  (1) las tasas de seleccion diferian entre escenarios (E0-E2 marcaban el 20%,
      E3 el 9,0% y E4 el 8,1%), lo que impedia comparar la disparidad. Se anaden
      E3' y E4': las mismas puntuaciones, marcando el 20% superior.
  (4) los parametros de bootstrap y permutacion no constaban. Ahora se fijan,
      se registran en el JSON de salida y quedan auditables.
  (6) faltaba el AUC de los escenarios de mitigacion. Se calcula para todos.

Se ejecuta por etapas, con puntos de control en disco:

    python src/rerun_revision.py A0   # particion
    python src/rerun_revision.py A1   # E0
    python src/rerun_revision.py A2   # E1
    python src/rerun_revision.py B    # E2, E3, E4
    python src/rerun_revision.py C    # AUC y variantes a capacidad fija
    python src/rerun_revision.py D1   # bootstrap
    python src/rerun_revision.py D2   # permutacion
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
import fairness_pipeline as fp

MUESTRA = Path('data/muestra_analitica.parquet')
CK = Path('/tmp/rev')
OUT = Path('results/revision')

# Parametros de inferencia: fijados aqui y registrados en la salida.
N_BOOT = 2_000
N_PERM = 5_000
ALPHA = 0.05
SEMILLA = fp.RANDOM_STATE

# Hiperparametros ya seleccionados sobre la cohorte de validacion.
PARAMS_XGB = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _por_codigos(serie: pd.Series, valores, relleno=np.nan) -> np.ndarray:
    """Traduce una columna categorica a numeros mapeando sus CATEGORIAS.

    Se opera sobre los codigos enteros, no sobre los 2,77 millones de cadenas,
    y asi la conversion no multiplica por doce el consumo de memoria.
    """
    v = np.asarray(valores, dtype=float)
    codes = serie.cat.codes.to_numpy()
    out = np.full(len(codes), relleno, dtype=float)
    validos = codes >= 0
    out[validos] = v[codes[validos]]
    return out


def leer_muestra(ruta: Path = MUESTRA) -> pd.DataFrame:
    """Carga la muestra analitica y deshace SOLO las categorias necesarias.

    El Parquet guarda el texto como `category` y ocupa 0,5 GB en memoria;
    convertirlo todo a `object` lo lleva a 5,9 GB y agota la maquina. Pero
    preprocess() condiciona sus mapeos a `dtype == object`, asi que una columna
    que llegue como categoria se salta en silencio el mapeo ordinal y queda sin
    convertir a numero. Se convierten unicamente esas columnas, que son de texto
    corto y apenas pesan; las de texto largo siguen como categoria y llegan asi
    al codificador one-hot, que las admite.
    """
    df = pd.read_parquet(ruta)

    # El codigo DANE son 2,77 millones de cadenas y por si solo pesa mas que
    # el resto junto; como categoria ocupa una fraccion y groupby lo admite.
    df['cole_cod_dane_establecimiento'] = df['cole_cod_dane_establecimiento'].astype('category')
    df = df.drop(columns=['estu_estudiante'], errors='ignore')   # no lo usa el modelo

    def es_cat(c):
        return isinstance(df[c].dtype, pd.CategoricalDtype)

    # Mapeos aplicados sobre las CATEGORIAS, no sobre los 2,77 millones de
    # valores: el resultado es identico al de preprocess() y no infla memoria.
    for col, mapping in fp.ORDINAL_MAPS.items():
        if col in df.columns and es_cat(col):
            cats = df[col].cat.categories
            desconocidas = {c for c in cats if str(c).strip() not in mapping}
            if desconocidas:
                prop = df[col].isin(desconocidas).mean()
                print(f'[ALERTA] {col}: {prop:.1%} en categorias NO reconocidas '
                      f'{sorted(map(str, desconocidas))[:8]}')
            df[col] = _por_codigos(
                df[col], [mapping.get(str(c).strip(), np.nan) for c in cats])

    for col in fp.BINARY_COLS:
        if col in df.columns and es_cat(col):
            df[col] = _por_codigos(
                df[col],
                [1.0 if str(c).strip().lower() in ('si', 'sí', 's') else 0.0
                 for c in df[col].cat.categories], relleno=0.0).astype(int)

    # Normalizacion de los tres atributos que preprocess() trata aparte.
    for col, fn in [('cole_area_ubicacion', lambda s: 'URBANO' if s.strip().upper() == 'URBANA'
                     else s.strip().upper()),
                    ('cole_naturaleza', lambda s: s.strip().upper()),
                    ('estu_genero', lambda s: s.strip()[:1].upper())]:
        if col in df.columns and es_cat(col):
            df[col] = df[col].cat.rename_categories(
                [fn(str(c)) for c in df[col].cat.categories]).astype(object)
    return df


def partes():
    """Particion temporal, identica a la del analisis principal."""
    df = fp.preprocess(leer_muestra())
    per = sorted(df['periodo'].astype(str).unique())
    test_p, val_p = per[-1], per[-2]
    tr = df[~df['periodo'].astype(str).isin([test_p, val_p])].copy()
    va = df[df['periodo'].astype(str) == val_p].copy()
    te = df[df['periodo'].astype(str) == test_p].copy()
    fp.add_school_history(tr, [va, te])
    return tr, va, te, per


def guardar(nombre, **arrays):
    CK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CK / f'{nombre}.npz', **arrays)


def cargar(nombre):
    return np.load(CK / f'{nombre}.npz', allow_pickle=True)


# --------------------------------------------------------------- etapa A
def etapa_A0():
    """Prepara la particion y la deja en disco.

    Cada etapa corre en su propio proceso: con 3,6 GB disponibles no caben a la
    vez los tres DataFrames, las matrices codificadas y el modelo.
    """
    tr, va, te, per = partes()
    log(f'particion  train={len(tr):,}  val={len(va):,}  test={len(te):,}')
    CK.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(CK / 'tr.parquet')
    te.to_parquet(CK / 'te.parquet')
    guardar('base',
            ytr=tr[fp.TARGET].values, yte=te[fp.TARGET].values,
            gtr=tr[fp.PROTECTED['zona']].astype(str).values,
            gz=te[fp.PROTECTED['zona']].astype(str).values,
            clusters=te[fp.CLUSTER_COL].astype(str).values)
    log('etapa A0 guardada')


def _entrena(incluir_protegidos: bool, etiqueta: str):
    b = cargar('base')
    tr = pd.read_parquet(CK / 'tr.parquet')
    Xtr, pre = fp.build_matrix(tr, incluir_protegidos)
    ytr = b['ytr']
    del tr
    m = Pipeline([('pre', pre), ('m', fp.build_estimator('xgboost', PARAMS_XGB))])
    m.fit(Xtr, ytr)
    del Xtr
    te = pd.read_parquet(CK / 'te.parquet')
    Xte, _ = fp.build_matrix(te, incluir_protegidos)
    del te
    pr = m.predict_proba(Xte)[:, 1]
    yte = b['yte']
    auc = fp.evaluate(yte, pr, fp.predict_at_capacity(pr, fp.CAPACIDAD_PRINCIPAL))['auc_roc']
    log(f'{etiqueta} listo  AUC={auc:.4f}')
    guardar(etiqueta, pr=pr)


def etapa_A1():
    """E0: modelo completo."""
    _entrena(True, 'E0')


def etapa_A2():
    """E1: ciego a zona, sector y sexo. El estrato se conserva (seccion 2.4)."""
    _entrena(False, 'E1')


# --------------------------------------------------------------- etapa B
def etapa_B():
    a = cargar('base')
    ytr, yte = a['ytr'], a['yte']
    tr = pd.read_parquet(CK / 'tr.parquet'); te = pd.read_parquet(CK / 'te.parquet')
    Xtr1, pre1 = fp.build_matrix(tr, False)
    Xte1, _ = fp.build_matrix(te, False)
    gtr = a['gtr']; gz = a['gz']
    res = {}

    # E2: reponderacion de Kamiran-Calders
    w = fp.reweighing_weights(ytr, gtr)
    m2 = Pipeline([('pre', pre1), ('m', fp.build_estimator('xgboost', PARAMS_XGB))])
    m2.fit(Xtr1, ytr, **{'m__sample_weight': w})
    pr2 = m2.predict_proba(Xte1)[:, 1]
    log(f'E2 listo  AUC={fp.evaluate(yte, pr2, fp.predict_at_capacity(pr2, 0.20))["auc_roc"]:.4f}')

    # E3: gradiente exponenciado bajo probabilidades igualadas
    from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    sub = fp.rng.choice(len(Xtr1), min(len(Xtr1), 40_000), replace=False)
    base = fp.WeightedPipeline(pre=pre1, estimator=LogisticRegression(max_iter=1000))
    eg = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds(), max_iter=8)
    eg.fit(Xtr1.iloc[sub], ytr[sub], sensitive_features=gtr[sub])
    pred3 = np.asarray(eg.predict(Xte1)).astype(int)
    # puntuacion continua del clasificador aleatorizado, para la variante E3'
    sc3 = np.asarray(eg._pmf_predict(Xte1))[:, 1]
    log(f'E3 listo  tasa de seleccion={pred3.mean():.3f}')

    # E4: umbrales especificos por grupo
    from fairlearn.postprocessing import ThresholdOptimizer
    topt = ThresholdOptimizer(
        estimator=Pipeline([('pre', pre1), ('m', LogisticRegression(max_iter=1000))]),
        constraints='equalized_odds', predict_method='predict_proba', prefit=False)
    topt.fit(Xtr1, ytr, sensitive_features=gtr)
    pred4 = np.asarray(topt.predict(Xte1, sensitive_features=gz)).astype(int)
    sc4 = np.asarray(topt._pmf_predict(Xte1, sensitive_features=gz))[:, 1]
    log(f'E4 listo  tasa de seleccion={pred4.mean():.3f}')

    guardar('B', pr2=pr2, pred3=pred3, sc3=sc3, pred4=pred4, sc4=sc4)
    log('etapa B guardada')


# --------------------------------------------------------------- etapa C
def etapa_C():
    a, b = cargar('base'), cargar('B')
    e0, e1 = cargar('E0')['pr'], cargar('E1')['pr']
    yte, gz = a['yte'], a['gz']
    cap = fp.CAPACIDAD_PRINCIPAL
    from sklearn.metrics import roc_auc_score

    filas = []

    def fila(nombre, score, pred, auc):
        d = fp.disparities(fp.group_rates(yte, pred, gz))
        t = fp.group_rates(yte, pred, gz).set_index('grupo')
        filas.append({
            'escenario': nombre,
            'auc_roc': auc,
            'tasa_seleccion': float(pred.mean()),
            'recall_global': float(((pred == 1) & (yte == 1)).sum() / max((yte == 1).sum(), 1)),
            'fnr_urbano': float(t.loc['URBANO', 'fnr']),
            'fnr_rural': float(t.loc['RURAL', 'fnr']),
            'dif_fnr': d['dif_fnr']})

    for nombre, pr in [('E0 modelo completo', e0),
                       ('E1 ciego al atributo', e1),
                       ('E2 reponderacion', b['pr2'])]:
        fila(nombre, pr, fp.predict_at_capacity(pr, cap), float(roc_auc_score(yte, pr)))

    # E3 y E4 tal como los producen sus metodos, con su propia tasa de seleccion
    fila('E3 restriccion de probabilidades igualadas', b['sc3'], b['pred3'],
         float(roc_auc_score(yte, b['sc3'])))
    fila('E4 umbrales por grupo', b['sc4'], b['pred4'],
         float(roc_auc_score(yte, b['sc4'])))

    # variantes a capacidad fija: misma puntuacion, se marca el 20% superior
    fila("E3' a capacidad fija", b['sc3'], fp.predict_at_capacity(b['sc3'], cap),
         float(roc_auc_score(yte, b['sc3'])))
    fila("E4' a capacidad fija", b['sc4'], fp.predict_at_capacity(b['sc4'], cap),
         float(roc_auc_score(yte, b['sc4'])))

    t = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / 'escenarios_con_auc.csv', index=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    log('etapa C guardada')


# --------------------------------------------------------------- etapa D
def _datos_inferencia():
    a = cargar('base')
    pred0 = fp.predict_at_capacity(cargar('E0')['pr'], fp.CAPACIDAD_PRINCIPAL)
    return a['yte'], pred0, a['gz'], a['clusters']


LOTE = Path('/tmp/rev/lotes')


def etapa_D1():
    """Bootstrap por conglomerado de colegio, en lotes reanudables.

    Cada llamada anade un lote y se detiene; se repite hasta reunir N_BOOT.
    El limite de tiempo por llamada no da para las 2.000 replicas de una vez.
    """
    yte, pred0, gz, clusters = _datos_inferencia()
    LOTE.mkdir(parents=True, exist_ok=True)
    hechas = sorted(LOTE.glob('boot_*.npz'))
    n_previas = sum(len(np.load(f)['dif_fnr']) for f in hechas)
    if n_previas >= N_BOOT:
        log(f'bootstrap ya completo: {n_previas} replicas')
        return
    objetivo = min(600, N_BOOT - n_previas)
    log(f'bootstrap: lote de {objetivo} (acumuladas {n_previas}/{N_BOOT})')

    idx = pd.Series(np.arange(len(yte))).groupby(pd.Series(clusters)).apply(np.array)
    acc: dict[str, list] = {}
    for _ in range(objetivo):
        sel = fp.rng.choice(len(idx), len(idx), replace=True)
        ii = np.concatenate([idx.iloc[s] for s in sel])
        for k, v in fp.disparities(fp.group_rates(yte[ii], pred0[ii], gz[ii])).items():
            acc.setdefault(k, []).append(v)
    np.savez_compressed(LOTE / f'boot_{len(hechas):03d}.npz',
                        **{k: np.asarray(v) for k, v in acc.items()})
    log(f'lote guardado; acumuladas {n_previas + objetivo}/{N_BOOT}')


def etapa_D1f():
    """Combina los lotes del bootstrap y calcula los intervalos."""
    acc: dict[str, list] = {}
    for f in sorted(LOTE.glob('boot_*.npz')):
        z = np.load(f)
        for k in z.files:
            acc.setdefault(k, []).append(z[k])
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    n = len(acc[fp.METRICA_PRIMARIA])
    out = {k: {'punto': float(np.nanmean(v)),
               'ic_inf': float(np.nanpercentile(v, 100 * ALPHA / 2)),
               'ic_sup': float(np.nanpercentile(v, 100 * (1 - ALPHA / 2)))}
           for k, v in acc.items()}
    out['_n_replicas'] = n
    (CK / 'boot.json').write_text(json.dumps(out, indent=1))
    d = out[fp.METRICA_PRIMARIA]
    log(f"{n} replicas | {fp.METRICA_PRIMARIA} punto={d['punto']:.4f} "
        f"IC95=[{d['ic_inf']:.4f}, {d['ic_sup']:.4f}]")


def etapa_D2():
    """Permutacion de la etiqueta de grupo A NIVEL DE COLEGIO.

    Solo sobre la metrica primaria preregistrada: el propio articulo declara
    que las pruebas de significancia se realizan unicamente sobre el eje y la
    metrica preinscritos, de modo que probar las demas inflaria el error de
    tipo I sin que nadie lo haya anunciado.
    """
    yte, pred0, gz, clusters = _datos_inferencia()
    LOTE.mkdir(parents=True, exist_ok=True)
    hechas = sorted(LOTE.glob('perm_*.npy'))
    nulos = [np.load(f) for f in hechas]
    n_previas = sum(len(x) for x in nulos)

    if n_previas < N_PERM:
        objetivo = min(1800, N_PERM - n_previas)
        log(f'permutacion: lote de {objetivo} (acumuladas {n_previas}/{N_PERM})')
        # la etiqueta de grupo se permuta POR COLEGIO, no por estudiante: asi
        # se preserva la estructura de conglomerado bajo la hipotesis nula
        gpc = pd.Series(gz).groupby(pd.Series(clusters)).agg(
            lambda s: s.value_counts().index[0])
        lote = []
        for _ in range(objetivo):
            mezcla = pd.Series(fp.rng.permutation(gpc.values), index=gpc.index)
            g = mezcla.reindex(clusters).values
            v = fp.disparities(fp.group_rates(yte, pred0, g)).get(fp.METRICA_PRIMARIA, np.nan)
            if np.isfinite(v):
                lote.append(v)
        np.save(LOTE / f'perm_{len(hechas):03d}.npy', np.asarray(lote))
        log(f'lote guardado; acumuladas {n_previas + len(lote)}/{N_PERM}')
        return

    null = np.concatenate(nulos)
    obs = fp.disparities(fp.group_rates(yte, pred0, gz))[fp.METRICA_PRIMARIA]
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    log(f'observado={obs:.4f}  nulo: media={null.mean():.4f} max={null.max():.4f}  '
        f'n={len(null)}')
    boot = json.loads((CK / 'boot.json').read_text())
    salida = {
        'parametros': {
            'n_boot': boot.get('_n_replicas', N_BOOT), 'n_perm': int(len(null)), 'alpha': ALPHA, 'semilla': SEMILLA,
            'unidad_remuestreo': 'colegio', 'unidad_permutacion': 'colegio',
            'metrica_contrastada': fp.METRICA_PRIMARIA,
            'eje_contrastado': fp.EJE_PRIMARIO,
            'intervalo': 'percentil 2,5 a 97,5',
            'p_minimo_alcanzable': 1 / (len(null) + 1)},
        'bootstrap': boot,
        'p_permutacion': p}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inferencia.json').write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    log(f'p de permutacion = {p:.6f}  (minimo alcanzable {1 / (N_PERM + 1):.6f})')


if __name__ == '__main__':
    {'A0': etapa_A0, 'A1': etapa_A1, 'A2': etapa_A2,
     'B': etapa_B, 'C': etapa_C, 'D1': etapa_D1, 'D1f': etapa_D1f, 'D2': etapa_D2}[sys.argv[1]]()
