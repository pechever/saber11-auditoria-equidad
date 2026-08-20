"""
Auditoria de equidad sobre los microdatos reales de Saber 11.

Ejecucion por etapas con checkpoints: el entorno limita cada llamada a ~45 s,
asi que cada etapa guarda su resultado y la siguiente lo retoma.

    python run_real.py A   # particion + ajuste + probabilidades (E0)
    python run_real.py B   # auditoria E0 + inferencia por conglomerado
    python run_real.py C   # E1 ceguera al atributo + deteccion de proxies
    python run_real.py D   # mitigacion E2, E3, E4
    python run_real.py E   # sensibilidad + figuras + volcado de resultados
"""
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fairness_pipeline as fp

CK = "/tmp/saber/ckpt"
OUT = "/sessions/keen-epic-mccarthy/mnt/outputs/resultados_reales"
os.makedirs(CK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TEST, VAL = "20242", "20232"
SUB_TUNING = 300_000     # submuestra para la busqueda de hiperparametros


def save(name, obj):
    with open(f"{CK}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(f"{CK}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def partes():
    d = pd.read_pickle("/tmp/saber/consolidado.pkl")
    p = d["periodo"].astype(str)
    tr, va, te = d[~p.isin([TEST, VAL])].copy(), d[p == VAL].copy(), d[p == TEST].copy()
    fp.add_school_history(tr, [va, te])
    return tr, va, te


# ---------------------------------------------------------------- etapa A
# Hiperparametros ya seleccionados sobre la cohorte de validacion 2023-2
# (submuestra de 300.000 del entrenamiento). Se fijan aqui para poder
# ajustar cada modelo en una llamada separada.
TUNING = {
    "regresion_logistica": {"params": {"C": 0.1}, "auc_val": 0.8188},
    "xgboost": {"params": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1,
                           "subsample": 0.8, "colsample_bytree": 0.8,
                           "min_child_weight": 20}, "auc_val": 0.8334},
}


def etapa_A(modelo=None):
    """Ajusta UN modelo sobre el entrenamiento completo y guarda sus
    probabilidades. Se llama una vez por modelo para caber en el limite de
    tiempo por llamada."""
    t0 = time.time()
    tr, va, te = partes()
    R = json.load(open(f"{CK}/R.json")) if os.path.exists(f"{CK}/R.json") else {}
    if not R:
        R = {"n_total": int(len(tr) + len(va) + len(te)),
             "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "cohortes_train": sorted(tr.periodo.astype(str).unique()),
             "cohorte_val": VAL, "cohorte_test": TEST,
             "tasa_base_train": float(tr[fp.TARGET].mean()),
             "tasa_base_test": float(te[fp.TARGET].mean()),
             "n_colegios": int(pd.concat([tr, va, te])[fp.CLUSTER_COL].nunique()),
             "tuning": TUNING, "desempeno_E0": {},
             "preregistro": {"eje": fp.EJE_PRIMARIO, "metrica": fp.METRICA_PRIMARIA,
                             "capacidad": fp.CAPACIDAD_PRINCIPAL}}

    Xtr, pre = fp.build_matrix(tr, True)
    Xte, _ = fp.build_matrix(te, True)
    ytr, yte = tr[fp.TARGET].values, te[fp.TARGET].values

    probas = load("probas") if os.path.exists(f"{CK}/probas.pkl") else {}
    for name in ([modelo] if modelo else list(TUNING)):
        pipe = Pipeline([("pre", pre),
                         ("m", fp.build_estimator(name, TUNING[name]["params"]))])
        pipe.fit(Xtr, ytr)
        pr = pipe.predict_proba(Xte)[:, 1]
        pd_ = fp.predict_at_capacity(pr, fp.CAPACIDAD_PRINCIPAL)
        R["desempeno_E0"][name] = fp.evaluate(yte, pr, pd_)
        probas[name] = pr
        print(f"  {name:22s} AUC={R['desempeno_E0'][name]['auc_roc']:.4f} "
              f"AP={R['desempeno_E0'][name]['auc_pr']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    R["mejor_modelo"] = max(R["desempeno_E0"], key=lambda k: R["desempeno_E0"][k]["auc_roc"])
    save("probas", probas)
    save("test_meta", te[[c for c in [fp.TARGET, fp.CLUSTER_COL, "periodo",
                                      *fp.PROTECTED.values()] if c in te.columns]])
    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa A lista para", modelo or "todos")


def _etapa_A_vieja():
    t0 = time.time()
    tr, va, te = partes()
    R = {"n_total": int(len(tr) + len(va) + len(te)),
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "cohortes_train": sorted(tr.periodo.astype(str).unique()),
         "cohorte_val": VAL, "cohorte_test": TEST,
         "tasa_base_train": float(tr[fp.TARGET].mean()),
         "tasa_base_test": float(te[fp.TARGET].mean()),
         "n_colegios": int(pd.concat([tr, va, te])[fp.CLUSTER_COL].nunique()),
         "preregistro": {"eje": fp.EJE_PRIMARIO, "metrica": fp.METRICA_PRIMARIA,
                         "capacidad": fp.CAPACIDAD_PRINCIPAL}}

    Xtr, pre = fp.build_matrix(tr, True)
    Xva, _ = fp.build_matrix(va, True)
    Xte, _ = fp.build_matrix(te, True)
    ytr, yva, yte = tr[fp.TARGET].values, va[fp.TARGET].values, te[fp.TARGET].values

    # Ajuste de hiperparametros sobre submuestra del entrenamiento, evaluando
    # en la cohorte de validacion completa. Se documenta en el manuscrito.
    idx = fp.rng.choice(len(Xtr), min(SUB_TUNING, len(Xtr)), replace=False)
    grid = {"regresion_logistica": [{"C": c} for c in (0.1, 1.0)],
            "xgboost": [{"n_estimators": ne, "max_depth": md, "learning_rate": 0.1,
                         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20}
                        for ne in (400,) for md in (6, 10)]}
    from sklearn.metrics import roc_auc_score
    tuning = {}
    for name, params in grid.items():
        mejor = (-1, None)
        for p in params:
            m = fp.build_estimator(name, p)
            pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr.iloc[idx], ytr[idx])
            auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
            if auc > mejor[0]:
                mejor = (auc, p)
        tuning[name] = {"params": mejor[1], "auc_val": float(mejor[0])}
        print(f"  [tuning] {name:22s} AUC_val={mejor[0]:.4f} {mejor[1]}", flush=True)
    R["tuning"] = tuning

    # Ajuste final sobre el entrenamiento COMPLETO
    perf, probas = {}, {}
    for name, info in tuning.items():
        pipe = Pipeline([("pre", pre), ("m", fp.build_estimator(name, info["params"]))])
        pipe.fit(Xtr, ytr)
        pr = pipe.predict_proba(Xte)[:, 1]
        pd_ = fp.predict_at_capacity(pr, fp.CAPACIDAD_PRINCIPAL)
        perf[name] = fp.evaluate(yte, pr, pd_)
        probas[name] = pr
        print(f"  {name:22s} AUC={perf[name]['auc_roc']:.4f} "
              f"AP={perf[name]['auc_pr']:.4f}", flush=True)
    R["desempeno_E0"] = perf
    R["mejor_modelo"] = max(perf, key=lambda k: perf[k]["auc_roc"])
    R["seg_etapa_A"] = round(time.time() - t0, 1)

    save("probas", probas)
    save("test_meta", te[[c for c in [fp.TARGET, fp.CLUSTER_COL, "periodo",
                                      *fp.PROTECTED.values()] if c in te.columns]])
    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa A lista")


# ---------------------------------------------------------------- etapa B
def etapa_B(n_boot=400, n_perm=400):
    n_boot, n_perm = int(n_boot), int(n_perm)
    R = json.load(open(f"{CK}/R.json"))
    te = load("test_meta")
    pr = load("probas")[R["mejor_modelo"]]
    y = te[fp.TARGET].values
    pred = fp.predict_at_capacity(pr, fp.CAPACIDAD_PRINCIPAL)
    save("pred0", pred)

    aud = {}
    clusters = te[fp.CLUSTER_COL].astype(str).values
    for eje, col in fp.PROTECTED.items():
        if col not in te.columns:
            continue
        g = te[col].astype(str).values
        m = g != "nan"
        tab = fp.group_rates(y[m], pred[m], g[m])
        tab.to_csv(f"{OUT}/tasas_E0_{eje}.csv", index=False)
        e = {"primario": eje == fp.EJE_PRIMARIO, "tabla": tab.to_dict("records"),
             "disparidades": fp.disparities(tab)}
        if eje == fp.EJE_PRIMARIO:
            e["inferencia_cluster"] = fp.cluster_bootstrap(
                y[m], pred[m], g[m], clusters[m], n_boot=n_boot, n_perm=n_perm)
        else:
            e["nota"] = "exploratorio: sin prueba de significancia (preregistro)"
        aud[eje] = e
        print(f"  {eje:14s} dif_FNR={e['disparidades']['dif_fnr']:.4f}", flush=True)
    R["auditoria_E0"] = aud

    inter = (te[fp.PROTECTED["zona"]].astype(str) + " x "
             + te["estrato_grupo"].astype(str)).values
    ok = ~pd.Series(inter).str.contains("nan").values
    ti = fp.group_rates(y[ok], pred[ok], inter[ok])
    ti.to_csv(f"{OUT}/tasas_interseccional.csv", index=False)
    R["auditoria_interseccional"] = {"tabla": ti.to_dict("records"),
                                     "disparidades": fp.disparities(ti)}
    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa B lista")


# ---------------------------------------------------------------- etapa C
def etapa_C():
    R = json.load(open(f"{CK}/R.json"))
    tr, va, te = partes()
    Xtr1, pre1 = fp.build_matrix(tr, False)
    Xte1, _ = fp.build_matrix(te, False)
    ytr, yte = tr[fp.TARGET].values, te[fp.TARGET].values
    best = R["mejor_modelo"]

    pipe = Pipeline([("pre", pre1),
                     ("m", fp.build_estimator(best, R["tuning"][best]["params"]))])
    pipe.fit(Xtr1, ytr)
    pr1 = pipe.predict_proba(Xte1)[:, 1]
    pred1 = fp.predict_at_capacity(pr1, fp.CAPACIDAD_PRINCIPAL)
    gz = te[fp.PROTECTED["zona"]].astype(str).values
    R["E1_ceguera"] = {"desempeno": fp.evaluate(yte, pr1, pred1),
                       "disparidades_zona": fp.disparities(fp.group_rates(yte, pred1, gz))}
    save("pred1", pred1)
    print(f"  E1 AUC={R['E1_ceguera']['desempeno']['auc_roc']:.4f} "
          f"dif_FNR={R['E1_ceguera']['disparidades_zona']['dif_fnr']:.4f}", flush=True)

    sub = te.sample(min(200_000, len(te)), random_state=fp.RANDOM_STATE)
    R["proxies"] = {e: fp.proxy_detection(sub, e) for e in fp.PROTECTED}
    print("  proxies:", {k: round(v["auc"], 3) for k, v in R["proxies"].items() if v},
          flush=True)
    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa C lista")


# ---------------------------------------------------------------- etapa D
def etapa_D(que="234"):
    R = json.load(open(f"{CK}/R.json"))
    tr, va, te = partes()
    Xtr1, pre1 = fp.build_matrix(tr, False)
    Xte1, _ = fp.build_matrix(te, False)
    ytr, yte = tr[fp.TARGET].values, te[fp.TARGET].values
    gz_tr = tr[fp.PROTECTED["zona"]].astype(str).values
    gz = te[fp.PROTECTED["zona"]].astype(str).values
    best = R["mejor_modelo"]

    if "2" in que:
        w = fp.reweighing_weights(ytr, gz_tr)
        p2 = Pipeline([("pre", pre1),
                       ("m", fp.build_estimator(best, R["tuning"][best]["params"]))])
        p2.fit(Xtr1, ytr, **{"m__sample_weight": w})
        pr2 = p2.predict_proba(Xte1)[:, 1]
        pred2 = fp.predict_at_capacity(pr2, fp.CAPACIDAD_PRINCIPAL)
        R["E2_reponderacion"] = {"desempeno": fp.evaluate(yte, pr2, pred2),
                                 "disparidades_zona": fp.disparities(fp.group_rates(yte, pred2, gz))}
        save("pred2", pred2)
        print(f"  E2 dif_FNR={R['E2_reponderacion']['disparidades_zona']['dif_fnr']:.4f}", flush=True)

    if "3" in que:
        try:
            from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
            idx = fp.rng.choice(len(Xtr1), min(150_000, len(Xtr1)), replace=False)
            base = fp.WeightedPipeline(pre=pre1, estimator=LogisticRegression(max_iter=800))
            eg = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds(), max_iter=6)
            eg.fit(Xtr1.iloc[idx], ytr[idx], sensitive_features=gz_tr[idx])
            pred3 = np.asarray(eg.predict(Xte1)).astype(int)
            R["E3_gradiente_exponenciado"] = {
                "tasa_seleccion": float(pred3.mean()),
                "n_entrenamiento": int(len(idx)),
                "disparidades_zona": fp.disparities(fp.group_rates(yte, pred3, gz))}
            save("pred3", pred3)
            print(f"  E3 dif_FNR={R['E3_gradiente_exponenciado']['disparidades_zona']['dif_fnr']:.4f}", flush=True)
        except Exception as e:
            R["E3_gradiente_exponenciado"] = {"error": str(e)[:200]}

    if "4" in que:
        try:
            from fairlearn.postprocessing import ThresholdOptimizer
            idx = fp.rng.choice(len(Xtr1), min(400_000, len(Xtr1)), replace=False)
            topt = ThresholdOptimizer(
                estimator=Pipeline([("pre", pre1), ("m", LogisticRegression(max_iter=800))]),
                constraints="equalized_odds", predict_method="predict_proba", prefit=False)
            topt.fit(Xtr1.iloc[idx], ytr[idx], sensitive_features=gz_tr[idx])
            pred4 = np.asarray(topt.predict(Xte1, sensitive_features=gz)).astype(int)
            R["E4_umbral_por_grupo"] = {
                "tasa_seleccion": float(pred4.mean()),
                "disparidades_zona": fp.disparities(fp.group_rates(yte, pred4, gz))}
            save("pred4", pred4)
            print(f"  E4 dif_FNR={R['E4_umbral_por_grupo']['disparidades_zona']['dif_fnr']:.4f}", flush=True)
        except Exception as e:
            R["E4_umbral_por_grupo"] = {"error": str(e)[:200]}

    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa D lista")


# ---------------------------------------------------------------- etapa E
def etapa_E():
    R = json.load(open(f"{CK}/R.json"))
    te = load("test_meta")
    pr = load("probas")[R["mejor_modelo"]]
    y = te[fp.TARGET].values
    gz = te[fp.PROTECTED["zona"]].astype(str).values

    R["sensibilidad_capacidad"] = {
        f"{c:.2f}": fp.disparities(fp.group_rates(
            y, fp.predict_at_capacity(pr, c), gz)) for c in (0.10, 0.20, 0.30)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = R["mejor_modelo"]
    pts = [("E0", R["desempeno_E0"][best]["auc_roc"],
            R["auditoria_E0"]["zona"]["disparidades"]["dif_fnr"])]
    for k, lab in [("E1_ceguera", "E1 blind"), ("E2_reponderacion", "E2 reweight"),
                   ("E3_gradiente_exponenciado", "E3 constraint"),
                   ("E4_umbral_por_grupo", "E4 thresholds")]:
        d = R.get(k, {})
        if "disparidades_zona" in d:
            pts.append((lab, d.get("desempeno", {}).get("auc_roc", np.nan),
                        d["disparidades_zona"]["dif_fnr"]))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for lab, auc, f in pts:
        if np.isfinite(auc):
            ax.scatter(f, auc, s=110)
            ax.annotate(lab, (f, auc), textcoords="offset points", xytext=(8, 5), fontsize=9)
        else:
            ax.axvline(f, ls=":", alpha=.6)
            ax.text(f, ax.get_ylim()[0], lab, rotation=90, fontsize=8, va="bottom")
    ax.set_xlabel("False negative rate disparity (urban vs rural)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Accuracy cost of reducing inequity\n(capacity = {fp.CAPACIDAD_PRINCIPAL:.0%}, "
                 f"test cohort {TEST})")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_curva_costo_equidad.png", dpi=300)
    plt.close(fig)

    json.dump(R, open(f"{OUT}/resultados.json", "w"), indent=2, ensure_ascii=False, default=str)
    json.dump(R, open(f"{CK}/R.json", "w"), indent=2, ensure_ascii=False, default=str)
    print("etapa E lista ->", OUT)


if __name__ == "__main__":
    {"A": etapa_A, "B": etapa_B, "C": etapa_C, "D": etapa_D, "E": etapa_E}[sys.argv[1]](
        *sys.argv[2:])
