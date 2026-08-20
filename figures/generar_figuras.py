"""Genera las cinco figuras del articulo a partir de los CSV de resultados.

Uso:
    python generar_figuras.py [--salida DIRECTORIO]

Todas las figuras se guardan a 300 ppp, con separador decimal espanol y una
paleta comun de dos grises; el rojo queda reservado a lineas de referencia y
anotaciones.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------- estilo comun
DK, GR, LT, AC = '#333333', '#8C8C8C', '#CFCFCF', '#B03A2E'
plt.rcParams.update({
    'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.edgecolor': '#555',
    'axes.labelcolor': DK, 'text.color': DK, 'xtick.color': DK, 'ytick.color': DK,
})
COMA1 = FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ','))
COMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))


def n3(v):
    return f'{v:.3f}'.replace('.', ',')


# ------------------------------------------------------------------ figura 1
def figura_1(res: Path, out: Path):
    """Brecha urbano-rural y validacion contra la serie publicada por el LEE.

    Las cifras de este estudio proceden de tabla1_composicion.csv; la serie del
    LEE se transcribe de los informes 92 y 98 y no es reproducible por codigo.
    """
    anios = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024], float)
    nan = np.nan
    propia = np.array([23.5, 23.5, nan, nan, 26.1, 24.8, 25.9])
    lee = np.array([nan, 24.1, 25.6, 26.0, 26.4, 25.2, nan])   # informes 92 y 98

    f, a = plt.subplots(figsize=(6.4, 3.3))
    a.axvspan(2019.5, 2021.5, color='#EFEFEF', zorder=0)
    a.plot(anios, lee, 's--', lw=1.6, ms=6, color=GR, label='Serie publicada (LEE)')
    a.plot(anios, propia, 'o-', lw=2.2, ms=7, color=DK, label='Este estudio')
    for x, y in zip(anios, propia):
        if y == y:
            a.annotate(f'{y:.1f}'.replace('.', ','), (x, y), textcoords='offset points',
                       xytext=(0, -16), ha='center', fontsize=9)
    a.text(2020.5, 22.2, '2020–2021\nexcluidas', ha='center', fontsize=8.5, color=GR)
    a.set_xticks(range(2018, 2025))
    a.set_xlabel('Cohorte')
    a.set_ylabel('Brecha urbano − rural (puntos)')
    a.set_ylim(21.5, 28.3)
    a.grid(alpha=.25)
    a.legend(fontsize=8.5, loc='lower right', framealpha=1, edgecolor='#BBB')
    a.yaxis.set_major_formatter(COMA)
    f.tight_layout()
    f.savefig(out / 'Figura_1.png', dpi=300, bbox_inches='tight')
    plt.close(f)


# ------------------------------------------------------------------ figura 2
def figura_2(res: Path, out: Path):
    """Diagrama del flujo de auditoria. Es un esquema, no depende de los datos."""
    W, H = 16.0, 5.2
    f, ax = plt.subplots(figsize=(11.4, 3.71))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    def caja(x, y, w, h, titulo, lineas, fc='#FFFFFF'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.09',
                                    fc=fc, ec=DK, lw=1.1))
        ax.text(x + w / 2, y + h - 0.30, titulo, ha='center', va='top',
                fontsize=8.8, fontweight='bold', color=DK)
        for i, l in enumerate(lineas):
            ax.text(x + w / 2, y + h - 0.70 - i * 0.36, l, ha='center', va='top',
                    fontsize=7.6, color=DK)

    def flecha(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=12, lw=1.3, color=DK))

    pasos = [
        ('1. Datos', ['2.774.006 registros', '5 cohortes · 11.230 colegios',
                      'Homologación de etiquetas'], '#FFFFFF'),
        ('2. Partición temporal', ['Entrena 2018 · 2019 · 2022', 'Valida 2023',
                                   'Prueba 2024'], '#FAFAFA'),
        ('3. Modelo', ['Logística vs. potenciación', 'del gradiente',
                       'AUC = 0,835'], '#FAFAFA'),
        ('4. Decisión', ['Marcar al 20%', 'de mayor riesgo',
                         '(capacidad, no umbral)'], LT),
        ('5. Escenarios', ['E0 completo · E1 ciego', 'E2 reponderación',
                           'E3 · E4 mitigación'], '#FAFAFA'),
    ]
    x = 0.15
    for i, (t, ls, fc) in enumerate(pasos):
        caja(x, 2.75, 2.94, 2.25, t, ls, fc)
        if i < 4:
            flecha(x + 2.94, 3.87, x + 3.14, 3.87)
        x += 3.14
    flecha(14.6, 2.75, 14.6, 2.15)

    ax.add_patch(FancyBboxPatch((0.15, 0.15), 15.7, 2.0,
                                boxstyle='round,pad=0.06,rounding_size=0.09',
                                fc='#F5F5F5', ec=DK, lw=1.1))
    ax.text(8.0, 1.92, '6. Auditoría', ha='center', va='top',
            fontsize=9.2, fontweight='bold', color=DK)
    bloques = [
        ('Tasas por grupo', 'tasa base, selección,', 'TFN, VPP'),
        ('Inferencia', 'remuestreo y permutación', 'de colegios, no estudiantes'),
        ('Variables sustitutas', '¿se recupera el atributo', 'que se eliminó?'),
        ('Interseccional y SHAP', 'zona × estrato,', 'contribución por grupo'),
    ]
    for i, (t, d1, d2) in enumerate(bloques):
        cx = 0.55 + i * 3.87
        ax.text(cx, 1.30, t, ha='left', va='top', fontsize=8.2, fontweight='bold', color=DK)
        ax.text(cx, 0.95, d1, ha='left', va='top', fontsize=7.4, color='#555')
        ax.text(cx, 0.62, d2, ha='left', va='top', fontsize=7.4, color='#555')
        if i < 3:
            ax.plot([cx + 3.55, cx + 3.55], [0.42, 1.45], color='#CCC', lw=1)
    f.savefig(out / 'Figura_2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(f)


# ------------------------------------------------------------------ figura 3
def figura_3(res: Path, out: Path):
    """Recuperacion de los atributos protegidos por clasificadores auxiliares."""
    p = pd.read_csv(res / 'tabla5_proxies.csv')
    col_auc = [c for c in p.columns if 'auc' in c.lower()][0]
    nombres = {'zona': 'Zona del colegio', 'naturaleza': 'Sector del colegio',
               'sector': 'Sector del colegio', 'sexo': 'Sexo', 'genero': 'Sexo'}
    etiquetas, valores = [], []
    for _, r in p.iterrows():
        clave = str(r.iloc[0]).lower()
        etiquetas.append(nombres.get(clave, str(r.iloc[0])))
        valores.append(float(r[col_auc]))

    # el eje preregistrado (zona) se resalta en gris claro
    colores = [LT if e.startswith('Zona') else DK for e in etiquetas]

    f, a = plt.subplots(figsize=(6.2, 2.8))
    b = a.barh(etiquetas[::-1], valores[::-1], .5,
               color=colores[::-1], edgecolor=DK, lw=1.1)
    for r, val in zip(b, valores[::-1]):
        a.text(val + .012, r.get_y() + r.get_height() / 2, n3(val),
               va='center', fontsize=11, fontweight='bold')
    a.axvline(.5, color=AC, ls='--', lw=1.3)
    a.text(.52, len(etiquetas) - 0.40, 'azar (0,5)', ha='left', va='center',
           fontsize=8.5, color=AC)
    a.set_xlim(0, 1.03)
    a.set_ylim(-.6, len(etiquetas) - 0.1)
    a.set_xlabel('AUC al recuperar el atributo eliminado')
    a.grid(alpha=.25, axis='x')
    a.xaxis.set_major_formatter(COMA1)
    f.tight_layout()
    f.savefig(out / 'Figura_3.png', dpi=300, bbox_inches='tight')
    plt.close(f)


# ------------------------------------------------------------------ figura 4
def figura_4(res: Path, out: Path):
    """Equidad frente a cobertura en los cinco escenarios de mitigacion."""
    t = pd.read_csv(res / 'tabla6_escenarios.csv')
    lab = ['E0\ncompleto', 'E1\nciego', 'E2\nreponderación',
           'E3\nrestricción', 'E4\numbral grupo']

    f, a = plt.subplots(figsize=(7.8, 3.6))
    x = np.arange(len(t)); w = .38
    a.axvspan(2.5, 4.5, color='#F2F2F2', zorder=0)
    a.bar(x - w / 2, t.dif_fnr, w, color=DK, edgecolor=DK, lw=1,
          label='Disparidad de TFN', zorder=3)
    a.bar(x + w / 2, t.recall_global, w, color=LT, edgecolor=DK, lw=1,
          label='Recall global (cobertura)', zorder=3)
    for i, (d, r) in enumerate(zip(t.dif_fnr, t.recall_global)):
        a.text(i - w / 2, d + .012, n3(d), ha='center', fontsize=8.5,
               fontweight='bold', zorder=4)
        a.text(i + w / 2, r + .012, n3(r), ha='center', fontsize=8.5,
               fontweight='bold', zorder=4)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=9)
    a.set_ylim(0, .80)
    a.grid(alpha=.25, axis='y')
    a.yaxis.set_major_formatter(COMA1)
    a.legend(fontsize=9, loc='upper left', framealpha=1, edgecolor='#BBB')
    a.annotate('paridad a costa de atender\na la mitad de los estudiantes',
               xy=(3.5, .32), xytext=(3.5, .72), ha='center', fontsize=8.5,
               style='italic', color=AC,
               arrowprops=dict(arrowstyle='->', color=AC, lw=1.1))
    f.tight_layout()
    f.savefig(out / 'Figura_4.png', dpi=300, bbox_inches='tight')
    plt.close(f)


# ------------------------------------------------------------------ figura 5
NOMBRES_SHAP = {
    'tasa_hist_colegio': 'Tasa histórica del colegio',
    'cole_jornada_NOCHE': 'Jornada nocturna',
    'cole_jornada_SABATINA': 'Jornada sabatina',
    'fami_trabajolabormadre_Es dueño de un negocio grande, tiene un cargo de '
    'nivel directivo o gerencial': 'Madre: cargo directivo',
    'fami_trabajolaborpadre_Trabaja en el hogar, no trabaja o estudia':
        'Padre: no trabaja o estudia',
    'estu_horassemanatrabaja': 'Horas trabajadas por semana',
    'fami_trabajolabormadre_No sabe': 'Madre: ocupación no sabe',
    'fami_trabajolabormadre_nan': 'Madre: ocupación sin dato',
    'estu_dedicacioninternet_No Navega Internet': 'No navega internet',
}


def figura_5(res: Path, out: Path, n_vars=9):
    """Contribucion absoluta media de los valores de Shapley por zona."""
    t = pd.read_csv(res / 'tabla7_shap.csv', index_col=0)
    t.index = [NOMBRES_SHAP.get(i, i[:28]) for i in t.index]
    t = t.head(n_vars).iloc[::-1]

    f, a = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(t)); h = .38
    a.barh(y - h / 2, t['URBANO'], h, color=LT, edgecolor=DK, lw=1, label='Urbano')
    a.barh(y + h / 2, t['RURAL'], h, color=DK, edgecolor=DK, lw=1, label='Rural')
    a.set_yticks(y); a.set_yticklabels(t.index, fontsize=9)
    a.set_xlabel('Valor absoluto medio de Shapley')
    a.legend(fontsize=9, loc='lower right', framealpha=1, edgecolor='#BBB')
    a.grid(alpha=.25, axis='x')
    a.set_xlim(0, 1.22)
    a.xaxis.set_major_formatter(COMA1)
    f.tight_layout()
    f.savefig(out / 'Figura_5.png', dpi=300, bbox_inches='tight')
    plt.close(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resultados', default='results/tables',
                    help='directorio con los CSV de resultados')
    ap.add_argument('--salida', default='results/figures',
                    help='directorio donde escribir las figuras')
    args = ap.parse_args()

    res, out = Path(args.resultados), Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    for nombre, fn in [('Figura 1', figura_1), ('Figura 2', figura_2),
                       ('Figura 3', figura_3), ('Figura 4', figura_4),
                       ('Figura 5', figura_5)]:
        try:
            fn(res, out)
            print(f'  {nombre}: generada')
        except FileNotFoundError as e:
            print(f'  {nombre}: OMITIDA, falta {e.filename}')
    print(f'figuras escritas en {out.resolve()}')


if __name__ == '__main__':
    main()
