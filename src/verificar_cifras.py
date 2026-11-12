# Coteja las cifras de las tablas del manuscrito contra los CSV de resultados.
import re, pandas as pd

txt = open('rui_contenido.md', encoding='utf-8').read()

def filas(cabecera_col0):
    """Devuelve las filas #TAB del bloque cuya primera celda de encabezado coincide."""
    bloques, act = [], None
    for ln in txt.split('\n'):
        if ln.startswith('#TAB '):
            celdas = [c.strip() for c in ln[5:].split('|')]
            if act is None:
                act = [celdas]
            else:
                act.append(celdas)
        elif act is not None:
            bloques.append(act); act = None
    if act: bloques.append(act)
    for b in bloques:
        if b[0][0] == cabecera_col0:
            return b
    return None

def num(s):
    return float(s.replace('.', '').replace(',', '.'))

fallos = []
def cmp(etiqueta, doc, real, dec=3):
    esperado = round(real, dec)
    if abs(doc - esperado) > 10 ** -dec / 2 + 1e-12:
        fallos.append(f'{etiqueta}: documento {doc} vs datos {esperado}')

# --- Tabla 7: escenarios de mitigacion ---
esc = pd.read_csv('resultados_reales/tabla6_escenarios_revision.csv')
b = filas('Escenario')
for i, fila in enumerate(b[1:]):
    r = esc.iloc[i]
    for j, col in enumerate(['auc_roc', 'tasa_seleccion', 'recall_global',
                             'fnr_urbano', 'fnr_rural', 'dif_fnr'], start=1):
        cmp(f'Tabla 7 {fila[0]} {col}', num(fila[j]), r[col])

# --- Tabla 5: tasas por zona del colegio ---
z = pd.read_csv('resultados_reales/tasas_E0_zona.csv').set_index('grupo')
b = filas('Grupo')
for fila in b[1:]:
    r = z.loc[fila[0].upper()]
    if int(fila[1].replace('.', '')) != int(r['n']):
        fallos.append(f"Tabla 5 {fila[0]} n: documento {fila[1]} vs datos {int(r['n'])}")
    for j, col in enumerate(['tasa_base', 'tasa_seleccion', 'tpr', 'fnr', 'fpr', 'ppv'], start=2):
        cmp(f'Tabla 5 {fila[0]} {col}', num(fila[j]), r[col])

# --- Tabla 6: interseccional zona x estrato ---
it = pd.read_csv('resultados_reales/tasas_interseccional.csv').set_index('grupo')
b6 = filas('Grupo')
for bloque in [x for x in [b6] if x]:
    pass
inter = None
for ln in txt.split('\n'):
    if ln.startswith('#TAB Urbano x Bajo'):
        inter = True
if inter:
    for ln in txt.split('\n'):
        if not ln.startswith('#TAB ') or ' x ' not in ln:
            continue
        c = [x.strip() for x in ln[5:].split('|')]
        clave = c[0].replace('Urbano', 'URBANO').replace('Rural', 'RURAL')
        if clave not in it.index:
            fallos.append(f'Tabla 6 grupo ausente en los datos: {c[0]}')
            continue
        r = it.loc[clave]
        if int(c[1].replace('.', '')) != int(r['n']):
            fallos.append(f"Tabla 6 {c[0]} n: documento {c[1]} vs datos {int(r['n'])}")
        for j, col in enumerate(['tasa_base', 'tasa_seleccion', 'fnr', 'ppv'], start=2):
            cmp(f'Tabla 6 {c[0]} {col}', num(c[j]), r[col])

# --- coherencia interna: disparidad = |TFN urbano - TFN rural| ---
b = filas('Escenario')
for fila in b[1:]:
    d = abs(num(fila[4]) - num(fila[5]))
    if abs(d - num(fila[6])) > 0.0015:
        fallos.append(f'Tabla 7 {fila[0]}: |{fila[4]}-{fila[5]}| = {d:.3f} != {fila[6]}')

# --- cifras sueltas citadas en el texto ---
e0, e1, e2 = esc.iloc[0], esc.iloc[1], esc.iloc[2]
red = (e0.dif_fnr - e2.dif_fnr) / e0.dif_fnr * 100
if 'un 36%' in txt and not (35.5 <= red < 36.5):
    fallos.append(f'reduccion E2 declarada 36% vs real {red:.1f}%')
pp = (e0.recall_global - e2.recall_global) * 100
if '0,6 puntos porcentuales de recall' in txt and not (0.55 <= pp < 0.65):
    fallos.append(f'costo en recall declarado 0,6 pp vs real {pp:.2f} pp')

# --- separador decimal en el texto: no deben quedar numeros con punto ---
cuerpo = '\n'.join(l for l in txt.split('\n') if not l.startswith(('#R ', '#AF', '#ABS', '#KWEN')))
sospechosos = set(re.findall(r'\b0\.\d+\b', cuerpo))
if sospechosos:
    fallos.append(f'decimales con punto en el texto en espanol: {sorted(sospechosos)}')

print('=== verificacion de cifras ===')
if fallos:
    for f in fallos:
        print(' FALLA', f)
else:
    print(' todas las cifras de las tablas coinciden con los resultados')
print(f'({len(fallos)} discrepancias)')
