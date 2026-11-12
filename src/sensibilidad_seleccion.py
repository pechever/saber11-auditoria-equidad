"""Analisis de sensibilidad a la seleccion muestral (observacion 2 del evaluador).

El articulo afirmaba que las disparidades estimadas son una COTA INFERIOR,
porque los estudiantes que desertaron antes del grado once no estan en los
datos. El evaluador objeto que la afirmacion no estaba sustentada, y tiene
razon: el argumento se habia escrito suponiendo que la disparidad perjudicaba
a los rurales, pero el hallazgo es el contrario.

Aqui se formaliza. Sea:

    m   el numero de estudiantes rurales ausentes necesarios para que la
        participacion rural pase del 17,2% observado al 26,7% de la poblacion
        en edad escolar;
    phi la proporcion de esos ausentes que el modelo marcaria.

Suponiendo que todos los ausentes son de bajo desempeno (son los
educativamente mas debiles), la TFN rural corregida es

    TFN_r(phi) = [FN_r + (1 - phi) m] / [P_r + m]

y la brecha corregida es TFN_u - TFN_r(phi). Igualando a la brecha observada
se obtiene un umbral de equilibrio phi* por encima del cual la brecha real
es MAYOR que la medida, y por debajo del cual es menor o se invierte.

Uso:
    python src/sensibilidad_seleccion.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fairness_pipeline as fp
from rerun_revision import cargar

SHARE_POBLACION = 0.267      # poblacion rural en edad escolar, informe LEE 79
OUT = Path('results/revision')


def main():
    a = cargar('base')
    yte, gz = a['yte'], a['gz']
    pred = fp.predict_at_capacity(cargar('E0')['pr'], fp.CAPACIDAD_PRINCIPAL)

    rural = gz == 'RURAL'
    n_r, n_tot = int(rural.sum()), len(gz)
    P_r = int(yte[rural].sum())                       # rurales de bajo desempeno
    FN_r = int(((pred == 0) & (yte == 1) & rural).sum())
    tfn_r = FN_r / P_r
    tfn_u = float(((pred == 0) & (yte == 1) & ~rural).sum() / yte[~rural].sum())
    brecha = tfn_u - tfn_r

    # ausentes necesarios para alcanzar la participacion poblacional
    m = int(round((SHARE_POBLACION * n_tot - n_r) / (1 - SHARE_POBLACION)))

    def tfn_corregida(phi):
        return (FN_r + (1 - phi) * m) / (P_r + m)

    # umbral de equilibrio: brecha corregida == brecha observada
    # TFN_r(phi*) = tfn_r  =>  phi* = 1 - (tfn_r (P_r+m) - FN_r) / m
    phi_eq = 1 - (tfn_r * (P_r + m) - FN_r) / m

    print(f'rurales observados      {n_r:,} de {n_tot:,}  ({n_r / n_tot:.1%})')
    print(f'ausentes para el {SHARE_POBLACION:.1%}   m = {m:,}')
    print(f'TFN urbana {tfn_u:.3f}   TFN rural {tfn_r:.3f}   brecha {brecha:.3f}')
    print(f'\numbral de equilibrio phi* = {phi_eq:.3f}')
    print('  por encima, la brecha real es MAYOR que la medida (cota inferior)')
    print('  por debajo, es menor, y puede invertirse\n')

    escenarios = []
    for phi in (0.0, 0.25, 0.50, phi_eq, 0.75, 0.90, 1.0):
        t = tfn_corregida(phi)
        escenarios.append({'phi': round(float(phi), 3),
                           'tfn_rural': round(float(t), 3),
                           'brecha': round(float(tfn_u - t), 3)})
    print(pd.DataFrame(escenarios).to_string(index=False))

    # referencia empirica: ¿a que tasa marca el modelo a los rurales mas
    # desfavorecidos, que son los que mas se parecen a quienes desertaron?
    inter = pd.read_csv('results/tables/tasas_interseccional.csv')
    rur = inter[inter.grupo.str.startswith('RURAL')].copy()
    rur['tpr'] = 1 - rur['fnr']
    print('\nTPR del modelo por subgrupo rural (referencia para phi):')
    print(rur[['grupo', 'n', 'tasa_base', 'tpr']].sort_values('tpr')
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sensibilidad_seleccion.json').write_text(json.dumps({
        'supuestos': {
            'share_poblacion_rural': SHARE_POBLACION,
            'todos_los_ausentes_son_bajo_desempeno': True,
            'fuente_share': 'LEE, Informe 79'},
        'observado': {'n_rural': n_r, 'n_total': n_tot, 'P_rural': P_r,
                      'FN_rural': FN_r, 'tfn_rural': tfn_r, 'tfn_urbano': tfn_u,
                      'brecha': brecha},
        'ausentes_m': m,
        'phi_equilibrio': float(phi_eq),
        'escenarios': escenarios,
        'tpr_subgrupos_rurales': rur[['grupo', 'n', 'tasa_base', 'tpr']].to_dict('records'),
    }, indent=1, ensure_ascii=False))
    print(f'\nguardado en {OUT / "sensibilidad_seleccion.json"}')


if __name__ == '__main__':
    main()
