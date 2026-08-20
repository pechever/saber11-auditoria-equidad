#!/usr/bin/env bash
# Extrae una cohorte del zip de DataIcfes y se queda solo con las columnas del estudio.
#
# Estrategia: filtrar columnas con `cut` (C, rapido) ANTES de tocar pandas.
# Leer los 422 MB completos con pandas excede el limite de tiempo por llamada;
# tras el cut el archivo baja a ~15% y se procesa en segundos.
#
# Uso: ./extraer_cohortes.sh 20182
set -euo pipefail

COH="$1"
ZIP=/sessions/keen-epic-mccarthy/mnt/uploads/OneDrive_1_1-8-2026.zip
OUT=/tmp/saber   # espacio de trabajo: el mount de outputs no permite borrar
TXT="$OUT/Examen_Saber_11_${COH}.txt"
SMALL="$OUT/small_${COH}.csv"
DST="$OUT/saber11_${COH}.pkl.gz"

mkdir -p "$OUT"
[ -f "$DST" ] && { echo "$COH ya procesado"; exit 0; }

KEEP="periodo,estu_consecutivo,cole_area_ubicacion,cole_naturaleza,cole_calendario,cole_caracter,cole_jornada,cole_genero,cole_bilingue,cole_depto_ubicacion,cole_cod_dane_establecimiento,cole_cod_mcpio_ubicacion,estu_genero,estu_horassemanatrabaja,estu_dedicacionlecturadiaria,estu_dedicacioninternet,estu_etnia,estu_discapacidad,estu_repite,estu_inse_individual,estu_nse_individual,estu_nse_establecimiento,fami_estratovivienda,fami_educacionmadre,fami_educacionpadre,fami_personashogar,fami_cuartoshogar,fami_numlibros,fami_situacioneconomica,fami_trabajolabormadre,fami_trabajolaborpadre,fami_tieneautomovil,fami_tienecomputador,fami_tieneinternet,fami_tienelavadora,fami_tienemotocicleta,fami_tieneserviciotv,fami_tieneconsolavideojuegos,fami_tienehornomicroogas,fami_comecarnepescadohuevo,fami_comecerealfrutoslegumbre,fami_comelechederivados,punt_global"

[ -f "$TXT" ] || unzip -o -j "$ZIP" "Examen_Saber_11_${COH}.txt" -d "$OUT" > /dev/null

# indices de las columnas deseadas, segun el encabezado real de ESTA cohorte
IDX=$(head -1 "$TXT" | tr ';' '\n' | nl -ba | awk -v keep="$KEEP" '
  BEGIN{n=split(keep,a,","); for(i=1;i<=n;i++) want[a[i]]=1}
  {gsub(/^[ \t]+/,"",$0); split($0,f,"\t"); name=$2; if(name in want) printf "%s%s", (out++?",":""), $1}')

cut -d';' -f"$IDX" "$TXT" > "$SMALL"
rm -f "$TXT"                      # liberar disco de inmediato

python3 - "$SMALL" "$DST" <<'PY'
import sys, os, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, sep=';', dtype=str, low_memory=False)
df.to_pickle(dst, compression='gzip')
print(f"  filas={len(df):,}  columnas={df.shape[1]}  pkl={os.path.getsize(dst)/1e6:.1f} MB")
PY

rm -f "$SMALL"
echo "$COH listo"
