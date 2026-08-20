# Datos

Este directorio está vacío en el repositorio. Los microdatos pesan cerca de
8,5 GB y pertenecen al Icfes, que los publica con sus propios términos de uso.

## 1. Microdatos de Saber 11 (obligatorios para el flujo completo)

**Fuente:** repositorio de datos abiertos DataIcfes, sección `04.Saber11`.

    https://www.icfes.gov.co/investigaciones/data-icfes/

Descargue las cohortes **2018-2, 2019-2, 2022-2, 2023-2 y 2024-2** y descomprima
los archivos en `data/raw/`. Las cohortes de 2020 y 2021 se excluyen del análisis
principal porque se aplicaron bajo condiciones de pandemia.

Estructura esperada:

    data/raw/
    ├── SB11_20182.TXT
    ├── SB11_20192.TXT
    ├── SB11_20222.TXT
    ├── SB11_20232.TXT
    └── SB11_20242.TXT

> **Advertencia sobre versiones.** El Icfes revisa retroactivamente los archivos
> publicados; el registro de cambios de mayo de 2026 documenta el recálculo de
> variables en cohortes que se remontan a 2016. Los resultados de este repositorio
> corresponden a la descarga del **1 de agosto de 2026**. Anote la fecha de su
> propia descarga: una posterior puede arrojar cifras distintas.

## 2. Fuentes auxiliares (opcionales)

Se usan para contexto y validación externa, no para entrenar el modelo.

| Fuente | Uso | Dónde obtenerla |
|---|---|---|
| Estadísticas municipales del MEN (`nudc-7mev`) | contexto municipal | https://www.datos.gov.co/Educaci-n/MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR-B-SICA/nudc-7mev |
| DIVIPOLA (códigos de municipio, DANE) | homologación geográfica | portal de datos abiertos del DANE |
| Informes 79, 92 y 98 del LEE (Javeriana) | validación externa de la brecha | https://www.javeriana.edu.co/lee |

El CSV municipal del MEN trae **cuatro defectos documentados** que `src/men_municipal.py`
corrige: pierde los ceros a la izquierda de los códigos de Antioquia y Atlántico,
lo que genera 1.271 municipios fantasma frente a los 1.123 reales, e incluye tres
filas agregadas con valor `NACIONAL` que no son municipios.

## 3. Directorios generados por el flujo

    data/interim/      columnas extraídas por cohorte
    data/processed/    muestra analítica consolidada

Ambos están en `.gitignore`.
