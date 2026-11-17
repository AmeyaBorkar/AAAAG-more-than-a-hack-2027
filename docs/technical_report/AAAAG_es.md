---
lang: es
anchor: es
outline: Español
header: AAAAG · Posesión del balón a partir del seguimiento automático
footer: AAAAG · More than a Hack 2027 · Página {page} de {total}
title: Posesión del balón fotograma a fotograma a partir del seguimiento automático
meta: Equipo AAAAG: Ameya Borkar, Arnav, Bakshi, Ayush · 17 de noviembre de 2026
meta: More than a Hack 2027, fase 1
meta: Repositorio: REPO_URL
crosslink: en | English version: page {page:en}
---

::: summary
**Resumen**

1. **Qué hace.** Pequeñas redes neuronales leen el seguimiento automático fotograma a fotograma. Un decodificador convierte su salida en interrupciones y posesiones completas.
2. **Resultado.** En los tres partidos públicos de Metrica, con etiquetas que nuestro código deriva de sus eventos, obtiene **0,788** de spellF1. El algoritmo base de los organizadores obtiene **0,407**. Nunca entrenamos con esos partidos, pero orientaron el diseño: la cifra es optimista y no comparable con la clasificación.
3. **Idea clave.** Evidencia aprendida de partidos externos etiquetados, decodificada con una distribución a priori de duraciones; solo la detección aprendida del balón muerto aportó +0,060 en Metrica.
4. **Limitación.** Se desconoce la convención de la referencia para posesiones cortas; con cinco convenciones plausibles, la puntuación varía unos ±0,05.
5. **Reproducción.** Un comando, `python -m aaaag submit`, reconstruye el archivo enviado byte a byte en 6 a 12 minutos con la CPU de un portátil.
:::

## 1. Aprender y decodificar supera las reglas

**Tarea.** Cada fotograma evaluado necesita H, A o D, aunque falta el balón en el 41 % (organizadores). La métrica de la clasificación, spellF1, empareja posesiones completas una a una por solapamiento temporal; un etiquetado perfecto 1 s tarde obtiene solo 0,6539. La puntuación depende de hallar cada posesión y sus bordes.

**Arquitectura.** Pasamos las coordenadas 0–1 de los organizadores a metros con el campo de su games.py, 105,3 × 68,0 m. Las redes dan a cada fotograma P(H, A, D): local, visitante o balón muerto. Un decodificador las convierte en posesiones; tres reglas fijas cierran las etiquetas (figura 1).

::: figure
Seguimiento | jugadores y balón (m)
Redes | 3 × 2 vistas | P(H, A, D)
Balón muerto | P(D) ≥ 0,30 es interrupción
Decodificador | H o A por tramo vivo
Reglas fijas | saque, toque | umbral 1,5 s
group: 1 | entrada
group: 2 | aprendido
group: 3-4 | decodificación estructurada
group: 5 | fijo
caption: **Figura 1.** Evidencia aprendida primero, reglas fijas al final; el decodificador ve cada parte entera.
:::

**Red.** Tres redes de unos 0,8 millones de parámetros leen el partido dos veces, una con el campo reflejado sobre su eje largo. Cada fotograma da 23 tokens: los 11 jugadores de cada equipo más cercanos al balón, y el balón. Los jugadores ausentes son tokens vacíos; nunca se usa la identidad de las pistas. Atención entre tokens y una convolución temporal dan ±5,1 s de contexto.

**Entradas y salidas.** Las entradas son posiciones, velocidades, visibilidad del balón y el último equipo que lo tuvo. Además de P(H, A, D), nombra al jugador con el balón (salida de poseedor) y puntúa un cambio de posesión (salida de cambio).

**Decisión 1: evidencia aprendida.** Sustituyó a las reglas manuales porque ganó con claridad (tabla 3). En los tres partidos públicos de Metrica (sección 2), su salida de balón muerto ganó **+0,060 de spellF1**. En IDSSE, siete partidos alemanes con etiquetas del proveedor por fotograma, su equipo aprendido ganó **+0,073**.

**Decisión 2: decodificar posesiones.** La métrica castiga la fragmentación, así que el decodificador impone duraciones plausibles. Una puerta de balón muerto marca las interrupciones; una segmentación semi-Markov divide luego el juego vivo en H y A. Superó a un voto mayoritario de ±0,5 s por **+0,018** en 5 de 5 partidos.

**Puntuaciones.** Cada posesión suma su evidencia de equipo y el logaritmo de la probabilidad de su duración. La distribución a priori es lognormal en segundos, ajustada con los cinco partidos de desarrollo de IDSSE (mediana 3,7 s). Cada cambio suma 1,5 veces los log-odds de la salida de cambio (acotados a ±6,9): un traspaso no visto cuesta hasta 10 unidades.

**Decisión 3: especialistas y promedio.** Dos fallos cuestan más de 0,05 cada uno: un balón visible en una interrupción y un balón en el aire. Una sola cámara sitúa el balón aéreo donde su línea de visión corta el césped. Entrenar desde cero con balones muertos visibles costó 0,010 con entrada normal, así que ajustamos copias de una red. La copia base aprende balones muertos visibles; tres especialistas, además, balones aéreos.

**Promedio de pesos.** Cada red entregada promedia los pesos de la copia base (½) y de los tres especialistas (⅙ cada uno), de un mismo origen (Wortsman et al., 2022). Ese origen se preentrenó con PFF FC y se entrenó después con IDSSE.

## 2. Todas las etiquetas proceden de otros partidos

**Excluido.** Ni etiquetas ni parámetros proceden del partido proporcionado, su vídeo o momentos concretos. Sus cinco minutos públicos solo pasan por una prueba de humo que imprime PASS o FAIL.

Table: **Tabla 1.** Cada conjunto etiquetado, quién lo etiquetó, cuánto y para qué.
| Conjunto (licencia) | Etiquetado por | Cantidad | Uso |
|---|---|---|---|
| [IDSSE](https://doi.org/10.6084/m9.figshare.28196177) (CC BY 4.0) | Proveedor de la DFL, por fotograma | 7 partidos de unos 145 000 fotogramas | Entrenamiento; 5 diseño, 2 reservados |
| [PFF FC, Mundial 2022](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) (términos de PFF) | Nuestro código, eventos de PFF FC | 44 + 16 partidos | Preentrenamiento; 16 reservados |
| [Datos de muestra de Metrica](https://github.com/metrica-sports/sample-data) (piden cita) | Nuestro código, eventos etiquetados a mano | 3 partidos | Ajuste del simulador; prueba |
| [Datos abiertos de SkillCorner](https://github.com/SkillCorner/opendata) (MIT) | SkillCorner, por fotograma | 20 partidos | Entrenamiento: −0,033 en IDSSE, descartado |
Note: Agradecemos a PFF FC (datos usados según sus términos, sin redistribuir), a DFL y Sportec Solutions por IDSSE (Bassek et al., 2025), a Metrica Sports, a SkillCorner y a los organizadores. Nada con licencia no comercial o GPL entra en el sistema.

**Etiquetas de eventos.** Nuestro código convierte los eventos de Metrica y PFF FC en etiquetas por fotograma, por tanto aproximadas. Metrica: una interrupción va del final del evento previo a un balón parado hasta este. Si no, el último pase, conducción, recuperación, tiro o pérdida indica el equipo. PFF FC: un evento de balón fuera abre una interrupción y el siguiente evento con balón o saque la cierra; solo estos cambian el equipo.

**Convenciones.** Una convención decide qué tramos cortos de un equipo conserva la referencia. Probamos cinco: cada toque, al menos 1 s, al menos 2 s, 1,5 s o un pase, e interrupciones de menos de 3 s ignoradas. Las reglas de tiempo y de pase siguen los indicadores de CHALLENGE.md: retener el balón, completar pases.

**Degradación.** Como el balón de IDSSE nunca falta, lo degrada un simulador ajustado con los tres partidos de Metrica. Oculta el balón en las interrupciones y abre huecos en juego vivo con las tasas de Metrica. Elimina el 5–20 % de las posiciones de jugadores y añade 0,10–0,50 m de ruido. Su balón falta en el 44,5 % de los fotogramas jugados, frente al 37,3 % de Metrica.

**Validación.** Cada experimento se declaró, con su regla de éxito, antes de ejecutarse. Cada partido de IDSSE se puntúa con redes entrenadas sin él. Nunca entrenamos con Metrica, pero el simulador copia sus huecos y Metrica confirmó la salida de balón muerto y el umbral de 1,5 s. Por eso sus cifras son optimistas.

## 3. Cerca del doble del método base

**Resultados.** En Metrica obtenemos 0,788 de spellF1 y el algoritmo base de los organizadores, 0,407, con las mismas etiquetas. Su 0,2433 publicado es ese método base en el partido proporcionado, así que las cifras no son comparables. Superamos al método base en los cuatro bancos por 0,28 a 0,39 (tabla 2), unas siete veces el ruido de ±0,04 por partido (organizadores).

**Bancos de prueba.** IDSSE lleva huecos de balón simulados; promediamos tres pares de redes, cada par dos redes de semillas distintas, promediadas como en producción. Metrica se puntúa también con el balón suavizado en 8 fotogramas, porque cada proveedor suaviza distinto. PFF FC: 16 partidos reservados.

Table: **Tabla 2.** spellF1, media (peor partido), en partidos con los que ningún sistema entrenó.
| Sistema | IDSSE | Metrica | Metrica, suavizado | PFF FC |
|---|---:|---:|---:|---:|
| Método base | 0,348 | 0,407 (0,391) | 0,384 (0,364) | 0,354 (0,285) |
| Reglas manuales | 0,516 | 0,498 (0,458) | 0,371 (0,339) | 0,359 (0,310) |
| **Nuestro sistema** | **0,742** | **0,788 (0,771)** | **0,740 (0,724)** | **0,632 (0,568)** |
Note: 5, 3, 3 y 16 partidos; no se guardaron los valores por partido de IDSSE de nuestro sistema. Método base: jugador más cercano, voto de ±2 s. Reglas: último equipo a menos de 1 m del balón durante 3 fotogramas, un modelo manual de balón en juego, voto de ±0,5 s. IDSSE también guio el diseño: unos 0,03 de más (L4).

Table: **Tabla 3.** Qué aporta cada idea, en spellF1, frente a una alternativa más simple.
| Idea | Alternativa más simple | Banco de prueba | Ganancia |
|---|---|---|---:|
| Equipo aprendido | Regla del último en controlar | IDSSE, 5 partidos | +0,073 |
| Balón muerto aprendido | Modelo manual del balón | Metrica, 3 partidos | +0,060 |
| Semi-Markov | Voto mayoritario de ±0,5 s | IDSSE, 5 partidos | +0,018 |
| Promedio de pesos | Solo la copia base | IDSSE, balón aéreo | +0,046 |
Note: Las ganancias se solapan; cada una se midió al añadir su idea. El equipo aprendido ganó en 4 de 5 partidos y empató en el único cuyos equipos no estaban en ningún partido de entrenamiento. El decodificador ganó en 5 de 5. El promedio ganó en 3 de 3 pares de redes; ninguna otra media bajó más de 0,003.

## 4. El seguimiento es nuestra única entrada

**Entradas.** Solo usamos el seguimiento táctico y los metadatos de los organizadores (tabla 4). Cada otro recurso era redundante o no superó una prueba declarada de antemano.

Table: **Tabla 4.** Recursos usados y descartados.
| Recurso | Estado | Uso, o la prueba que no superó |
|---|---|---|
| Seguimiento táctico | Usado | En metros; equipo por pista; sin identidades |
| Metadatos, games.py | Usado | Tamaño del campo, partes evaluadas, saques (R6) |
| Smart Tagging | Descartado | Etiquetas simuladas: +0,003, umbral +0,005 |
| Vídeo y audio | Descartado | Detectores de balón y de silbato bajo umbral |
| Homografía | Innecesaria | El seguimiento ya está en el campo |
| Señal de televisión | No usada | No sincronizada (organizadores) |
| Modelo PathCRF | No usado | Entrenado con IDSSE ([Kim et al., 2026](https://arxiv.org/abs/2602.12080)): fuga |
Note: Las etiquetas se simularon sobre IDSSE; los detectores se probaron con el vídeo proporcionado, así que no damos cifras.

## 5. Fallos simulados enseñan el seguimiento ruidoso

**Balón ausente.** Leer un balón ausente como muerto declara muertos los huecos del juego vivo. Ocultamos el balón al margen de las etiquetas en la mitad de cada lote de entrenamiento, y ese atajo deja de servir. En huecos largos simulados, la red declara muerto el 1–2 % del juego vivo; el modelo manual, el 22–29 %.

Table: **Tabla 5.** Fallos del seguimiento automático y cómo los trata el sistema.
| Fallo | Tratamiento | Efecto (spellF1) |
|---|---|---|
| Balón oculto al reanudar | Ocultarlo al entrenar | +0,013 |
| Balón visible parado | Ajuste: balón muerto visible | +0,057 a +0,062 |
| Balón aéreo en el césped | Especialistas, cámara 15–25 m | +0,046 a 18 m |
| Balón suavizado | Suavizado; salida de poseedor | +0,062 a 8 fotogramas |
| Pistas fragmentadas | Cercanos, sin identidades | −0,0004: sin coste |
| Fragmentos del otro equipo | Ninguno eficaz | −0,009 |
Note: Las ganancias comparan entrenar con y sin el tratamiento; los costes, entrada limpia y con fallo, en IDSSE. Balón aéreo: la altura real del balón de IDSSE vista por una cámara a 18 m. Los organizadores indican 46 pistas para 22 jugadores.

## 6. La posesión solo cambia con evidencia sostenida

**R1. Balón muerto.** La salida de balón muerto, entrenada con el estado arbitral del balón que da el proveedor, marca interrupción donde P(D) ≥ 0,30. Las interrupciones duran al menos 2 s; los tramos vivos, al menos 1 s. Frente a 0,40 la ganancia es pequeña, +0,003, pero se mantuvo en los cuatro pares de redes y en cada convención limpia: conservamos 0,30.

**R2. Reanudaciones.** Ninguna regla nombra al equipo que reanuda: la red ve cada reanudación con ±5,1 s de contexto. La primera posesión tras una interrupción queda exenta de R5: cada reanudación tiene un poseedor claro (CHALLENGE.md). Una regla del equipo que reanuda costó 0,014.

**R3. Balones sueltos.** Un balón suelto o disputado sigue con el último equipo (CHALLENGE.md). El decodificador solo cambia de equipo si coinciden la evidencia de equipo y la salida de cambio: un balón disputado no cambia la posesión. La red sabe qué equipo tuvo el último jugador a menos de 1 m del balón durante 3 fotogramas.

**R4. Toques sin control.** Desvíos y duelos aéreos ganados sin control no cambian nada. Un cambio real se desplaza al toque más cercano del equipo que gana el balón, dentro de ±0,25 s. Un toque es un cambio brusco de la velocidad del balón junto a un jugador.

**R5. Posesiones cortas.** Una posesión de menos de 1,5 s solo sobrevive si dos compañeros tienen el balón a menos de 1 m durante ella: un pase completado. Si no, toma el equipo de la posesión anterior, de la más corta a la más larga, y se une a sus vecinas. En Metrica, con redes anteriores, el umbral de 1,5 s tuvo la mejor media en las cinco convenciones: 0,793 frente a 0,741 sin él.

**R6. Fuera de evaluación.** X fuera de las partes evaluadas por los organizadores (games.py), y D en fotogramas evaluados antes del saque de cada parte, como señala su FAQ. Cada fotograma evaluado recibe H, A o D, porque spellF1 sustituye una X por la etiqueta anterior.

## 7. La convención es el mayor riesgo

**L1. Convención, unos ±0,05.** Cómo trata la referencia las posesiones cortas pesa más que cualquier cambio de modelo. En IDSSE, el mejor decodificador para las convenciones de toque, 1 s y 2 s obtiene de 0,78 a 0,855. Con una referencia por toque, nuestro umbral de 1,5 s cuesta unos 0,06; así limitamos el peor caso.

**L2. Pruebas simuladas.** La ganancia con balón aéreo procede de una cámara simulada a 18 m; una real puede diferir. Un balón suavizado aún cuesta unos 0,05 en Metrica (tabla 2). Con más tiempo: vídeo con balones aéreos anotados.

**L3. Balón oculto.** En huecos largos, una red que solo ve a los jugadores acierta el equipo tanto como la nuestra; entrenar con esos huecos no ayudó. Los tramos de equipo erróneo dentro de posesiones largas siguen siendo el error principal. Con más tiempo: un indicador de toque robusto al suavizado.

**L4. Datos limitados.** Los siete partidos de IDSSE comparten proveedor; cinco, equipo local. Dos partidos reservados de IDSSE dieron a una versión anterior 0,710, frente a 0,736 en desarrollo (como la tabla 2). Con más tiempo: otras ligas.

## 8. Un comando reconstruye el archivo exacto

**Pasos.** Clonar REPO_URL y ejecutar tres comandos con CPython 3.13; 3.12 y 3.14 dan los mismos bytes. Las redes están en models/, fijadas por SHA-256: sin GPU ni entrenamiento.

```
python scripts/bootstrap.py
export MTAH_DATA_DIR=/path/to/bundle
.venv/bin/python -m aaaag submit
# Windows PowerShell: las dos últimas líneas son
$env:MTAH_DATA_DIR = "C:\path\to\bundle"
.venv\Scripts\python -m aaaag submit
```

1. **Entorno.** Un .venv con las versiones de requirements-lock.txt.
2. **Datos.** MTAH_DATA_DIR es el paquete de los organizadores, tal cual.
3. **Construcción.** submit escribe y valida outputs/submission/AAAAG.csv.
4. **Duración.** Unos **6 a 12 minutos** con la CPU de un portátil.
5. **Verificación.** SHA-256, igual en otro portátil con 1 y 8 hilos: c57e6e7762864ca736cc339bdc49af1b399c9c259dfcfb93eca98e67d8e30cf2

**Reentrenamiento.** La sección 7 de REPORT.md da los comandos: torch, datos externos, unas 2 horas por ajuste en la GPU de 12 GB de un portátil. Reentrenar reproduce las puntuaciones, no los bytes.

**Referencias.** Bassek et al. (2025), IDSSE, Scientific Data. Kim et al. (2026), PathCRF, [arXiv:2602.12080](https://arxiv.org/abs/2602.12080). Wortsman et al. (2022), model soups, ICML.
