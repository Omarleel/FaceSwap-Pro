# Modelos y componentes de terceros

FaceSwap-Pro no incluye pesos de terceros.

## HifiFace

- El método HifiFace usa un extractor de identidad sensible a forma 3DMM y Semantic
  Facial Fusion.
- La implementación conectada es `xuehy/HiFiFace-pytorch`, no el repositorio oficial
  de los autores.
- El repositorio de implementación declara licencia MIT; los checkpoints, modelos
  auxiliares y conjuntos de entrenamiento pueden tener condiciones diferentes.
- Requiere Deep3DFaceRecon, un backbone ArcFace y Basel Face Model (BFM).
- BFM debe obtenerse directamente bajo sus condiciones.

## MediaPipe Face Landmarker

- Se usa únicamente en el backend asistido por malla.
- Produce landmarks, pose, blendshapes y matriz facial.
- No genera ni condiciona la identidad sintética.

## InsightFace / INSwapper

- Se conserva como backend rápido y como servicio de detección/reconocimiento.
- La licencia del código y las condiciones de los modelos entrenados no son
  necesariamente iguales.

## Seguridad de checkpoints

PyTorch tradicionalmente carga checkpoints serializados. No abras pesos de orígenes
no confiables. Conserva hashes y procedencia en el manifiesto del proyecto.
