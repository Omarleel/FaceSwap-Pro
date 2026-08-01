# Postproceso asistido por malla MediaPipe

Backend: `insightface_inswapper_mediapipe_mesh`.

Este flujo conserva el generador base y aplica después:

1. estimación de 478 landmarks mediante Face Landmarker;
2. warp piecewise-affine del crop generado;
3. máscara construida desde el óvalo facial;
4. reducción gradual de opacidad en poses poco fiables.

No cambia la representación latente ni los coeficientes usados por INSwapper. Por
tanto, sus capacidades se registran así:

```text
geometry_conditioning = none
geometry_postprocess = mediapipe_mesh_warp
truly_3d_aware = false
```

Perfil:

```powershell
faceswap-pro run --config .\config\quality_mesh_assisted.yaml
```
