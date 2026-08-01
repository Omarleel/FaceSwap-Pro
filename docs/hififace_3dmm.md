# Backend `hififace_3dmm`

## Flujo

```text
source image ─► reconstrucción 3DMM ─► identidad + forma source ─┐
                                                               ├─► generador HifiFace
 target crop ─► reconstrucción 3DMM ─► expresión + pose target ─┘
                                                               │
                                            Semantic Facial Fusion
                                                               │
                                                imagen + máscara aprendida
```

A diferencia del backend MediaPipe, los parámetros geométricos participan en el
vector de identidad que condiciona el decoder. La geometría no se añade únicamente
como deformación posterior.

## Adaptador

`XuehyHifiFaceRuntime` carga de forma diferida la implementación externa. El dominio
solo conoce el protocolo `HifiFaceRuntime`, por lo que puede sustituirse por otra
implementación, un servicio o una futura exportación ONNX/TensorRT.

El adaptador:

- valida el checkpoint generador y los modelos auxiliares;
- fuerza `use_ddp=False` para inferencia de una GPU;
- evita cargar LPIPS/VGG, que es una dependencia exclusiva de entrenamiento;
- usa la plantilla de alineación publicada por el runtime;
- admite de una a cuatro iteraciones;
- dilata la máscara aprendida para no recortar la fusión semántica interna.

## Límites conocidos

- salida nativa 256×256;
- inferencia PyTorch, no ONNX;
- el checkpoint público procede de una implementación no oficial;
- procesamiento independiente por frame;
- la propia implementación advierte que mirada y boca pueden degradarse en video,
  según el checkpoint elegido.

Estos límites se exponen como metadatos; no se ocultan bajo el nombre “3D perfecto”.
