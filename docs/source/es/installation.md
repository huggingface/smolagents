# Opciones de instalación

La biblioteca `smolagents` se puede instalar usando pip. Existen varias formas y opciones disponibles para realizar la instalación.

## Requisitos Previos
- Python 3.10 o una versión más reciente
- Gestor de paquetes para Python: [`pip`](https://pip.pypa.io/en/stable/) o [`uv`](https://docs.astral.sh/uv/)

## Entorno Virtual

Instalar `smolagents` en un entorno virtual de Python es altamente recomendable. Los entornos virtuales permiten mantener las dependencias 
de tu proyecto aisladas tanto de otros proyectos como de Python en el sistema, evitando conflictos de versiones y simplificando la administración de paquetes.

<hfoptions id="virtual-environment">
<hfoption id="venv">
Usando [`venv`](https://docs.python.org/3/library/venv.html):
```bash
python -m venv .venv
source .venv/bin/activate
```
</hfoption>
<hfoption id="uv">

Usando [`uv`](https://docs.astral.sh/uv/):
```bash
uv venv .venv
source .venv/bin/activate
```
</hfoption>
</hfoptions>

## Instalación Básica

Para instalar la biblioteca principal (core) de smolagents, usa:

<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install smolagents
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install smolagents
```
</hfoption>
</hfoptions>

## Instalación con Complementos

Existen dependencias adicionales (extras) en `smolagents` que puedes instalar conforme a tus necesidades.
La instalación de estos extras se realiza con la siguiente sintaxis:

<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[extra1,extra2]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[extra1,extra2]"
```
</hfoption>
</hfoptions>

### Herramientas

Estos complementos incluyen diversas herramientas e integraciones:

<hfoptions id="installation">
<hfoption id="pip">
- **toolkit**: Instala un paquete estándar de herramientas para tareas habituales.
  ```bash
  pip install "smolagents[toolkit]"
  ```
- **mcp**: Incorpora el Protocolo de Contexto de Modelo (MCP) para facilitar la integración de herramientas y servicios externos.
  ```bash
  pip install "smolagents[mcp]"
  ```
</hfoption>
<hfoption id="uv">
- **toolkit**: Instala un paquete estándar de herramientas para tareas habituales.
  ```bash
  uv pip install "smolagents[toolkit]"
  ```
- **mcp**: Incorpora el Protocolo de Contexto de Modelo (MCP) para facilitar la integración de herramientas y servicios externos.
  ```bash
  uv pip install "smolagents[mcp]"
  ```
</hfoption>
</hfoptions>

### Integración de Modelos

Las funcionalidades adicionales facilitan la conexión con diversos modelos y frameworks de inteligencia artificial.

<hfoptions id="installation">
<hfoption id="pip">
- **openai**: Integración para los modelos de OpenAI a través de API.
  ```bash
  pip install "smolagents[openai]"
  ```
- **transformers**: Permite el uso de modelos Transformers de Hugging Face.
  ```bash
  pip install "smolagents[transformers]"
  ```
- **vllm**: Agrega compatibilidad con vLLM para una inferencia de modelos más eficiente.
  ```bash
  pip install "smolagents[vllm]"
  ```
- **mlx-lm**: Incorpora funcionalidades específicas para MLX-LM.
  ```bash
  pip install "smolagents[mlx-lm]"
  ```
- **litellm**: Habilita el uso de LiteLLM en tareas de inferencia con modelos optimizados.
  ```bash
  pip install "smolagents[litellm]"
  ```
- **bedrock**:  Amplía la compatibilidad con servicios de modelos alojados en AWS Bedrock.
  ```bash
  pip install "smolagents[bedrock]"
  ```
</hfoption>
<hfoption id="uv">
- **openai**: Integración para los modelos de OpenAI a través de API.
  ```bash
  uv pip install "smolagents[openai]"
  ```
- **transformers**: Permite el uso de modelos Transformers de Hugging Face.
  ```bash
  uv pip install "smolagents[transformers]"
  ```
- **vllm**: Agrega compatibilidad con vLLM para una inferencia de modelos más eficiente.
  ```bash
  uv pip install "smolagents[vllm]"
  ```
- **mlx-lm**: Incorpora funcionalidades específicas para MLX-LM.
  ```bash
  uv pip install "smolagents[mlx-lm]"
  ```
- **litellm**: Habilita el uso de LiteLLM en tareas de inferencia con modelos optimizados.
  ```bash
  uv pip install "smolagents[litellm]"
  ```
- **bedrock**: Amplía la compatibilidad con servicios de modelos alojados en AWS Bedrock.
  ```bash
  uv pip install "smolagents[bedrock]"
  ```
</hfoption>
</hfoptions>

### Capacidades Multimodales

Funciones adicionales para procesar varios tipos de datos:

<hfoptions id="installation">
<hfoption id="pip">
- **vision**: Despliega funciones avanzadas para el procesamiento de imágenes y visión por computadora.
  ```bash
  pip install "smolagents[vision]"
  ```
- **audio**: Incorpora soporte para tareas de procesamiento de audio.
  ```bash
  pip install "smolagents[audio]"
  ```
</hfoption>
<hfoption id="uv">
- **vision**: Despliega funciones avanzadas para el procesamiento de imágenes y visión por computadora.
  ```bash
  uv pip install "smolagents[vision]"
  ```
- **audio**: Incorpora soporte para tareas de procesamiento de audio.
  ```bash
  uv pip install "smolagents[audio]"
  ```
</hfoption>
</hfoptions>

### Ejecución Remota

Extensiones para ejecutar código a distancia:

<hfoptions id="installation">
<hfoption id="pip">
- **docker**: Funcionalidad para ejecutar scripts en entornos Docker.
  ```bash
  pip install "smolagents[docker]"
  ```
- **e2b**: Facilita la ejecución remota mediante soporte E2B.
  ```bash
  pip install "smolagents[e2b]"
  ```
</hfoption>
<hfoption id="uv">
- **docker**: Funcionalidad para ejecutar scripts en entornos Docker.
  ```bash
  uv pip install "smolagents[docker]"
  ```
- **e2b**: Facilita la ejecución remota mediante soporte E2B.
  ```bash
  uv pip install "smolagents[e2b]"
  ```
</hfoption>
</hfoptions>

### Telemetría e Interfaz de Usuario

Módulos complementarios para telemetría, monitoreo y diseño de interfaz:

<hfoptions id="installation">
<hfoption id="pip">
- **telemetry**: Agrega funcionalidades para actividades de monitoreo y trazabilidad.
  ```bash
  pip install "smolagents[telemetry]"
  ```
- **gradio**: Permite la utilización de componentes interactivos en Gradio UI.
  ```bash
  pip install "smolagents[gradio]"
  ```
</hfoption>
<hfoption id="uv">
- **telemetry**: Agrega funcionalidades para actividades de monitoreo y trazabilidad.
  ```bash
  uv pip install "smolagents[telemetry]"
  ```
- **gradio**: Permite la utilización de componentes interactivos en Gradio UI.
  ```bash
  uv pip install "smolagents[gradio]"
  ```
</hfoption>
</hfoptions>

### Instalación Completa

Para instalar todos los complementos disponibles, puedes usar:

<hfoptions id="installation">
<hfoption id="pip">
```bash
pip install "smolagents[all]"
```
</hfoption>
<hfoption id="uv">
```bash
uv pip install "smolagents[all]"
```
</hfoption>
</hfoptions>

## Verificación de la Instalación

Después de la instalación, puedes verificar que `smolagents` esté instalado correctamente ejecutando:

```python
import smolagents
print(smolagents.__version__)
```

## Próximos Pasos

Una vez que `smolagents` esté instalado correctamente, puedes:

- Aprende los conceptos básicos revisando el [Tutorial Guiado](guided_tour).
- Explora los ejemplos prácticos y aplicaciones en las [Guías Prácticas](examples/text_to_sql).
- Profundiza en los conceptos avanzados mediante las [Guías Conceptuales](conceptual_guides/intro_agents).
- Revisa los [Tutoriales](tutorials/building_good_agents) para el desarrollo de agentes.
- Consulta la [Documentación API](./reference/index) para obtener información detallada sobre clases y funciones.
