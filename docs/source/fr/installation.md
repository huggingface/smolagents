# Options d'installation

La librairie `smolagents` peut être installée en utilisant pip. Voici quelques méthodes d'installation et options différentes.

## Prérequis
- Python 3.10 ou plus récent
- Gestionnaire de paquets Python (Python Package Manager) : [`pip`](https://pip.pypa.io/en/stable/) ou [`uv`](https://docs.astral.sh/uv/)

## Environnement virtuel

Il est très recommandé d'installer `smolagents` dans un environnement virtuel Python.
Un environnement virtuel isole les dépendances de votre projet d'autres projets Python et de l'installation système de python,
empêchant les conflits de versions et rendant la gestion de paquets plus fiable.

<hfoptions id="virtual-environment">
<hfoption id="venv">

Avec [`venv`](https://docs.python.org/3/library/venv.html):

```bash
python -m venv .venv
source .venv/bin/activate
```

</hfoption>
<hfoption id="uv">

Avec [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate
```

</hfoption>
</hfoptions>

## Installation de base

Installez la bibliothèque principale `smolagents` avec :

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

## Installation avec options supplémentaires

`smolagents` propose plusieurs dépendances optionnelles (extras) qui peuvent être installées en fonction de vos besoins.
Vous pouvez installer ces extras en utilisant la syntaxe suivante :
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

### Outils
Ces extras incluent différents outils et intégrations :
<hfoptions id="installation">
<hfoption id="pip">
- **toolkit** : Installe un ensemble par défaut d'outils pour les tâches courantes.
  ```bash
  pip install "smolagents[toolkit]"
  ```
- **mcp** : Ajoute la prise en charge du Model Context Protocol (MCP) pour s'intégrer à des outils et services externes.
  ```bash
  pip install "smolagents[mcp]"
  ```
</hfoption>
<hfoption id="uv">
- **toolkit** : Installe un ensemble par défaut d'outils pour les tâches courantes.
  ```bash
  uv pip install "smolagents[toolkit]"
  ```
- **mcp** : Ajoute la prise en charge du Model Context Protocol (MCP) pour s'intégrer à des outils et services externes.
  ```bash
  uv pip install "smolagents[mcp]"
  ```
</hfoption>
</hfoptions>

### Intégration de modèles
Ces extras permettent l'intégration avec différents modèles et frameworks d'IA :
<hfoptions id="installation">
<hfoption id="pip">
- **openai** : Ajoute la prise en charge des modèles de l'API OpenAI.
  ```bash
  pip install "smolagents[openai]"
  ```
- **transformers** : Active l'utilisation des modèles Hugging Face Transformers.
  ```bash
  pip install "smolagents[transformers]"
  ```
- **vllm** : Ajoute la prise en charge de VLLM pour une inférence de modèles efficace.
  ```bash
  pip install "smolagents[vllm]"
  ```
- **mlx-lm** : Active la prise en charge des modèles MLX-LM.
  ```bash
  pip install "smolagents[mlx-lm]"
  ```
- **litellm** : Ajoute la prise en charge de LiteLLM pour une inférence de modèles légère.
  ```bash
  pip install "smolagents[litellm]"
  ```
- **bedrock** : Active la prise en charge des modèles AWS Bedrock.
  ```bash
  pip install "smolagents[bedrock]"
  ```
</hfoption>
<hfoption id="uv">
- **openai** : Ajoute la prise en charge des modèles de l'API OpenAI.
  ```bash
  uv pip install "smolagents[openai]"
  ```
- **transformers** : Active l'utilisation des modèles Hugging Face Transformers.
  ```bash
  uv pip install "smolagents[transformers]"
  ```
- **vllm** : Ajoute la prise en charge de VLLM pour une inférence de modèles efficace.
  ```bash
  uv pip install "smolagents[vllm]"
  ```
- **mlx-lm** : Active la prise en charge des modèles MLX-LM.
  ```bash
  uv pip install "smolagents[mlx-lm]"
  ```
- **litellm** : Ajoute la prise en charge de LiteLLM pour une inférence de modèles légère.
  ```bash
  uv pip install "smolagents[litellm]"
  ```
- **bedrock** : Active la prise en charge des modèles AWS Bedrock.
  ```bash
  uv pip install "smolagents[bedrock]"
  ```
</hfoption>
</hfoptions>

### Capacités multimodales
Extras pour la prise en charge de différents types de médias et d'entrées :
<hfoptions id="installation">
<hfoption id="pip">
- **vision** : Ajoute la prise en charge du traitement d'images et des tâches de vision par ordinateur.
  ```bash
  pip install "smolagents[vision]"
  ```
- **audio** : Active les capacités de traitement audio.
  ```bash
  pip install "smolagents[audio]"
  ```
</hfoption>
<hfoption id="uv">
- **vision** : Ajoute la prise en charge du traitement d'images et des tâches de vision par ordinateur.
  ```bash
  uv pip install "smolagents[vision]"
  ```
- **audio** : Active les capacités de traitement audio.
  ```bash
  uv pip install "smolagents[audio]"
  ```
</hfoption>
</hfoptions>

### Exécution distante
Extras pour exécuter du code à distance :
<hfoptions id="installation">
<hfoption id="pip">
- **blaxel** : Ajoute la prise en charge des sandboxes Blaxel - des machines virtuelles à démarrage rapide avec hibernation (recommandé).
  ```bash
  pip install "smolagents[blaxel]"
  ```
- **e2b** : Active la prise en charge de E2B pour l'exécution à distance.
  ```bash
  pip install "smolagents[e2b]"
  ```
- **docker** : Ajoute la prise en charge de l'exécution de code dans des conteneurs Docker.
  ```bash
  pip install "smolagents[docker]"
  ```
</hfoption>
<hfoption id="uv">
- **blaxel** : Ajoute la prise en charge des sandboxes Blaxel - des machines virtuelles à démarrage rapide avec hibernation (recommandé).
  ```bash
  uv pip install "smolagents[blaxel]"
  ```
- **e2b** : Active la prise en charge de E2B pour l'exécution à distance.
  ```bash
  uv pip install "smolagents[e2b]"
  ```
- **docker** : Ajoute la prise en charge de l'exécution de code dans des conteneurs Docker.
  ```bash
  uv pip install "smolagents[docker]"
  ```
</hfoption>
</hfoptions>

### Télémétrie et interface utilisateur
Extras pour la télémétrie, la supervision et les composants d'interface utilisateur :
<hfoptions id="installation">
<hfoption id="pip">
- **telemetry** : Ajoute la prise en charge de la supervision et du traçage.
  ```bash
  pip install "smolagents[telemetry]"
  ```
- **gradio** : Ajoute la prise en charge des composants d'interface interactifs Gradio.
  ```bash
  pip install "smolagents[gradio]"
  ```
</hfoption>
<hfoption id="uv">
- **telemetry** : Ajoute la prise en charge de la supervision et du traçage.
  ```bash
  uv pip install "smolagents[telemetry]"
  ```
- **gradio** : Ajoute la prise en charge des composants d'interface interactifs Gradio.
  ```bash
  uv pip install "smolagents[gradio]"
  ```
</hfoption>
</hfoptions>

### Installation complète
Pour installer tous les extras disponibles, vous pouvez utiliser :
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

## Vérifier l'installation
Après l'installation, vous pouvez vérifier que `smolagents` est correctement installé en exécutant :
```python
import smolagents
print(smolagents.__version__)
```

## Prochaines étapes
Une fois que vous avez correctement installé `smolagents`, vous pouvez :
- Suivre la [visite guidée](./guided_tour) pour apprendre les bases.
- Explorer les [guides pratiques](./examples/text_to_sql) pour des exemples concrets.
- Lire les [guides conceptuels](./conceptual_guides/intro_agents) pour des explications plus théoriques.
- Consulter les [tutoriels](./tutorials/building_good_agents) pour des explications détaillées sur la création d'agents.
- Parcourir la [référence API](./reference/index) pour des informations détaillées sur les classes et fonctions.
