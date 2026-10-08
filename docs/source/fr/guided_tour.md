# Agents - Visite guidée

[[open-in-colab]]

Dans cette visite guidée, vous allez apprendre comment construire un agent, comment l'exécuter et comment le personnaliser pour qu'il fonctionne au mieux pour votre cas d'usage.

## Choisir un type d'agent : CodeAgent ou ToolCallingAgent

`smolagents` fournit deux classes d'agents : [`CodeAgent`] et [`ToolCallingAgent`], qui représentent deux paradigmes différents pour la façon dont les agents interagissent avec les outils.
La différence clé réside dans la manière dont les actions sont spécifiées et exécutées : génération de code vs appels d'outils structurés.

- [`CodeAgent`] génère des appels d'outils sous forme de fragments de code Python.
  - Le code est exécuté soit localement (potentiellement non sécurisé), soit dans un sandbox sécurisé.
  - Les outils sont exposés comme des fonctions Python (via des bindings).
  - Exemple d'appel d'outil :
    ```py
    result = search_docs("What is the capital of France?")
    print(result)
    ```
  - Atouts :
    - Très expressif : permet une logique complexe et du contrôle de flux, peut combiner des outils, boucler, transformer, raisonner.
    - Flexible : pas besoin de prédéfinir toutes les actions possibles, peut générer dynamiquement de nouvelles actions/outils.
    - Raisonnement émergent : idéal pour les problèmes en plusieurs étapes ou à logique dynamique.
  - Limites :
    - Risque d'erreurs : il faut gérer les erreurs de syntaxe et les exceptions.
    - Moins prévisible : plus sujet à des sorties inattendues ou non sûres.
    - Nécessite un environnement d'exécution sécurisé.

- [`ToolCallingAgent`] écrit des appels d'outils au format JSON structuré.
  - C'est le format courant utilisé dans de nombreux frameworks (API OpenAI, etc.), qui permet des interactions structurées avec les outils sans exécuter de code.
  - Les outils sont définis par un schéma JSON : nom, description, types de paramètres, etc.
  - Exemple d'appel d'outil :
    ```json
    {
      "tool_call": {
        "name": "search_docs",
        "arguments": {
          "query": "What is the capital of France?"
        }
      }
    }
    ```
  - Atouts :
    - Fiable : moins enclin aux hallucinations, les sorties sont structurées et validées.
    - Sûr : les arguments sont strictement validés, aucun risque d'exécution de code arbitraire.
    - Interopérable : facile à relier à des API ou services externes.
  - Limites :
    - Faible expressivité : ne permet pas facilement de combiner ou de transformer dynamiquement les résultats, ni d'implémenter une logique ou un contrôle de flux complexes.
    - Peu flexible : il faut définir toutes les actions possibles à l'avance, limité aux outils prédéfinis.
    - Pas de synthèse de code : limité aux capacités des outils.

Quand utiliser quel type d'agent :
- Utilisez [`CodeAgent`] lorsque :
  - Vous avez besoin de raisonnement, de chaînage ou de composition dynamique.
  - Les outils sont des fonctions qui peuvent être combinées (par ex. parsing + maths + requêtes).
  - Votre agent est un « problem solver » ou un programmeur.

- Utilisez [`ToolCallingAgent`] lorsque :
  - Vous avez des outils simples et atomiques (par ex. appeler une API, récupérer un document).
  - Vous voulez une grande fiabilité et une validation claire.
  - Votre agent joue le rôle de dispatcher ou de contrôleur.

## CodeAgent

[`CodeAgent`] génère des fragments de code Python pour réaliser des actions et résoudre des tâches.

Par défaut, l'exécution du code Python se fait dans votre environnement local.
Cela devrait être sûr, car les seules fonctions qui peuvent être appelées sont les outils que vous avez fournis (en particulier s'il ne s'agit que d'outils Hugging Face) et un ensemble de fonctions prédéfinies sûres comme `print` ou des fonctions du module `math`. Vous êtes donc déjà limité quant à ce qui peut être exécuté.

L'interpréteur Python n'autorise pas non plus par défaut les imports en dehors d'une liste sûre, donc toutes les attaques les plus évidentes ne devraient pas poser problème.
Vous pouvez autoriser des imports supplémentaires en passant les modules autorisés comme liste de chaînes de caractères dans l'argument `additional_authorized_imports` lors de l'initialisation de votre [`CodeAgent`] :

```py
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

De plus, comme couche de sécurité supplémentaire, l'accès aux sous-modules est interdit par défaut, sauf s'il est explicitement autorisé dans la liste des imports.
Par exemple, pour accéder au sous-module `numpy.random`, vous devez ajouter `'numpy.random'` à la liste `additional_authorized_imports`.
Cela peut aussi être autorisé en utilisant `numpy.*`, ce qui autorisera `numpy` ainsi que tout sous-paquet comme `numpy.random` et ses propres sous-paquets.

> [!WARNING]
> Le LLM peut générer du code arbitraire qui sera ensuite exécuté : n'ajoutez pas d'importations non sûres !

L'exécution s'arrêtera sur tout code qui tente de réaliser une opération illégale ou en cas d'erreur Python classique dans le code généré par l'agent.

Vous pouvez aussi utiliser [Blaxel](https://blaxel.ai), [E2B](https://e2b.dev/docs#what-is-e2-b) ou Docker à la place d'un interpréteur Python local. Pour Blaxel, commencez par [définir les variables d'environnement `BL_API_KEY` et `BL_WORKSPACE`](https://app.blaxel.ai/profile/security), puis passez `executor_type="blaxel"` lors de l'initialisation de l'agent. Pour E2B, commencez par [définir la variable d'environnement `E2B_API_KEY`](https://e2b.dev/dashboard?tab=keys), puis passez `executor_type="e2b"`. Pour Docker, passez `executor_type="docker"`.


> [!TIP]
> Pour en savoir plus sur l'exécution de code, consultez [ce tutoriel](tutorials/secure_code_execution).

### ToolCallingAgent

[`ToolCallingAgent`] produit des appels d'outils au format JSON, qui est le format courant utilisé dans de nombreux frameworks (API OpenAI, etc.), ce qui permet des interactions structurées avec les outils sans exécution de code. Nous utilisons l'outil intégré WebSearchTool (fourni par l'extra toolkit de smolagents, qui sera décrit plus loin) pour permettre à notre agent d'effectuer des recherches web.   

Il fonctionne de manière très similaire à [`CodeAgent`], évidemment sans `additional_authorized_imports` puisqu'il n'exécute pas de code :

```py
from smolagents import ToolCallingAgent, WebSearchTool

agent = ToolCallingAgent(tools=[WebSearchTool()], model=model)
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

## Utiliser la CLI

Vous pouvez démarrer rapidement avec smolagents en utilisant l'interface en ligne de commande :

```bash
# Exécuter avec un prompt direct et des options
smolagent "Plan a trip to Tokyo, Kyoto and Osaka between Mar 28 and Apr 7."  --model-type "InferenceClientModel" --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" --imports "pandas numpy" --tools "web_search"

# Exécuter en mode interactif : se lance lorsqu'aucun prompt n'est fourni et vous guide pour choisir les arguments
smolagent
```

## Construire votre agent

Pour initialiser un agent minimal, vous avez besoin au moins de ces deux arguments :

- `model`, un modèle de génération de texte qui alimente votre agent – car l'agent est différent d'un simple LLM : c'est un système qui utilise un LLM comme moteur. Vous pouvez utiliser l'une des options suivantes :
    - [`TransformersModel`] prend un pipeline `transformers` pré‑initialisé pour faire l'inférence en local avec `transformers`.
    - [`InferenceClientModel`] s'appuie sur un `huggingface_hub.InferenceClient` et supporte tous les Inference Providers du Hub : Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, et plus encore.
    - [`LiteLLMModel`] vous permet également d'appeler plus de 100 modèles et fournisseurs différents via [LiteLLM](https://docs.litellm.ai/) !
    - [`AzureOpenAIModel`] vous permet d'utiliser les modèles OpenAI déployés sur [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service).
    - [`AmazonBedrockModel`] vous permet d'utiliser Amazon Bedrock sur [AWS](https://aws.amazon.com/bedrock/?nc1=h_ls).
    - [`MLXModel`] crée un pipeline [mlx-lm](https://pypi.org/project/mlx-lm/) pour exécuter l'inférence en local.

- `tools`, une liste de `Tools` que l'agent peut utiliser pour résoudre la tâche. Elle peut être vide. Vous pouvez aussi ajouter la boîte à outils par défaut en plus de votre liste `tools` en définissant l'argument optionnel `add_base_tools=True`.

Une fois que vous avez ces deux arguments, `tools` et `model`, vous pouvez créer un agent et l'exécuter. Vous pouvez utiliser le LLM de votre choix, que ce soit via les [Inference Providers](https://huggingface.co/blog/inference-providers), [transformers](https://github.com/huggingface/transformers/), [ollama](https://ollama.com/), [LiteLLM](https://www.litellm.ai/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), [Amazon Bedrock](https://aws.amazon.com/bedrock/?nc1=h_ls) ou [mlx-lm](https://pypi.org/project/mlx-lm/).

Toutes les classes de modèles acceptent des paramètres supplémentaires (comme `temperature`, `max_tokens`, `top_p`, etc.) directement à l'instanciation.
Ces paramètres sont automatiquement transmis aux appels de complétion du modèle sous‑jacent, ce qui vous permet de configurer le comportement du modèle (créativité, longueur de réponse, stratégie d'échantillonnage, etc.).

<hfoptions id="Pick a LLM">
<hfoption id="Inference Providers">

Les Inference Providers nécessitent un `HF_TOKEN` pour s'authentifier, mais un compte HF gratuit inclut déjà des crédits. Passez en PRO pour augmenter votre quota de crédits inclus.

Pour accéder aux modèles restreints ou augmenter vos limites de débit avec un compte PRO, vous devez définir la variable d'environnement `HF_TOKEN` ou passer le paramètre `token` à l'initialisation de `InferenceClientModel`. Vous pouvez récupérer votre token sur votre [page de paramètres](https://huggingface.co/settings/tokens).

```python
from smolagents import CodeAgent, InferenceClientModel

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = InferenceClientModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>") # Vous pouvez choisir de ne pas passer de model_id à InferenceClientModel pour utiliser un modèle par défaut
# vous pouvez aussi spécifier un provider particulier, par ex. provider="together" ou provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Local Transformers Model">

```python
# !pip install 'smolagents[transformers]'
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="OpenAI or Anthropic API">

Pour utiliser `LiteLLMModel`, vous devez définir la variable d'environnement `ANTHROPIC_API_KEY` ou `OPENAI_API_KEY`, ou passer le paramètre `api_key` lors de l'initialisation.

```python
# !pip install 'smolagents[litellm]'
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY") # Vous pouvez aussi utiliser 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Ollama">

```python
# !pip install 'smolagents[litellm]'
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2", # Ce modèle est un peu faible pour des comportements agentiques complexes
    api_base="http://localhost:11434", # remplacez par 127.0.0.1:11434 ou par un serveur distant compatible OpenAI si nécessaire
    api_key="YOUR_API_KEY", # remplacez par une clé API si nécessaire
    num_ctx=8192, # la valeur par défaut d'ollama est 2048, ce qui échouera sur des tâches un peu longues. 8192 fonctionne pour les tâches simples, plus est mieux. Consultez https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator pour estimer la VRAM nécessaire pour le modèle choisi.
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Azure OpenAI">

Pour se connecter à Azure OpenAI, vous pouvez utiliser directement `AzureOpenAIModel`, ou utiliser `LiteLLMModel` et le configurer en conséquence.

Pour initialiser une instance d'`AzureOpenAIModel`, vous devez passer le nom de déploiement de votre modèle, puis soit passer les arguments `azure_endpoint`, `api_key` et `api_version`, soit définir les variables d'environnement `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` et `OPENAI_API_VERSION`.

```python
# !pip install 'smolagents[openai]'
from smolagents import CodeAgent, AzureOpenAIModel

model = AzureOpenAIModel(model_id="gpt-4o-mini")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

De manière similaire, vous pouvez configurer `LiteLLMModel` pour se connecter à Azure OpenAI comme suit :

- passez le nom de déploiement de votre modèle comme `model_id` et préfixez‑le avec `azure/`
- assurez‑vous de définir la variable d'environnement `AZURE_API_VERSION`
- soit passez les arguments `api_base` et `api_key`, soit définissez les variables d'environnement `AZURE_API_KEY` et `AZURE_API_BASE`

```python
import os
from smolagents import CodeAgent, LiteLLMModel

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-35-turbo-16k-deployment" # exemple de nom de déploiement

os.environ["AZURE_API_KEY"] = "" # api_key
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2024-10-01-preview"

model = LiteLLMModel(model_id="azure/" + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
   "Could you give me the 118th number in the Fibonacci sequence?",
)
```

</hfoption>
<hfoption id="Amazon Bedrock">

La classe `AmazonBedrockModel` fournit une intégration native avec Amazon Bedrock, autorisant des appels d'API directs et une configuration complète.

Utilisation basique :

```python
# !pip install 'smolagents[bedrock]'
from smolagents import CodeAgent, AmazonBedrockModel

model = AmazonBedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

Configuration avancée :

```python
import boto3
from smolagents import AmazonBedrockModel

# Créer un client Bedrock personnalisé
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY'
)

additional_api_config = {
    "inferenceConfig": {
        "maxTokens": 3000
    },
    "guardrailConfig": {
        "guardrailIdentifier": "identify1",
        "guardrailVersion": 'v1'
    },
}

# Initialisation avec une configuration complète
model = AmazonBedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
    client=bedrock_client,  # Utiliser le client personnalisé
    **additional_api_config
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

Utiliser LiteLLMModel :

Vous pouvez aussi utiliser `LiteLLMModel` avec les modèles Bedrock :

```python
from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(model_name="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
agent = CodeAgent(tools=[], model=model)

agent.run("Explain the concept of quantum computing")
```

</hfoption>
<hfoption id="mlx-lm">

```python
# !pip install 'smolagents[mlx-lm]'
from smolagents import CodeAgent, MLXModel

mlx_model = MLXModel("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
agent = CodeAgent(model=mlx_model, tools=[], add_base_tools=True)

agent.run("Could you give me the 118th number in the Fibonacci sequence?")
```

</hfoption>
</hfoptions>

### Gestion des paramètres de modèle

Lors de l'initialisation des modèles, vous pouvez passer des arguments nommés qui seront transmis comme paramètres de complétion à l'API du modèle sous‑jacent lors de l'inférence.

Pour un contrôle fin de la gestion des paramètres, la valeur sentinelle `REMOVE_PARAMETER` vous permet d'exclure explicitement
des paramètres qui pourraient autrement être définis par défaut ou transmis d'ailleurs :

```python
from smolagents import OpenAIModel, REMOVE_PARAMETER

# Supprimer le paramètre "stop"
model = OpenAIModel(
    model_id="gpt-5",
    stop=REMOVE_PARAMETER,  # Garantit que "stop" n'est pas inclus dans les appels d'API
    temperature=0.7
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
```

C'est particulièrement utile lorsque :
- Vous voulez surcharger des paramètres par défaut qui pourraient être appliqués automatiquement.
- Vous devez vous assurer que certains paramètres sont complètement exclus des appels d'API.
- Vous voulez laisser le fournisseur du modèle utiliser ses propres valeurs par défaut pour certains paramètres.

## Configuration avancée de l'agent

### Personnaliser les conditions de terminaison de l'agent

Par défaut, un agent continue de s'exécuter jusqu'à ce qu'il appelle la fonction `final_answer` ou atteigne le nombre maximal d'étapes.
Le paramètre `final_answer_checks` vous donne davantage de contrôle sur le moment et la façon dont un agent termine son exécution :

```python
from smolagents import CodeAgent, InferenceClientModel

# Définir une fonction de vérification personnalisée pour la réponse finale
def is_integer(final_answer: str, agent_memory=None) -> bool:
    """Return True if final_answer is an integer."""
    try:
        int(final_answer)
        return True
    except ValueError:
        return False

# Initialiser un agent avec une vérification personnalisée de la réponse finale
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    final_answer_checks=[is_integer]
)

agent.run("Calculate the least common multiple of 3 and 7")
```

Le paramètre `final_answer_checks` accepte une liste de fonctions qui chacune :
- prennent la `final_answer` de l'agent et l'agent lui‑même comme paramètres ;
- renvoient un booléen indiquant si la `final_answer` est valide (`True`) ou non (`False`).

Si une fonction renvoie `False`, l'agent journalise le message d'erreur et poursuit son exécution.
Ce mécanisme de validation permet :
- d'imposer des contraintes de format de sortie (par ex. s'assurer que les réponses sont numériques pour les problèmes de maths) ;
- d'implémenter des règles de validation spécifiques à un domaine ;
- de créer des agents plus robustes qui valident eux‑mêmes leurs sorties.

## Inspecter l'exécution d'un agent

Voici quelques attributs utiles pour inspecter ce qui s'est passé après une exécution :
- `agent.logs` stocke les journaux détaillés de l'agent. À chaque étape de l'exécution, tout est stocké dans un dictionnaire qui est ensuite ajouté à `agent.logs`.
- L'appel à `agent.write_memory_to_messages()` écrit la mémoire de l'agent sous forme de liste de messages de chat visibles par le modèle. Cette méthode parcourt chaque étape du log et ne stocke que ce qui l'intéresse sous forme de message : par exemple, elle enregistre le prompt système et la tâche dans des messages séparés, puis pour chaque étape, elle stocke la sortie du LLM dans un message, et la sortie d'appel d'outil dans un autre. Utilisez‑la si vous souhaitez une vue plus haut niveau de ce qu'il s'est passé – mais tous les logs ne seront pas forcément retranscrits par cette méthode.

## Outils

Un outil est une fonction atomique utilisée par un agent. Pour être utilisé par un LLM, il a également besoin de quelques attributs qui constituent son API et seront utilisés pour décrire au LLM comment appeler cet outil :
- Un nom ;
- Une description ;
- Des types et descriptions d'entrées ;
- Un type de sortie.

Vous pouvez par exemple consulter [`PythonInterpreterTool`] : il possède un nom, une description, des descriptions d'entrées, un type de sortie et une méthode `forward` qui réalise l'action.

Lorsque l'agent est initialisé, les attributs de l'outil sont utilisés pour générer une description d'outil qui est intégrée au prompt système de l'agent. Cela permet à l'agent de savoir quels outils il peut utiliser et pourquoi.

**Informations de schéma** : pour les outils qui définissent un `output_schema` (comme les outils MCP avec sortie structurée), le prompt système de `CodeAgent` inclut automatiquement le schéma JSON. Cela aide l'agent à comprendre la structure attendue des sorties d'outils et à y accéder correctement.

### Boîte à outils par défaut

Si vous installez `smolagents` avec l'extra "toolkit", il inclut une boîte à outils par défaut que vous pouvez ajouter à votre agent à l'initialisation avec l'argument `add_base_tools=True` :

- **Recherche web DuckDuckGo*** : effectue une recherche web via le navigateur DuckDuckGo.
- **Interpréteur de code Python** : exécute votre code Python généré par le LLM dans un environnement sécurisé. Cet outil ne sera ajouté à [`ToolCallingAgent`] que si vous l'initialisez avec `add_base_tools=True`, puisque les agents basés sur le code peuvent déjà exécuter du code Python nativement.
- **Transcripteur** : un pipeline de reconnaissance vocale basé sur Whisper-Turbo qui transcrit un audio en texte.

Vous pouvez utiliser un outil manuellement en l'appelant avec ses arguments.

```python
# !pip install 'smolagents[toolkit]'
from smolagents import WebSearchTool

search_tool = WebSearchTool()
print(search_tool("Who's the current president of Russia?"))
```

### Créer un nouvel outil

Vous pouvez créer vos propres outils pour les cas d'usage qui ne sont pas couverts par les outils par défaut de Hugging Face.
Par exemple, créons un outil qui renvoie le modèle le plus téléchargé pour une tâche donnée sur le Hub.

Vous commencez avec le code ci‑dessous :

```python
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(most_downloaded_model.id)
```

Ce code peut être rapidement converti en outil en l'encapsulant dans une fonction et en lui ajoutant le décorateur `tool`.
Ce n'est pas la seule manière de construire un outil : vous pouvez aussi le définir directement comme une sous‑classe de [`Tool`], ce qui offre plus de flexibilité, par exemple pour initialiser des attributs de classe lourds.

Voyons comment cela fonctionne pour les deux options :

<hfoptions id="build-a-tool">
<hfoption id="Decorate a function with @tool">

```py
from smolagents import tool

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id
```

La fonction a besoin de :
- Un nom clair. Le nom doit être suffisamment descriptif de ce que fait l'outil pour aider le LLM qui alimente l'agent. Puisque cet outil renvoie le modèle ayant le plus de téléchargements pour une tâche donnée, appelons‑le `model_download_tool`.
- Des annotations de type pour les entrées et la sortie.
- Une description qui inclut une section `Args:` où chaque argument est décrit (sans indication de type cette fois, celui‑ci sera récupéré à partir de l'annotation de type). Comme pour le nom de l'outil, cette description est un manuel d'utilisation pour le LLM qui alimente votre agent, donc ne la négligez pas.

Tous ces éléments seront automatiquement intégrés dans le prompt système de l'agent lors de l'initialisation : faites en sorte qu'ils soient aussi clairs que possible !

> [!TIP]
> Ce format de définition est le même que pour les schémas d'outils utilisés dans `apply_chat_template`, à la différence près du décorateur `tool` ajouté : vous pouvez en lire plus sur notre API d'utilisation d'outils [ici](https://huggingface.co/blog/unified-tool-use#passing-tools-to-a-chat-template).


Vous pouvez ensuite initialiser directement votre agent :
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
agent.run(
    "Peux-tu me donner le nom du modèle qui a le plus de téléchargements pour la tâche 'text-to-video' sur le Hub Hugging Face ?"
)
```
</hfoption>
<hfoption id="Subclass Tool">

```py
from smolagents import Tool

class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return most_downloaded_model.id
```

La sous‑classe a besoin des attributs suivants :
- Un `name` clair. Le nom doit être suffisamment descriptif de ce que fait l'outil pour aider le LLM qui alimente l'agent. Puisque cet outil renvoie le modèle ayant le plus de téléchargements pour une tâche donnée, appelons‑le `model_download_tool`.
- Une `description`. Comme pour `name`, cette description est un manuel d'utilisation pour le LLM qui alimente votre agent, donc ne la négligez pas.
- Des types d'entrées et leurs descriptions.
- Un type de sortie.
Tous ces attributs seront automatiquement intégrés dans le prompt système de l'agent lors de l'initialisation : faites en sorte qu'ils soient aussi clairs que possible !


Vous pouvez ensuite initialiser directement votre agent :
```py
from smolagents import CodeAgent, InferenceClientModel
agent = CodeAgent(tools=[ModelDownloadTool()], model=InferenceClientModel())
agent.run(
    "Peux-tu me donner le nom du modèle qui a le plus de téléchargements pour la tâche 'text-to-video' sur le Hub Hugging Face ?"
)
```
</hfoption>
</hfoptions>

Vous obtenez les journaux suivants :
```text
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                          │
│ Peux-tu me donner le nom du modèle qui a le plus de téléchargements pour la tâche        │
│ 'text-to-video' sur le Hub Hugging Face ?                                                │
│                                                                                          │
╰─ InferenceClientModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 model_name = model_download_tool(task="text-to-video")                               │
│   2 print(model_name)                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
ByteDance/AnimateDiff-Lightning

Out: None
[Step 0: Duration 0.27 seconds| Input tokens: 2,069 | Output tokens: 60]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer("ByteDance/AnimateDiff-Lightning")                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: ByteDance/AnimateDiff-Lightning
[Step 1: Duration 0.10 seconds| Input tokens: 4,288 | Output tokens: 148]
Out[20]: 'ByteDance/AnimateDiff-Lightning'
```

> [!TIP]
> Pour en savoir plus sur les outils, consultez le [tutoriel dédié](./tutorials/tools#what-is-a-tool-and-how-to-build-one).

## Multi‑agents

Les systèmes multi‑agents ont été introduits avec le framework [Autogen](https://huggingface.co/papers/2308.08155) de Microsoft.

Dans ce type de framework, plusieurs agents travaillent ensemble pour résoudre votre tâche, au lieu d'un seul.
Empiriquement, cela donne de meilleures performances sur la plupart des benchmarks. La raison est conceptuellement simple : pour de nombreuses tâches, plutôt que d'utiliser un système qui fait tout, il est préférable de spécialiser des unités sur des sous‑tâches. Ici, avoir des agents avec des jeux d'outils et des mémoires séparés permet d'obtenir une spécialisation efficace. Par exemple, pourquoi remplir la mémoire de l'agent générant le code avec tout le contenu des pages web visitées par l'agent de recherche web ? Il est préférable de les garder séparés.

Vous pouvez facilement construire des systèmes hiérarchiques multi‑agents avec `smolagents`.

Pour cela, assurez‑vous simplement que votre agent possède les attributs `name` et `description`, qui seront ensuite intégrés dans le prompt système de l'agent manager pour lui indiquer comment appeler cet agent géré, comme nous le faisons aussi pour les outils.
Vous pouvez ensuite passer cet agent géré dans le paramètre `managed_agents` lors de l'initialisation de l'agent manager.

Voici un exemple de création d'un agent qui pilote un agent de recherche web spécifique en utilisant notre [`WebSearchTool`] natif :

```py
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> Pour un exemple détaillé d'implémentation multi‑agents efficace, consultez [comment nous avons propulsé notre système multi‑agents en tête du classement GAIA](https://huggingface.co/blog/beating-gaia).


## Discuter avec votre agent et visualiser ses pensées dans une interface Gradio

Vous pouvez utiliser `GradioUI` pour soumettre des tâches de manière interactive à votre agent et observer son raisonnement et son exécution. Voici un exemple :

```py
from smolagents import (
    load_tool,
    CodeAgent,
    InferenceClientModel,
    GradioUI
)

# Importer un outil depuis le Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = InferenceClientModel(model_id=model_id)

# Initialiser l'agent avec l'outil de génération d'images
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
```

En coulisses, lorsque l'utilisateur saisit une nouvelle requête, l'agent est lancé avec `agent.run(user_request, reset=False)`.
Le drapeau `reset=False` signifie que la mémoire de l'agent n'est pas vidée avant de lancer cette nouvelle tâche, ce qui permet à la conversation de se poursuivre.

Vous pouvez aussi utiliser cet argument `reset=False` pour garder la conversation ouverte dans toute autre application agentique.

Dans les interfaces Gradio, si vous souhaitez permettre aux utilisateurs d'interrompre un agent en cours d'exécution, vous pouvez le faire avec un bouton qui déclenche la méthode `agent.interrupt()`.
Cela arrêtera l'agent à la fin de son étape actuelle, puis lèvera une erreur.

## Prochaines étapes

Enfin, lorsque vous avez configuré votre agent comme vous le souhaitez, vous pouvez le partager sur le Hub !

```py
agent.push_to_hub("m-ric/my_agent")
```

De même, pour charger un agent qui a été poussé sur le Hub, si vous faites confiance au code de ses outils, utilisez :
```py
agent.from_hub("m-ric/my_agent", trust_remote_code=True)
```

Pour un usage plus approfondi, vous voudrez ensuite consulter nos tutoriels :
- [l'explication de la manière dont fonctionnent nos code agents](./tutorials/secure_code_execution)
- [ce guide sur la façon de construire de bons agents](./tutorials/building_good_agents)
- [le guide détaillé sur l'utilisation des outils](./tutorials/building_good_agents).
