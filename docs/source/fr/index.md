# `smolagents`

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png" style="max-width:700px"/>
</div>

## Qu'est-ce que Smolagents ?

`smolagents` est une bibliothèque Python open source conçue pour rendre la création et l'exécution d'agents extrêmement simple, avec seulement quelques lignes de code.

Voici les principales fonctionnalités de `smolagents` :

✨ **Simplicité** : La logique des agents tient en environ un millier de lignes de code. Nous avons gardé les abstractions au minimum au‑dessus du code brut !

🧑‍💻 **Support de première classe pour les Code Agents** : [`CodeAgent`](reference/agents#smolagents.CodeAgent) écrit ses actions en code (par opposition aux « agents utilisés pour écrire du code ») pour invoquer des outils ou effectuer des calculs, ce qui permet une composition naturelle (imbriquation de fonctions, boucles, conditions). Pour rendre cela sûr, nous prenons en charge [l'exécution en environnement sandboxé](tutorials/secure_code_execution) via [Modal](https://modal.com/), [Blaxel](https://blaxel.ai), [E2B](https://e2b.dev/) ou Docker.

📡 **Support classique d'agents appelant des outils** : En plus des CodeAgents, [`ToolCallingAgent`](reference/agents#smolagents.ToolCallingAgent) prend en charge l'appel d'outils basé sur du JSON/texte pour les scénarios où ce paradigme est préféré.

🤗 **Intégration au Hub** : Partagez et chargez facilement des agents et des outils vers/depuis le Hub sous forme de Spaces Gradio.

🌐 **Indépendant du modèle** : Intégrez facilement n'importe quel grand modèle de langage (LLM), qu'il soit hébergé sur le Hub via les [Inference providers](https://huggingface.co/docs/inference-providers/index), accessible via des API comme OpenAI, Anthropic ou bien d'autres via l'intégration LiteLLM, ou exécuté localement avec Transformers ou Ollama. Alimenter un agent avec le LLM de votre choix est simple et flexible.

👁️ **Indépendant de la modalité** : Au‑delà du texte, les agents peuvent gérer des entrées de vision, de vidéo et d'audio, élargissant le champ des applications possibles. Consultez [ce tutoriel](examples/web_browser) pour la vision.

🛠️ **Indépendant des outils** : Vous pouvez utiliser des outils provenant de n'importe quel [serveur MCP](reference/tools#smolagents.ToolCollection.from_mcp), de [LangChain](reference/tools#smolagents.Tool.from_langchain), et même utiliser un [Space du Hub](reference/tools#smolagents.Tool.from_space) comme outil.

💻 **Outils en ligne de commande (CLI)** : Livré avec des utilitaires en ligne de commande (CLI: smolagent, webagent) pour exécuter rapidement des agents sans écrire de code passe‑partout.

## Démarrage rapide

[[open-in-colab]]

Commencez avec smolagents en quelques minutes ! Ce guide vous montre comment créer et exécuter votre premier agent.

### Installation

Installez smolagents avec pip :

```bash
pip install 'smolagents[toolkit]'  # Includes default tools like web search
```

### Créez votre premier agent

Voici un exemple minimal pour créer et exécuter un agent :

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialise un modèle (en utilisant l'API Hugging Face Inference)
model = InferenceClientModel()  # Utilise un modèle par défaut

# Crée un agent sans outils
agent = CodeAgent(tools=[], model=model)

# Exécute l'agent avec une tâche
result = agent.run("Calcule la somme des nombres de 1 à 10")
print(result)
```

C'est tout ! Votre agent va utiliser du code Python pour résoudre la tâche et renvoyer le résultat.

### Ajouter des outils

Rendons notre agent plus puissant en lui ajoutant quelques outils :

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
  tools=[DuckDuckGoSearchTool()],
  model=model,
)

# L'agent peut maintenant rechercher sur le web !
result = agent.run("Quel temps fait‑il actuellement à Paris ?")
print(result)
```

### Utiliser différents modèles

Vous pouvez utiliser différents modèles avec votre agent :

```python
# Utiliser un modèle spécifique depuis Hugging Face
model = InferenceClientModel(model_id="meta-llama/Llama-2-70b-chat-hf")

# Utiliser OpenAI/Anthropic (nécessite 'smolagents[litellm]')
from smolagents import LiteLLMModel
model = LiteLLMModel(model_id="gpt-4")

# Utiliser des modèles locaux (nécessite 'smolagents[transformers]')
from smolagents import TransformersModel
model = TransformersModel(model_id="meta-llama/Llama-2-7b-chat-hf")
```

## Prochaines étapes

- Découvrez comment configurer smolagents avec différents modèles et outils dans le [guide d'installation](installation)
- Consultez la [visite guidée](guided_tour) pour des fonctionnalités plus avancées
- Apprenez à [créer des outils personnalisés](tutorials/tools)
- Explorez [l'exécution de code sécurisée](tutorials/secure_code_execution)
- Voyez comment créer des [systèmes multi‑agents](tutorials/building_good_agents)

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Visite guidée</div>
      <p class="text-gray-700">Apprenez les bases et familiarisez‑vous avec l'utilisation des agents. Commencez ici si vous utilisez des agents pour la première fois&nbsp;!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guides pratiques</div>
      <p class="text-gray-700">Guides pratiques pour vous aider à atteindre un objectif précis&nbsp;: créer un agent qui génère et teste des requêtes SQL&nbsp;!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guides conceptuels</div>
      <p class="text-gray-700">Explications de haut niveau pour mieux comprendre les sujets importants.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutoriels</div>
      <p class="text-gray-700">Tutoriels transverses qui couvrent les aspects importants de la création d'agents.</p>
    </a>
  </div>
</div>
