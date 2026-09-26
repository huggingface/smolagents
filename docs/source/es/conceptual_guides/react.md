# ¿Cómo funcionan los agentes multipaso?

El framework ReAct ([Yao et al., 2022](https://huggingface.co/papers/2210.03629)) es actualmente el principal enfoque para construir agentes.

El nombre se basa en la unión de dos palabras, "Reason" (razonar) y "Act" (actuar). En efecto, los agentes que siguen esta arquitectura resuelven su tarea en tantos pasos como haga falta, y cada paso consiste en un paso de razonamiento seguido de un paso de acción, en el que formula las llamadas a herramientas que lo acercarán a resolver la tarea en cuestión.

Todos los agentes de `smolagents` se basan en una única clase `MultiStepAgent`, que es una abstracción del framework ReAct.

A nivel básico, esta clase ejecuta acciones en un ciclo con los siguientes pasos, donde las variables y el conocimiento existentes se incorporan a los registros del agente como se muestra a continuación:

Inicialización: el prompt del sistema se almacena en un `SystemPromptStep`, y la consulta del usuario se registra en un `TaskStep`.

Bucle while (bucle ReAct):

- Usa `agent.write_memory_to_messages()` para escribir los registros del agente en una lista de [mensajes de chat](https://huggingface.co/docs/transformers/en/chat_templating) legibles por el LLM.
- Envía estos mensajes a un objeto `Model` para obtener su respuesta. Analiza la respuesta para obtener la acción (un blob JSON para `ToolCallingAgent`, un fragmento de código para `CodeAgent`).
- Ejecuta la acción y registra el resultado en la memoria (un `ActionStep`).
- Al final de cada paso, ejecutamos todas las funciones de callback definidas en `agent.step_callbacks`.

Opcionalmente, cuando la planificación está activada, un plan puede revisarse periódicamente y almacenarse en un `PlanningStep`. Esto incluye incorporar a la memoria hechos sobre la tarea en cuestión.

Para un `CodeAgent`, se ve como en la figura siguiente.

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/codeagent_docs.png"
    />
</div>

Aquí tienes un resumen en vídeo de cómo funciona:

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

Implementamos dos versiones de agentes:
- [`CodeAgent`] genera sus llamadas a herramientas como fragmentos de código Python.
- [`ToolCallingAgent`] escribe sus llamadas a herramientas como JSON, como es habitual en muchos frameworks. Según tus necesidades, se puede usar cualquiera de los dos enfoques. Por ejemplo, la navegación web a menudo requiere esperar después de cada interacción con la página, así que las llamadas a herramientas en JSON pueden encajar bien.

> [!TIP]
> Lee la entrada de blog [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) para saber más sobre los agentes multipaso.
