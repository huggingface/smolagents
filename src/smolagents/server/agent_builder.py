import ast
import importlib
import math
from typing import Any, List, Dict, Callable

import yaml
from smolagents import CodeAgent, Tool, LogLevel, evaluate_ast, AUTHORIZED_TYPES
from smolagents.agents import MultiStepAgent
from smolagents.utils import UserInputError,Display

BASE_PYTHON_TOOLS = {
    "print": print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}


class InputTool(Tool):
    name = "user_input"
    description = "requests user input for specific questions, but stops subsequent code execution.."
    inputs = {"question": {"type": "string", "description": "question for user"}}
    output_type = "string"

    def forward(self, question):
        raise UserInputError(question)


class Component(Display):
    def __init__(self, value):
        self.value = value

    #def __str__(self):
        # return '<final_answer>The result store in memory.</final_answer>'
        #return "The print() has returned a result stored in memory."

    def display(self,expr):
        res = "There has a result stored in memory"
        if hasattr(expr,'id'):
            res += f" that we can use '{expr.id}' variable whose type is object in next code snippet to complete the task."
        return res


    def to_dict(self):
        return self.value


class APITool(Tool):
    def __init__(
            self,
            name: str,
            description: str,
            inputs: Dict[str, Dict[str, str]],
            output_type: str,
            function: Callable,
            expression=None
    ):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = output_type
        self.forward = function
        self.expression = expression
        self.is_initialized = True

        self.state = {}
        self.static_tools = {'Component': Component, **BASE_PYTHON_TOOLS}
        self.custom_tools = {}
        self.authorized_imports = ["*"]

    def validate_expression(self):
        if self.expression is None:
            return

        call: ast.Expr = None
        for node in reversed(self.expression.body):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node
                break
        if call is not None:
            def forward(**kwargs):
                state = {}
                state.update(kwargs)
                static_tools = {'Component': Component, **BASE_PYTHON_TOOLS}
                custom_tools = {}
                authorized_imports = ["*"]
                result = None
                for node in self.expression.body:
                    result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                return result

            self.forward = forward
        else:
            for node in self.expression.body:
                evaluate_ast(node, self.state, self.static_tools, self.custom_tools, self.authorized_imports)
            func_def = self.expression.body[-1]

            if isinstance(func_def, ast.FunctionDef):
                self.forward = self.custom_tools[func_def.name]
            else:
                raise ValueError("You must last function must specify as a main endpoint")

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        self.validate_expression()
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value, should be one of {AUTHORIZED_TYPES}."
                )

        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES


class AgentBuilder:
    def __init__(self, model_list=None):
        self.agents = {}
        self.tools = {}
        self.model_list = model_list

    def build(self):
        agents = {}
        for _ in range(len(self.agents)):
            parse_deps(agents, self)

        for _, agent_data in agents.items():
            if isinstance(agent_data, dict):
                for name in agent_data['managed_agents']:
                    if isinstance(name, str):
                        raise ValueError(f"You specify a managed_agent named {name} is not find.")
        return agents


def parse_tool(tool_config: Dict[str, Any]) -> Tool:
    tool_data = {'name': tool_config['metadata'].get('name', 'default_tool'),
                 'description': tool_config['metadata'].get('description', ''),
                 'input': {}}

    if 'spec' in tool_config and 'input' in tool_config['spec']:
        for input_item in tool_config['spec']['input']:
            tool_data['input'].update(input_item)

    tool_data['output'] = tool_config['spec'].get('output', {})
    tool_data['output_type'] = tool_data['output'].get('type')
    tool_data['output_destination'] = tool_data['output'].get('destination')
    tool_data['behavior'] = tool_config['spec'].get('behavior', {})
    tool_data['script'] = tool_data['behavior'].get('script')
    script = tool_data['script']
    if script is not None:
        expression = ast.parse(script)

    tool_data['forward'] = None
    tool = APITool(tool_data['name'], tool_data['description'], tool_data['input'], tool_data['output_type'],
                   tool_data['forward'], expression)
    return tool


def parse_agent_tool(config: Dict[str, Any], builder: AgentBuilder, tools: List) -> List:
    if 'tools' in config['spec']:
        for tool_config in config['spec']['tools']:
            if isinstance(tool_config, str):
                name = tool_config
                tool = builder.tools.get(name)
                if tool is None:
                    tool = name
            elif tool_config.get('spec') is None:
                name = tool_config['metadata'].get('name')
                tool = builder.tools.get(name)
                if tool is None:
                    tool = name
            else:
                tool = parse_tool(tool_config)
            tools.append(tool)
    return tools


def parse_kind_tool(config: Dict[str, Any], tools: List):
    if config['spec'].get('tools'):
        for tool_config in config['spec']['tools']:
            tool = parse_tool(tool_config)
            tools.append(tool)
    return tools


def parse_managed_agent_as_tool(config: Dict[str, Any], builder: AgentBuilder, managed_agents: List) -> List:
    for name in config['spec'].get('managed_agents', []):
        if isinstance(name, str):
            agent = builder.agents.get(name)
            if agent is None:
                agent = name  # lazy get agent
            managed_agents.append(agent)
    return managed_agents


def parse_agent(config: Dict[str, Any], builder: AgentBuilder) -> Dict[str, Any]:
    agent_data = {}
    model = config['spec'].get('model')
    if model is not None:
        model = builder.model_list.get_model(model)
    if model is None or not hasattr(model, '__call__'):
        raise ValueError("You must specify a model.")

    # behavior = config['spec'].get('behavior', {})  # model config
    # agent_data.update(behavior)

    agent_data['name'] = config['metadata'].get('name', 'default_agent')
    agent_data['description'] = config['metadata'].get('description', '')
    agent_data['model'] = model

    # 处理 prompt_templates
    agent_data['prompt_templates'] = {}
    if config['spec'].get('prompt_templates'):
        for key, value in config['spec']['prompt_templates'].items():
            if isinstance(value, dict):  # 处理嵌套的 prompt_templates
                agent_data['prompt_templates'][key] = value
            else:
                agent_data['prompt_templates'][key] = value

    # 处理其它属性
    agent_data['grammar'] = config['spec'].get('grammar', {})
    agent_data['planning_interval'] = config['spec'].get('planning_interval')
    agent_data['max_steps'] = config['spec'].get('max_steps', None)
    agent_data['add_base_tools'] = config['spec'].get('add_base_tools', False)
    agent_data['verbosity_level'] = config['spec'].get('verbosity_level', "INFO")  # 获取日志级别字符串
    agent_data['verbosity_level'] = getattr(LogLevel, agent_data['verbosity_level'].upper(),
                                            LogLevel.INFO)  # 转换为 LogLevel 枚举
    agent_data['provide_run_summary'] = config['spec'].get('provide_run_summary', False)

    agent_data['tools'] = parse_agent_tool(config, builder, [InputTool()])
    agent_data['managed_agents'] = parse_managed_agent_as_tool(config, builder, [])
    return agent_data


def parse_code_agent(config: Dict[str, Any], builder: AgentBuilder) -> Dict[str, Any]:
    agent_data = parse_agent(config, builder)
    agent_data['additional_authorized_imports'] = config['spec'].get('additional_authorized_imports', [])
    agent_data['executor_type'] = config['spec'].get('executor_type', "local")
    agent_data['max_print_outputs_length'] = config['spec'].get('max_print_outputs_length')
    agent_data['kind'] = lambda **kwargs: CodeAgent(**kwargs)
    return agent_data


def parse_deps(agents: Dict[str, Any], builder: AgentBuilder):
    for key, agent_data in builder.agents.items():
        agent = agents.get(key)
        if agent is None:
            agent = agent_data.copy()
            agent['managed_agents'] = agent_data['managed_agents'].copy()
            agents[key] = agent
        if isinstance(agent, Dict):
            managed_agents = agent['managed_agents']
            lazy_count = 0
            for i in range(len(managed_agents)):
                if isinstance(managed_agents[i], str):
                    managed_agent = agents.get(managed_agents[i])
                    if managed_agent is None or isinstance(managed_agent, Dict):
                        lazy_count += 1
                    else:
                        managed_agents[i] = managed_agent
            if lazy_count == 0:
                kind = agent['kind']
                del agent['kind']
                agents[key] = kind(**agent)


def parse_yaml(config_list: List[Dict[str, Any]], model_list) -> AgentBuilder:
    builder = AgentBuilder(model_list)
    # parse agents
    for config in config_list:
        kind = config['kind']
        if kind == 'CodeAgent':
            agent = parse_code_agent(config, builder)
            builder.agents[agent['name']] = agent
        elif kind == 'Tool':
            for tool in parse_kind_tool(config, []):
                builder.tools[tool.name] = tool
        else:
            raise ValueError("You must specify a kind that in [CodeAgent,Tool].")
    # validate tools
    for key, agent_data in builder.agents.items():
        tools = agent_data['tools']
        for i in range(len(tools)):
            if isinstance(tools[i], str):
                tool = builder.tools.get(tools[i])
                if tool is None:
                    raise ValueError(f"You specify a tool that named {tool} is not find or cycle dependency.")
                else:
                    tools[i] = tool

    return builder


def agent_builder(chat_id: str, agent_id: str, model_list) -> MultiStepAgent:
    # 根据agent_id找到对于的YAML配置,解析YAML为dataclass，并根据dataclass生成Agent
    config = yaml.safe_load_all(
        importlib.resources.files("smolagents.server").joinpath(f"{agent_id}.yaml").read_text(encoding='utf-8')
    )
    builder = parse_yaml(config, model_list)
    agents = builder.build()
    for _, agent in agents.items():
        return agent
    return None
