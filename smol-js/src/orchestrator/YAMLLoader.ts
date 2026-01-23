/**
 * YAMLLoader - Loads and parses YAML workflow definitions into runnable agents
 */

import * as fs from 'fs';
import * as path from 'path';
import YAML from 'yaml';
import type { YAMLWorkflowDefinition, YAMLAgentDefinition, YAMLModelDefinition } from '../types.js';
import { Agent } from '../agents/Agent.js';
import { CodeAgent } from '../agents/CodeAgent.js';
import { ToolUseAgent } from '../agents/ToolUseAgent.js';
import { Tool } from '../tools/Tool.js';
import { AgentTool } from '../tools/AgentTool.js';
import { OpenAIModel } from '../models/OpenAIModel.js';
import { ReadFileTool } from '../tools/ReadFileTool.js';
import { WriteFileTool } from '../tools/WriteFileTool.js';
import { CurlTool } from '../tools/CurlTool.js';
import { ExaSearchTool } from '../tools/ExaSearchTool.js';
import { ExaGetContentsTool } from '../tools/ExaGetContentsTool.js';
import { ExaResearchTool } from '../tools/ExaResearchTool.js';
import { FinalAnswerTool } from '../tools/defaultTools.js';

// Registry of built-in tool types
const TOOL_REGISTRY: Record<string, new (config?: Record<string, unknown>) => Tool> = {
  read_file: ReadFileTool as unknown as new (config?: Record<string, unknown>) => Tool,
  write_file: WriteFileTool as unknown as new (config?: Record<string, unknown>) => Tool,
  curl: CurlTool as unknown as new (config?: Record<string, unknown>) => Tool,
  exa_search: ExaSearchTool as unknown as new (config?: Record<string, unknown>) => Tool,
  exa_get_contents: ExaGetContentsTool as unknown as new (config?: Record<string, unknown>) => Tool,
  exa_research: ExaResearchTool as unknown as new (config?: Record<string, unknown>) => Tool,
  final_answer: FinalAnswerTool as unknown as new (config?: Record<string, unknown>) => Tool,
};

export interface LoadedWorkflow {
  name: string;
  description?: string;
  entrypointAgent: Agent;
  agents: Map<string, Agent>;
  tools: Map<string, Tool>;
}

export class YAMLLoader {
  private customTools: Map<string, new (config?: Record<string, unknown>) => Tool> = new Map();

  /**
   * Register a custom tool type for use in YAML definitions.
   */
  registerToolType(typeName: string, toolClass: new (config?: Record<string, unknown>) => Tool): void {
    this.customTools.set(typeName, toolClass);
  }

  /**
   * Load a workflow from a YAML file path.
   */
  loadFromFile(filePath: string): LoadedWorkflow {
    const absolutePath = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);

    if (!fs.existsSync(absolutePath)) {
      throw new Error(`Workflow file not found: ${absolutePath}`);
    }

    const content = fs.readFileSync(absolutePath, 'utf-8');
    return this.loadFromString(content);
  }

  /**
   * Load a workflow from a YAML string.
   */
  loadFromString(yamlContent: string): LoadedWorkflow {
    const definition = YAML.parse(yamlContent) as YAMLWorkflowDefinition;
    return this.buildWorkflow(definition);
  }

  /**
   * Build a runnable workflow from a parsed definition.
   */
  private buildWorkflow(definition: YAMLWorkflowDefinition): LoadedWorkflow {
    if (!definition.name) {
      throw new Error('Workflow must have a name');
    }
    if (!definition.entrypoint) {
      throw new Error('Workflow must have an entrypoint agent');
    }
    if (!definition.agents) {
      throw new Error('Workflow must define at least one agent');
    }

    // Build tools
    const tools = new Map<string, Tool>();
    if (definition.tools) {
      for (const [name, toolDef] of Object.entries(definition.tools)) {
        const tool = this.buildTool(name, toolDef.type, toolDef.config);
        tools.set(name, tool);
      }
    }

    // Build agents (handling dependencies between agents)
    const agents = new Map<string, Agent>();
    const agentDefs = definition.agents;

    // Topological sort: build agents that don't depend on other agents first
    const resolved = new Set<string>();
    const maxIterations = Object.keys(agentDefs).length * 2;
    let iterations = 0;

    while (resolved.size < Object.keys(agentDefs).length && iterations < maxIterations) {
      iterations++;
      for (const [agentName, agentDef] of Object.entries(agentDefs)) {
        if (resolved.has(agentName)) continue;

        // Check if all agent dependencies are resolved
        const agentDeps = agentDef.agents ?? [];
        const allDepsResolved = agentDeps.every(dep => resolved.has(dep));

        if (allDepsResolved) {
          const agent = this.buildAgent(
            agentName,
            agentDef,
            definition.model,
            tools,
            agents,
            definition.globalMaxContextLength
          );
          agents.set(agentName, agent);
          resolved.add(agentName);
        }
      }
    }

    if (resolved.size < Object.keys(agentDefs).length) {
      const unresolved = Object.keys(agentDefs).filter(n => !resolved.has(n));
      throw new Error(`Circular or unresolvable agent dependencies: ${unresolved.join(', ')}`);
    }

    const entrypointAgent = agents.get(definition.entrypoint);
    if (!entrypointAgent) {
      throw new Error(`Entrypoint agent "${definition.entrypoint}" not found in agents`);
    }

    return {
      name: definition.name,
      description: definition.description,
      entrypointAgent,
      agents,
      tools,
    };
  }

  /**
   * Build a tool instance from a type name and config.
   */
  private buildTool(name: string, type: string, config?: Record<string, unknown>): Tool {
    const ToolClass = TOOL_REGISTRY[type] ?? this.customTools.get(type);

    if (!ToolClass) {
      throw new Error(`Unknown tool type: ${type}. Available types: ${[...Object.keys(TOOL_REGISTRY), ...this.customTools.keys()].join(', ')}`);
    }

    const tool = new ToolClass(config);
    // Override name if different from type
    if (name !== type && name !== tool.name) {
      Object.defineProperty(tool, 'name', { value: name, writable: false });
    }
    return tool;
  }

  /**
   * Build an agent instance from a YAML definition.
   */
  private buildAgent(
    name: string,
    definition: YAMLAgentDefinition,
    globalModel?: YAMLModelDefinition,
    availableTools?: Map<string, Tool>,
    resolvedAgents?: Map<string, Agent>,
    globalMaxContextLength?: number
  ): Agent {
    // Build model
    const modelConfig = definition.model ?? globalModel;
    const model = new OpenAIModel({
      modelId: modelConfig?.modelId,
      apiKey: modelConfig?.apiKey,
      baseUrl: modelConfig?.baseUrl,
      maxTokens: definition.maxTokens ?? modelConfig?.maxTokens,
      temperature: definition.temperature ?? modelConfig?.temperature,
      timeout: modelConfig?.timeout,
    });

    // Collect tools
    const agentTools: Tool[] = [];

    // Add referenced tools
    if (definition.tools && availableTools) {
      for (const toolName of definition.tools) {
        const tool = availableTools.get(toolName);
        if (tool) {
          agentTools.push(tool);
        } else {
          // Try to create from registry directly
          const ToolClass = TOOL_REGISTRY[toolName] ?? this.customTools.get(toolName);
          if (ToolClass) {
            agentTools.push(new ToolClass());
          } else {
            throw new Error(`Tool "${toolName}" not found for agent "${name}"`);
          }
        }
      }
    }

    // Add sub-agents as tools
    if (definition.agents && resolvedAgents) {
      for (const subAgentName of definition.agents) {
        const subAgent = resolvedAgents.get(subAgentName);
        if (!subAgent) {
          throw new Error(`Sub-agent "${subAgentName}" not found for agent "${name}"`);
        }
        agentTools.push(new AgentTool({
          agent: subAgent,
          name: subAgentName,
          description: definition.description
            ? `Sub-agent: ${subAgentName}`
            : `Delegate tasks to the ${subAgentName} agent`,
        }));
      }
    }

    const maxContextLength = definition.maxContextLength ?? globalMaxContextLength;

    // Build the agent based on type
    if (definition.type === 'CodeAgent') {
      return new CodeAgent({
        model,
        tools: agentTools,
        maxSteps: definition.maxSteps,
        customInstructions: definition.customInstructions,
        persistent: definition.persistent,
        maxContextLength,
        memoryStrategy: definition.memoryStrategy,
        maxTokens: definition.maxTokens,
        temperature: definition.temperature,
        name,
      });
    } else {
      // Default: ToolUseAgent
      return new ToolUseAgent({
        model,
        tools: agentTools,
        maxSteps: definition.maxSteps,
        customInstructions: definition.customInstructions,
        persistent: definition.persistent,
        maxContextLength,
        memoryStrategy: definition.memoryStrategy,
        maxTokens: definition.maxTokens,
        temperature: definition.temperature,
        name,
      });
    }
  }
}
