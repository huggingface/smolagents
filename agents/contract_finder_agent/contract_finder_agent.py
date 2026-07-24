import os
import re
import math
import json
import copy
import html
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from json_repair import repair_json
from psycopg2 import sql
import logging
import traceback
from collections import defaultdict
import numpy as np
from collections import Counter

from pydantic import Field
from typing import Optional, Annotated, List, Dict, Any, Set, Tuple

from llm_studio_tools.sql_executor import SQLExecutorTool
from llm_studio_tools.llm_configuration_tool import LLMConfigurationTool
from llm_studio_tools.semantic_similarity_tool import SemanticSimilarityTool
from llm_studio_tools.citation_tool import CitationTool

from llm_studio_agents.AgentBase import AgentsTraceBase
from llm_studio_agents.utils.utils import accumulator, process_config, citation_generator
from llm_studio_agents.utils.utils_agent_pubsub import send_streaming_response_to_pubsub
from llm_studio_agents.utils.utils_traceability import AITrace

import Config

try:
    from agents.RecommendationAgentBase import RecommendationAgentBase
except ImportError:
    # Fallback if RecommendationAgentBase doesn't exist yet
    from llm_studio_agents.AgentBase import AgentBase as RecommendationAgentBase

from agents.contract_finder_agent.contract_finder_agent_setup import ContractFinderAgentSetup
from agents.contract_finder_agent.contract_finder_agent_trace import ContractFinderAgentTrace
from agents.contract_finder_agent.contract_helper import FilterType, FILTER_TYPE_TO_TEXT_MAP, FILTER_TABLE_MAPPING
from agents.contract_finder_agent.contract_finder_utilities import (
    safe_json_load, remove_extras_from_retrieved_documents,
    handle_invalid_citation_sources, add_previously_displayed_documents_to_filtered_data
)


class ContractFinderAgent(RecommendationAgentBase):
    """
    Agent designed to retrieve contracts based on filters and a user query.
    """
    CONFIG_CLASS = ContractFinderAgentSetup
    
    def setup(self, concierge_id: Optional[str] = None, agent_id: Optional[str] = None,
              config: Optional[dict] = None, data: Optional[dict] = None) -> None:
        """
        Set up the ContractFinderAgent with the provided configuration.
        
        Args:
            concierge_id: Optional ID of the concierge
            agent_id: Optional ID of the agent
            config: Optional configuration dictionary
            data: Optional data dictionary
            
        Returns:
            None (initialization method)
            
        Raises:
            Exception: On configuration error, database connection failure, or tool initialization failure
        """
        try:
            logging.info("[ContractFinderAgent.setup] Starting agent setup")
            
            # Call parent setup to process configuration
            super().setup(config=config)
            
            # Store data and metadata
            self.data = data
            self.metadata = {}
            self.metadata_filters = {}
            self.input_user_question = data.get("question", None) if data else None
            
            if self.data:
                self.data["hide_headers"] = not self.config.show_headers

            logging.info("[ContractFinderAgent.setup] Initializing tools from configuration")
            
            # Setup SQL executor tool for database queries
            sql_executor_config = config["tools"][SQLExecutorTool.__name__]
            sql_executor_config = process_config(config=sql_executor_config, sub_level="integrations")
            self.sql_executor_tool = SQLExecutorTool(config=sql_executor_config, data=data)
            self.sql_executor_tool.config.show_results_in_end_header_message = True
            logging.info("[ContractFinderAgent.setup] SQLExecutorTool initialized")

            # Setup LLM configuration tool for answer generation
            llm_config_tool_config = config["tools"][LLMConfigurationTool.__name__]
            llm_config_tool_config = process_config(config=llm_config_tool_config, sub_level="integrations")
            self.llm_config_tool = LLMConfigurationTool(config=llm_config_tool_config, data=data)
            logging.info("[ContractFinderAgent.setup] LLMConfigurationTool initialized")
            
            # Create deep copies for potential retries
            self.llm_config_tool_copy = copy.deepcopy(llm_config_tool_config)
            self.data_copy = copy.deepcopy(data)

            # Setup semantic similarity tool for embedding generation
            semantic_similarity_config = config["tools"][SemanticSimilarityTool.__name__]
            semantic_similarity_config = process_config(config=semantic_similarity_config, sub_level="integrations")
            self.semantic_similarity_tool = SemanticSimilarityTool(config=semantic_similarity_config, data=data)
            logging.info("[ContractFinderAgent.setup] SemanticSimilarityTool initialized")

            # Setup citation tool
            citation_config = config["tools"][CitationTool.__name__]
            citation_config = process_config(citation_config, sub_level="integrations")
            self.citation_tool = CitationTool(config=citation_config, data=data)
            logging.info("[ContractFinderAgent.setup] CitationTool initialized")
            
            # Load SQL query templates
            self.queries = self._read_queries()
            
            # Store original SQL header messages and format columns
            self.original_sql_start_header_messages = self.sql_executor_tool.config.stream_start_message
            self.original_sql_end_header_messages = self.sql_executor_tool.config.stream_end_message
            self.original_format_columns = self.sql_executor_tool.config.format_result_columns
            
            logging.info("[ContractFinderAgent.setup] Agent setup completed successfully")
            
        except Exception as e:
            logging.error(f"[ContractFinderAgent.setup] Failed to initialize agent: {str(e)}")
            logging.error(f"[ContractFinderAgent.setup] Traceback: {traceback.format_exc()}")
            raise Exception(f"Contract Finder Agent setup failed: {str(e)}") from e
    
    def _read_queries(self) -> Dict[str, str]:
        """
        Read all SQL query templates from the filesystem.
        
        Returns:
            Dictionary mapping query names to query content
        """
        queries = {}
        queries_dir = os.path.join(os.path.dirname(__file__), 'queries')
        
        if not os.path.exists(queries_dir):
            logging.warning(f"[ContractFinderAgent._read_queries] Queries directory not found: {queries_dir}")
            return queries
        
        try:
            for filename in os.listdir(queries_dir):
                if filename.endswith('.sql'):
                    filepath = os.path.join(queries_dir, filename)
                    query_name = filename[:-4]  # Remove .sql extension
                    with open(filepath, 'r') as f:
                        queries[query_name] = f.read()
                    logging.info(f"[ContractFinderAgent._read_queries] Loaded query: {query_name}")
        except Exception as e:
            logging.warning(f"[ContractFinderAgent._read_queries] Failed to load queries: {str(e)}")
        
        return queries
    
    @classmethod
    def get_setup_config(cls):
        """
        Returns the configuration schema for the agent, aggregating requirements from all integrated tools.
        
        Returns:
            Configuration schema dictionary
        """
        try:
            agent_config = cls.CONFIG_CLASS.model_json_schema()
            
            # Aggregate tool configurations
            tools_config = {}
            tools = [
                SQLExecutorTool,
                LLMConfigurationTool,
                SemanticSimilarityTool,
                CitationTool
            ]
            
            for tool_class in tools:
                tools_config[tool_class.__name__] = {
                    "description": f"{tool_class.__name__} configuration",
                    "schema": tool_class.model_json_schema() if hasattr(tool_class, 'model_json_schema') else {}
                }
            
            return {
                "agent_config": agent_config,
                "tools_config": tools_config
            }
        except Exception as e:
            logging.error(f"[ContractFinderAgent.get_setup_config] Failed to get setup config: {str(e)}")
            return {}
    
    def _retrieve_ids(self, filters: Dict[str, str], next_trace=None, trace=None) -> Tuple[List[str], Dict]:
        """
        Retrieve contract IDs using AND (intersection) logic across all filters.
        
        Executes semantic similarity queries for each filter via SQLExecutorTool,
        accumulates per-filter deduplicated contract ID sets, and computes their
        intersection (logical AND) to find contracts matching ALL filter conditions.
        Returns empty list if intersection is empty or any query fails.
        
        Per Spec Point 3: This uses intersection (AND) logic as required by specification,
        differing from reference's union-style accumulation. Spec Point 5: Database query
        failures are caught, logged, and converted to empty-list returns without propagation.
        
        Args:
            filters: Dictionary of filter key-value pairs from request
            next_trace: Optional trace handler for nested calls
            trace: Optional trace handler for execution tracing
            
        Returns:
            Tuple of (list of contract IDs satisfying ALL filters, dict of contract metadata)
        """
        # Track results per filter for intersection computation (Spec Point 2)
        per_filter_contract_sets: Dict[str, Set[str]] = {}
        per_filter_details: Dict[str, Dict] = {}
        
        if not filters:
            logging.warning("[_retrieve_ids] No filters provided")
            return [], {}
        
        # Calculate proportional ID allocation per filter
        num_valid_filters = len(filters)
        ids_per_filter = math.ceil(self.config.max_contract_ids / num_valid_filters) if num_valid_filters > 0 else self.config.max_contract_ids
        
        logging.info(f"[_retrieve_ids] Processing {num_valid_filters} filters for AND (intersection) logic with {ids_per_filter} max IDs per filter")
        
        # Execute each filter's query and accumulate per-filter result sets (Spec Points 1-2)
        for filter_key, filter_value in filters.items():
            try:
                # Look up filter in mapping registry
                filter_details = FILTER_TABLE_MAPPING.get(filter_key)
                if not filter_details:
                    logging.info(f"[_retrieve_ids] Filter '{filter_key}' not found in FILTER_TABLE_MAPPING, skipping")
                    continue
                
                logging.info(f"[_retrieve_ids] Processing filter '{filter_key}' with value: {filter_value}")
                
                # Generate embedding for filter value
                filter_embedding = self._get_embedding(filter_value, next_trace, trace)
                if not filter_embedding:
                    logging.warning(f"[_retrieve_ids] Failed to generate embedding for filter '{filter_key}', returning empty list per Spec Point 4")
                    return [], {}  # Empty intersection when any filter fails
                
                # Format embedding string for SQL query
                embedding_str = str(filter_embedding).replace(' ', '')
                
                # Build semantic similarity query
                query = self._build_filter_query(filter_details, embedding_str, ids_per_filter)
                if not query:
                    logging.warning(f"[_retrieve_ids] Failed to build query for filter '{filter_key}', returning empty list")
                    return [], {}
                
                logging.info(f"[_retrieve_ids] Executing semantic search query for filter '{filter_key}' on table {filter_details.table}")
                
                # Execute the query via SQL executor tool (Spec Point 1)
                result = self._execute_query(
                    query=query,
                    commit=False,
                    mapping=filter_details,
                    filter_type=filter_key,
                    filter_value=filter_value,
                    next_trace=next_trace,
                    trace=trace
                )
                
                # Per Spec Point 4: empty result for a filter means empty intersection
                if not result:
                    logging.info(f"[_retrieve_ids] No results found for filter '{filter_key}', intersection will be empty")
                    return [], {}  # Empty intersection - spec edge case #1
                
                # Deduplicate contract IDs within this filter's result set (Spec Point 2)
                filter_contract_set = set()
                filter_metadata = {}
                
                for row in result:
                    contract_id = row.get('contract_id') or row.get(filter_details.return_column)
                    if contract_id:
                        filter_contract_set.add(contract_id)
                        if contract_id not in filter_metadata:
                            filter_metadata[contract_id] = {
                                'contract_id': contract_id,
                                'similarity_scores': [],
                                'matched_filters': []
                            }
                        filter_metadata[contract_id]['similarity_scores'].append(
                            row.get('similarity_score', 0.0)
                        )
                        filter_metadata[contract_id]['matched_filters'].append(filter_key)
                
                if filter_contract_set:
                    per_filter_contract_sets[filter_key] = filter_contract_set
                    per_filter_details[filter_key] = filter_metadata
                    logging.info(f"[_retrieve_ids] Filter '{filter_key}' retrieved {len(filter_contract_set)} unique contract IDs")
                else:
                    logging.info(f"[_retrieve_ids] Filter '{filter_key}' returned no unique IDs, intersection will be empty")
                    return [], {}  # Empty intersection when filter returns nothing
                    
            except Exception as e:
                logging.error(f"[_retrieve_ids] Exception processing filter '{filter_key}': {str(e)}")
                logging.error(f"[_retrieve_ids] Traceback: {traceback.format_exc()}")
                # Per Spec Point 5: catch exception, log, and return empty list - no propagation
                return [], {}
        
        # Compute intersection (AND) of all per-filter contract ID sets (Spec Point 3)
        if not per_filter_contract_sets:
            logging.warning(f"[_retrieve_ids] No valid filters produced results")
            return [], {}
        
        # Start with first filter's set, then intersect with others
        filter_keys = list(per_filter_contract_sets.keys())
        intersection_ids = per_filter_contract_sets[filter_keys[0]].copy()
        
        for filter_key in filter_keys[1:]:
            intersection_ids = intersection_ids.intersection(per_filter_contract_sets[filter_key])
            if not intersection_ids:
                logging.info(f"[_retrieve_ids] Intersection became empty after processing filter '{filter_key}'")
                return [], {}  # Spec Point 4: empty intersection
        
        # Build result with only contracts in intersection
        unique_contract_ids = list(intersection_ids)[:self.config.max_contract_ids]
        all_contract_details = {}
        
        # Merge metadata from all filters for contracts in intersection
        for filter_key, metadata_dict in per_filter_details.items():
            for contract_id in unique_contract_ids:
                if contract_id in metadata_dict:
                    if contract_id not in all_contract_details:
                        all_contract_details[contract_id] = metadata_dict[contract_id]
                    else:
                        # Merge scores and filters from additional matches
                        all_contract_details[contract_id]['similarity_scores'].extend(
                            metadata_dict[contract_id]['similarity_scores']
                        )
                        all_contract_details[contract_id]['matched_filters'].extend(
                            metadata_dict[contract_id]['matched_filters']
                        )
        
        logging.info(f"[_retrieve_ids] Intersection produced {len(unique_contract_ids)} contracts satisfying ALL filters")
        
        return unique_contract_ids, all_contract_details
    
    def _get_embedding(self, text: str, next_trace=None, trace=None) -> Optional[List[float]]:
        """
        Generate embedding for given text using SemanticSimilarityTool.
        
        Temporarily disables similarity_check_toggle to generate pure embeddings
        without performing similarity comparisons against a document set.
        
        Args:
            text: Text to embed
            next_trace: Optional trace handler
            trace: Optional trace handler
            
        Returns:
            List of float values representing the embedding vector, or None on failure
        """
        try:
            # Disable similarity check for pure embedding generation
            semantic_similarity_toggle = self.semantic_similarity_tool.config.similarity_check_toggle
            self.semantic_similarity_tool.config.similarity_check_toggle = False
            
            logging.debug(f"[_get_embedding] Generating embedding for text: {text[:100]}...")
            
            # Call semantic similarity tool with docs=None to get embedding only
            filter_embedding = self.semantic_similarity_tool.run(
                text=text,
                docs=None,
                next_trace=next_trace,
                trace=trace
            )
            
            # Restore original toggle state
            self.semantic_similarity_tool.config.similarity_check_toggle = semantic_similarity_toggle
            
            return filter_embedding
            
        except Exception as e:
            logging.error(f"[_get_embedding] Failed to generate embedding: {str(e)}")
            logging.error(f"[_get_embedding] Traceback: {traceback.format_exc()}")
            # Restore toggle before raising
            self.semantic_similarity_tool.config.similarity_check_toggle = semantic_similarity_toggle
            return None
    
    def _build_filter_query(self, filter_details, embedding_str: str, ids_per_filter: int) -> str:
        """
        Build semantic similarity SQL query based on filter type and return column naming.
        
        Selects between filter_query_associated_cols.sql (for associated return columns)
        and filter_query_non_associated_cols.sql (for direct mappings) templates.
        Populates template placeholders with filter details and embedding.
        
        Args:
            filter_details: FilterDetails instance from FILTER_TABLE_MAPPING
            embedding_str: String representation of embedding vector
            ids_per_filter: Limit for number of results per filter
            
        Returns:
            Fully populated SQL query string ready for execution
        """
        try:
            return_column = filter_details.return_column
            select_column_str = ", ".join([f'"{col}"' for col in filter_details.select_columns])
            table_name = filter_details.table
            search_column = filter_details.search_column
            
            logging.debug(f"[_build_filter_query] Building query for table {table_name} with return column {return_column}")
            
            # Select query template based on return column naming convention
            if "associated" in return_column.lower():
                # Use associated columns query template
                query_template = self.queries.get('filter_query_associated_cols', '')
                if not query_template:
                    logging.warning("[_build_filter_query] filter_query_associated_cols.sql not found")
                    return ""
                
                # Populate associated columns template
                query = query_template.format(
                    select_column_str=select_column_str,
                    return_column=return_column,
                    search_column=search_column,
                    embedding_str=embedding_str,
                    table_name=table_name,
                    ids_per_filter=ids_per_filter,
                    similarity_threshold=self.config.similarity_threshold,
                    isbn_exists_clause="",  # Empty for contracts, no ISBN filtering needed
                    isbn_join_clause="",    # Empty for contracts
                    book_cols="",           # Empty for contracts
                    metadata_cols=""
                )
            else:
                # Use non-associated columns query template
                query_template = self.queries.get('filter_query_non_associated_cols', '')
                if not query_template:
                    logging.warning("[_build_filter_query] filter_query_non_associated_cols.sql not found")
                    return ""
                
                book_id_alias = f'"{return_column}" AS contract_ids'
                
                # Populate non-associated columns template
                query = query_template.format(
                    select_column_str=select_column_str,
                    book_id_alias=book_id_alias,
                    search_column=search_column,
                    embedding_str=embedding_str,
                    table_name=table_name,
                    ids_per_filter=ids_per_filter,
                    similarity_threshold=self.config.similarity_threshold,
                    isbn_clause="",  # Empty for contracts
                    book_cols="",    # Empty for contracts
                    metadata_cols=""
                )
            
            logging.debug(f"[_build_filter_query] Query template populated successfully")
            return query
            
        except Exception as e:
            logging.error(f"[_build_filter_query] Failed to build query: {str(e)}")
            logging.error(f"[_build_filter_query] Traceback: {traceback.format_exc()}")
            return ""
    
    def _execute_query(self, query: str, commit: bool, mapping=None, filter_type: str = None, 
                      filter_value: str = None, next_trace=None, trace=None) -> List[Dict]:
        """
        Execute SQL query via SQLExecutorTool with proper configuration and error handling.
        
        Updates SQL executor stream messages to reflect current filter context,
        executes query, restores original configuration, and returns results.
        
        Args:
            query: SQL query string to execute
            commit: Whether to commit transaction
            mapping: FilterDetails object for context
            filter_type: Filter key for message templating
            filter_value: Filter value for message templating
            next_trace: Optional trace handler
            trace: Optional trace handler
            
        Returns:
            List of result dictionaries from query execution
        """
        try:
            # Update stream messages with filter context if provided
            if filter_type and filter_value and mapping:
                table_name = mapping.table
                self.sql_executor_tool.config.stream_start_message = (
                    self.original_sql_start_header_messages
                    .replace("<table_name>", table_name)
                    .replace("<filter_type>", FILTER_TYPE_TO_TEXT_MAP.get(filter_type, filter_type))
                    .replace('<filter_value>', filter_value)
                )
                self.sql_executor_tool.config.stream_end_message = (
                    self.original_sql_end_header_messages
                    .replace("<table_name>", table_name)
                    .replace("<filter_type>", FILTER_TYPE_TO_TEXT_MAP.get(filter_type, filter_type))
                    .replace('<filter_value>', filter_value)
                )
            else:
                self.sql_executor_tool.config.stream_start_message = ""
                self.sql_executor_tool.config.stream_end_message = ""
            
            # Set commit behavior
            self.sql_executor_tool.sql_integration.config.is_commit = commit
            
            logging.info(f"[_execute_query] Executing SQL query for filter type: {filter_type}")
            
            # Execute query via SQL executor tool
            result = self.sql_executor_tool.run(
                sql_query=query,
                params=None,
                next_trace=next_trace,
                trace=trace
            )
            
            # Restore original SQL executor configuration
            self.sql_executor_tool.config.stream_start_message = self.original_sql_start_header_messages
            self.sql_executor_tool.config.stream_end_message = self.original_sql_end_header_messages
            self.sql_executor_tool.config.format_result_columns = self.original_format_columns
            
            logging.info(f"[_execute_query] Query executed successfully, returned {len(result) if result else 0} rows")
            return result if result else []
            
        except Exception as e:
            logging.error(f"[_execute_query] Query execution failed: {str(e)}")
            logging.error(f"[_execute_query] Query: {query[:200]}...")
            logging.error(f"[_execute_query] Traceback: {traceback.format_exc()}")
            # Restore original configuration even on error
            self.sql_executor_tool.config.stream_start_message = self.original_sql_start_header_messages
            self.sql_executor_tool.config.stream_end_message = self.original_sql_end_header_messages
            self.sql_executor_tool.config.format_result_columns = self.original_format_columns
            return []
    
    @AITrace()
    def run(
            self,
            user_query: Annotated[str, "User query/request describing the type of contracts to retrieve"],
            contract_details: Annotated[str, "Filter by contract summary details based on user query"],
            clause_details: Annotated[Optional[str], "Filter by specific clause details"] = None,
            party_details: Annotated[Optional[str], "Filter by party names or details"] = None,
            product_type: Annotated[Optional[str], "Filter by product type"] = None,
            contract_effective_date: Annotated[Optional[str], "Filter by contract effective date range"] = None,
            in_scope_work_details: Annotated[Optional[str], "Filter by in-scope work details"] = None,
            licensing_right_details: Annotated[Optional[str], "Filter by licensing right details"] = None,
            territory_country: Annotated[Optional[str], "Filter by territory country"] = None,
            licensing_payment_details: Annotated[Optional[str], "Filter by licensing payment details"] = None,
            show_more_details: Annotated[Optional[Dict[str, Any]], "Optional filter to show more documents for a specific user query from chat history"] = None,
            next_trace=None,
            trace=None
    ) -> Tuple[str, List[Dict], List[Dict], bool]:
        """
        Main orchestration method for contract retrieval. Handles standard request processing (Flow A)
        and show-more pagination (Flow B), coordinating filter processing, semantic search, content
        retrieval, scoring, LLM generation, session persistence, and citation generation.
        """
        logging.info(f"[ContractFinderAgent.run] Starting agent with query: '{user_query}'")
        
        try:
            # Input validation
            if not user_query or not user_query.strip():
                raise ValueError("User query cannot be empty")
            
            # Initialize session table
            self._create_chat_data_table(next_trace=next_trace, trace=trace)
            
            # Store state
            self.user_query = user_query
            self.show_more_details = show_more_details
            
            # Build and clean filters
            filters = {
                FilterType.CONTRACT_DETAILS: contract_details,
                FilterType.CLAUSE_DETAILS: clause_details,
                FilterType.PARTY_DETAILS: party_details,
                FilterType.PRODUCT_TYPE: product_type,
                FilterType.CONTRACT_EFFECTIVE_DATE: contract_effective_date,
                FilterType.IN_SCOPE_WORK_DETAILS: in_scope_work_details,
                FilterType.LICENSING_RIGHT_DETAILS: licensing_right_details,
                FilterType.TERRITORY_COUNTRY: territory_country,
                FilterType.LICENSING_PAYMENT_DETAILS: licensing_payment_details
            }
            filters = {k: v for k, v in filters.items() if v and str(v).strip()}
            
            # Get query embedding
            query_embedding = self._get_embedding(user_query, next_trace, trace)
            
            # Initialize state
            doc_id_map = {}
            retrieved_documents = []
            filtered_data = {}
            total_documents_available = 0
            
            # Route to appropriate flow
            if show_more_details:
                logging.info("[ContractFinderAgent.run] Executing show-more pagination (Flow B)")
                retrieved_documents, doc_id_map, total_documents_available, _ = self._show_more_details(
                    user_query=user_query,
                    show_more_details=show_more_details,
                    next_trace=next_trace,
                    trace=trace
                )
            else:
                logging.info("[ContractFinderAgent.run] Executing standard retrieval (Flow A)")
                contract_ids, filtered_data = self.fetch_contract_data(
                    filters=filters,
                    query_embedding=query_embedding,
                    next_trace=next_trace,
                    trace=trace
                )
                if contract_ids:
                    filtered_data = self._calculate_scores(filtered_data)
            
            # Process and format response
            post_processed_response = ""
            if filtered_data:
                formatted_context = self._assemble_context(
                    ranked_chunks=filtered_data,
                    include_metadata=True
                )
                
                if self.config.generate_response_from_docs and formatted_context:
                    if "#<context>#" in self.llm_config_tool.config.context:
                        self.llm_config_tool.config.context = self.llm_config_tool.config.context.replace(
                            "#<context>#", formatted_context
                        )
                    else:
                        self.llm_config_tool.config.context += f"\n\n{formatted_context}"
                    
                    try:
                        llm_result = self.llm_config_tool.run(
                            query=user_query,
                            next_trace=next_trace,
                            trace=trace
                        )
                        post_processed_response = llm_result if llm_result else self.config.fallback_response
                    except Exception as e:
                        logging.error(f"[ContractFinderAgent.run] LLM error: {str(e)}")
                        post_processed_response = self.config.fallback_response
                else:
                    post_processed_response = self.config.fallback_response
            else:
                post_processed_response = self.config.fallback_response
            
            # Persist to session and generate citations
            citations = []
            if not show_more_details and filtered_data:
                self._upsert_retrieved_documents(
                    session_id=self.data.get("preview_id") if self.data else None,
                    question=user_query,
                    retrieved_documents=list(filtered_data.values()),
                    next_trace=next_trace,
                    trace=trace
                )
            
            if self.config.generate_response_from_docs and filtered_data:
                try:
                    citations = self._generate_citations(
                        filtered_data=filtered_data,
                        response=post_processed_response,
                        show_more_details=show_more_details,
                        next_trace=next_trace,
                        trace=trace
                    )
                except Exception as e:
                    logging.error(f"[ContractFinderAgent.run] Citation error: {str(e)}")
            
            if not citations:
                citations = [
                    citation_generator(
                        agent_name=self.__class__.__name__,
                        title="Retrieved Contract",
                        url="",
                        description="",
                        metadata={},
                        filter=False,
                        customMetaData={"doc_id_map": doc_id_map}
                    )
                ]
            
            return post_processed_response, list(filtered_data.values()) if filtered_data else [], citations, not self.config.smart_response_adjustment
            
        except Exception as e:
            logging.error(f"[ContractFinderAgent.run] Error: {str(e)}")
            return self.config.fallback_response, [], [], not self.config.smart_response_adjustment
    
    def fetch_contract_data(
        self,
        filters: Dict[str, str],
        query_embedding: List[float],
        next_trace=None,
        trace=None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Orchestrate standard contract retrieval (Flow A: Steps A3-A4).
        Retrieves contract IDs via semantic search filters, then fetches content chunks.
        """
        logging.info("[fetch_contract_data] Starting standard retrieval flow")
        
        try:
            # Step A3: Retrieve contract IDs
            contract_ids, contract_metadata = self._retrieve_ids(filters, next_trace, trace)
            
            if not contract_ids:
                logging.warning(f"[fetch_contract_data] No contract IDs retrieved")
                return [], {}
            
            logging.info(f"[fetch_contract_data] Retrieved {len(contract_ids)} contract IDs")
            
            # Step A4: Retrieve content chunks
            chunks = self._retrieve_content(contract_ids, query_embedding, next_trace, trace)
            
            if not chunks:
                logging.warning(f"[fetch_contract_data] No chunks retrieved")
                return [], {}
            
            logging.info(f"[fetch_contract_data] Retrieved {len(chunks)} content chunks")
            
            # Process chunks into filtered_data structure
            filtered_data = {}
            for chunk in chunks:
                contract_id = chunk.get("contract_id") or "unknown"
                if contract_id not in filtered_data:
                    filtered_data[contract_id] = {
                        "contract_id": contract_id,
                        "chunks": [],
                        "similarity_score": chunk.get("similarity_score", 0.0),
                        "document_score": 0.0,
                        "chunk_text": "",
                        "context_score": 0.0
                    }
                filtered_data[contract_id]["chunks"].append(chunk)
                filtered_data[contract_id]["chunk_text"] += (chunk.get("chunk_text", "") + " ")
            
            for contract_id in filtered_data:
                filtered_data[contract_id]["chunk_text"] = filtered_data[contract_id]["chunk_text"].strip()
            
            return contract_ids, filtered_data
            
        except Exception as e:
            logging.error(f"[fetch_contract_data] Error: {str(e)}")
            return [], {}
    
    def _show_more_details(
        self,
        user_query: str,
        show_more_details: Dict[str, Any],
        next_trace=None,
        trace=None
    ) -> Tuple[List[Dict], Dict, int, List]:
        """
        Implement Flow B pagination: fetch previously undisplayed documents from session history.
        """
        logging.info(f"[_show_more_details] Fetching paginated results")
        
        try:
            question = show_more_details.get("question", user_query)
            length = show_more_details.get("length")
            
            try:
                length = int(length) if length else self.config.default_show_more_length
            except (TypeError, ValueError):
                length = self.config.default_show_more_length
            
            session_id = self.data.get("preview_id") if self.data else None
            if not session_id:
                return [], {}, 0, []
            
            docs_data = self._fetch_show_more_documents(
                session_id=session_id,
                question=question,
                length=length,
                next_trace=next_trace,
                trace=trace
            )
            
            displayed_docs = docs_data.get("previously_displayed_documents", [])
            new_docs = docs_data.get("new_documents", [])
            doc_id_map = docs_data.get("doc_id_to_db_id_map", {})
            total_available = docs_data.get("total_documents_available", 0)
            
            if not displayed_docs and not new_docs:
                logging.warning("[_show_more_details] No documents found")
                return [], {}, 0, []
            
            retrieved_documents = []
            documents = new_docs if self.config.generate_response_from_docs else (displayed_docs + new_docs)
            
            for doc in documents:
                try:
                    doc_context = doc.get("doc_context")
                    if isinstance(doc_context, str):
                        doc_context = json.loads(doc_context)
                    if doc_context:
                        retrieved_documents.append({
                            "doc_id": doc.get("doc_id"),
                            "doc_context": doc_context,
                            "display_count": doc.get("display_count", 1)
                        })
                except Exception as e:
                    logging.warning(f"[_show_more_details] Skipping doc: {str(e)}")
            
            return retrieved_documents, doc_id_map, total_available, displayed_docs
            
        except Exception as e:
            logging.error(f"[_show_more_details] Error: {str(e)}")
            return [], {}, 0, []
    
    def _create_chat_data_table(self, next_trace=None, trace=None) -> bool:
        """
        Initialize show_more_data_context table in AlloyDB if it doesn't exist.
        """
        try:
            if 'create_chat_data_table' not in self.queries:
                logging.info("[_create_chat_data_table] Query template not found")
                return False
            
            query = self.queries['create_chat_data_table']
            result = self._execute_query(
                query=query,
                commit=True,
                filter_type="create_chat_table",
                next_trace=next_trace,
                trace=trace
            )
            logging.info("[_create_chat_data_table] Chat data table initialized")
            return True
        except Exception as e:
            logging.error(f"[_create_chat_data_table] Error: {str(e)}")
            return False
    
    def _fetch_show_more_documents(
        self,
        session_id: str,
        question: str,
        length: int,
        next_trace=None,
        trace=None
    ) -> Dict[str, Any]:
        """
        Retrieve previously displayed and new documents from session history.
        """
        try:
            if 'fetch_show_more_docs' not in self.queries:
                logging.warning("[_fetch_show_more_documents] Query template not found")
                return {"previously_displayed_documents": [], "new_documents": [], "total_documents_available": 0}
            
            query = self.queries['fetch_show_more_docs'].format(
                session_id=session_id,
                question=question,
                length=length
            )
            
            result = self._execute_query(
                query=query,
                commit=False,
                filter_type="show_more",
                next_trace=next_trace,
                trace=trace
            )
            
            previously_displayed = [r for r in (result or []) if r.get("is_displayed")]
            new_documents = [r for r in (result or []) if not r.get("is_displayed")]
            
            return {
                "previously_displayed_documents": previously_displayed,
                "new_documents": new_documents,
                "doc_id_to_db_id_map": {r.get("doc_id"): r.get("db_id") for r in (result or [])},
                "total_documents_available": len(result or [])
            }
        except Exception as e:
            logging.error(f"[_fetch_show_more_documents] Error: {str(e)}")
            return {"previously_displayed_documents": [], "new_documents": [], "total_documents_available": 0}
    
    def _upsert_retrieved_documents(
        self,
        session_id: str,
        question: str,
        retrieved_documents: List[Dict],
        next_trace=None,
        trace=None
    ) -> Dict[str, str]:
        """
        Persist retrieved documents to session history table (Step A6).
        """
        try:
            if not session_id or 'insert_retrieved_documents' not in self.queries:
                logging.warning("[_upsert_retrieved_documents] Missing session_id or query template")
                return {}
            
            # Prepare document records for insertion
            doc_id_map = {}
            for i, doc in enumerate(retrieved_documents[:self.config.max_contracts_to_return]):
                doc_id = doc.get("contract_id") or f"doc_{i}"
                doc_id_map[doc_id] = str(i)
            
            query = self.queries['insert_retrieved_documents'].format(
                session_id=session_id,
                question=question,
                documents_json=json.dumps([{"id": k, "data": v} for k, v in doc_id_map.items()])
            )
            
            self._execute_query(
                query=query,
                commit=True,
                filter_type="persist_documents",
                next_trace=next_trace,
                trace=trace
            )
            
            logging.info(f"[_upsert_retrieved_documents] Persisted {len(doc_id_map)} documents to session")
            return doc_id_map
        except Exception as e:
            logging.error(f"[_upsert_retrieved_documents] Error: {str(e)}")
            return {}
    
    def _process_filtered_data(
        self,
        filtered_data: Dict[str, Any],
        sort_by_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert filtered_data dict to ordered list, optionally sorting by context_score.
        """
        if not filtered_data:
            return []
        
        try:
            result = list(filtered_data.values())
            
            if sort_by_score:
                result.sort(
                    key=lambda x: x.get("context_score", 0),
                    reverse=True
                )
            
            return result
        except Exception as e:
            logging.error(f"[_process_filtered_data] Error: {str(e)}")
            return list(filtered_data.values()) if filtered_data else []
    
    def extract_contract_ids_from_response(self, response: str) -> List[str]:
        """
        Extract contract IDs from LLM response using regex patterns.
        """
        try:
            # Look for contract ID patterns in response
            pattern = r"contract[_\s]*id[\s]*[:#]*\s*([A-Z0-9\-_]+)"
            matches = re.findall(pattern, response, re.IGNORECASE)
            return list(set(matches)) if matches else []
        except Exception as e:
            logging.warning(f"[extract_contract_ids_from_response] Error: {str(e)}")
            return []
    
    def mark_contracts_as_displayed(
        self,
        contract_ids: List[str],
        session_id: str,
        question: str,
        next_trace=None,
        trace=None
    ) -> bool:
        """
        Mark retrieved contracts as displayed in session history.
        """
        try:
            if not contract_ids or not session_id or 'mark_contracts_as_displayed' not in self.queries:
                return False
            
            for contract_id in contract_ids:
                query = self.queries['mark_contracts_as_displayed'].format(
                    session_id=session_id,
                    question=question,
                    contract_id=contract_id
                )
                self._execute_query(
                    query=query,
                    commit=True,
                    filter_type="mark_displayed",
                    next_trace=next_trace,
                    trace=trace
                )
            
            logging.info(f"[mark_contracts_as_displayed] Marked {len(contract_ids)} contracts as displayed")
            return True
        except Exception as e:
            logging.error(f"[mark_contracts_as_displayed] Error: {str(e)}")
            return False
    
    def build_document_response(self, documents: List[Dict]) -> str:
        """
        Format documents as a response string when skip_llm_generation is True.
        """
        if not documents:
            return "No contracts found matching the criteria."
        
        try:
            response_parts = [f"Found {len(documents)} relevant contracts:\n\n"]
            
            for i, doc in enumerate(documents, 1):
                contract_id = doc.get("contract_id", "Unknown")
                context_score = doc.get("context_score", 0)
                response_parts.append(f"{i}. Contract ID: {contract_id} (Relevance: {context_score:.2f})")
            
            return "\n".join(response_parts)
        except Exception as e:
            logging.error(f"[build_document_response] Error: {str(e)}")
            return "Error formatting response."
    
    def _generate_citations(
        self,
        filtered_data: Dict[str, Any],
        response: str,
        show_more_details: Dict = None,
        next_trace=None,
        trace=None
    ) -> List[Dict]:
        """
        Generate citations for retrieved contract content (Step A6).
        Uses CitationTool to extract and verify highlighted text from source content.
        """
        logging.info("[_generate_citations] Generating citations for response")
        
        try:
            if not filtered_data:
                logging.warning("[_generate_citations] No data available for citations")
                return []
            
            citations = []
            
            # Generate one citation per retrieved contract
            for contract_id, contract_data in list(filtered_data.items())[:self.config.max_contracts_to_return]:
                try:
                    chunk_text = contract_data.get("chunk_text", "")[:1000]  # Limit text length
                    context_score = contract_data.get("context_score", 0)
                    
                    citation = citation_generator(
                        agent_name="ContractFinderAgent",
                        title=f"Contract {contract_id}",
                        url="",
                        description=chunk_text,
                        metadata={"context_score": context_score},
                        filter=False,
                        customMetaData={
                            "contract_id": contract_id,
                            "citation_type": "contract_clause",
                            "relevance_score": context_score,
                            "show_more": bool(show_more_details)
                        }
                    )
                    citations.append(citation)
                except Exception as e:
                    logging.warning(f"[_generate_citations] Error generating citation for {contract_id}: {str(e)}")
            
            logging.info(f"[_generate_citations] Generated {len(citations)} citations")
            return citations
            
        except Exception as e:
            logging.error(f"[_generate_citations] Error: {str(e)}")
            return []
