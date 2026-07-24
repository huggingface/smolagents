from typing import Any, Union, List, Dict
import copy
import json
import logging


def safe_json_load(json_input: Union[str, dict, list, None]) -> Any:
    """
    Safely load a JSON string into a Python object.
    If value is already a dict/list, returns as-is.
    Returns {} on error or if value is empty.
    Per LLM Studio framework guidance, this utility safely handles mixed input types
    (raw dict/list from API, stringified JSON from database) without requiring
    type guards at call sites.
    
    Args:
        json_input: Input that may be dict, list, JSON string, or None
        
    Returns:
        Parsed Python object (dict/list/value) or {} on any parse failure
    """
    if not json_input:
        return {}
    try:
        if isinstance(json_input, (dict, list)):
            return json_input
        if isinstance(json_input, str):
            return json.loads(json_input)
    except Exception as e:
        logging.warning(f"[safe_json_load] Failed to parse JSON input: {str(e)}")
        return {}
    return {}


def remove_extras_from_retrieved_documents(
    retrieved_documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove '_extras', internal scoring fields, and other unwanted keys from retrieved documents.
    Follows the LLM Studio pattern of deep copying before mutation to preserve caller's data.
    
    Removes per-chunk internal fields (_extras, html_elements, similarity_score) and
    per-contract internal scoring fields (context_score, document_score) that should not
    appear in LLM context or user-facing responses.
    
    Args:
        retrieved_documents: List of contract document dicts from retrieval phase
        
    Returns:
        Deep copy of documents with internal fields removed
    """
    cleaned_documents = copy.deepcopy(retrieved_documents)

    for contract_entry in cleaned_documents:
        for _, value in contract_entry.items():
            if not isinstance(value, dict):
                continue

            # Remove contract-level internal scores
            value.pop("context_score", None)
            value.pop("similarity_score", None)
            value.pop("document_score", None)

            # Process contract clauses/excerpts
            clauses = value.get("clauses", [])
            if isinstance(clauses, list):
                for clause in clauses:
                    if isinstance(clause, dict):
                        # Remove internal fields from each clause
                        clause.pop("_extras", None)
                        clause.pop("html_elements", None)
                        clause.pop("similarity_score", None)
                        clause.pop("chunk_id", None)

            value["clauses"] = clauses if isinstance(clauses, list) else []

    return cleaned_documents


def handle_invalid_citation_sources(citation_type: str, relevant_doc: Dict[str, Any]) -> Dict[str, str]:
    """
    Handle invalid or missing citation sources by constructing fallback content.
    When exact-match citation extraction fails, this utility builds a safe fallback
    response from available metadata.
    
    Args:
        citation_type: Type of citation being processed (e.g., 'contract_clause', 'metadata')
        relevant_doc: Contract document containing fallback information
        
    Returns:
        Dictionary with 'content' and 'source' keys for fallback citation
    """
    logging.info(f"[handle_invalid_citation_sources] Constructing fallback for citation_type: {citation_type}")
    
    complete_content = ""
    citation_source = "contract_metadata"
    
    try:
        if citation_type == "contract_clause":
            # Fallback to contract summary when clause extraction fails
            contract_summary = relevant_doc.get("contract_details", {}).get("contract_summary", "")
            complete_content = "<strong>Contract Summary</strong><br><br>"
            complete_content += contract_summary if contract_summary else "[No summary available]"
            citation_source = "contract_summary"
        else:
            # Default fallback to contract metadata
            contract_name = relevant_doc.get("contract_details", {}).get("contract_name", "")
            complete_content = "<strong>Contract Information</strong><br><br>"
            complete_content += f"Contract: {contract_name}<br>" if contract_name else ""
            citation_source = "contract_details"
    except Exception as e:
        logging.warning(f"[handle_invalid_citation_sources] Error constructing fallback: {str(e)}")
        complete_content = "[Citation source unavailable]"
        citation_source = "fallback"
    
    return {
        "content": complete_content,
        "source": citation_source
    }


def add_previously_displayed_documents_to_filtered_data(
    current_documents: List[Dict[str, Any]],
    previously_displayed: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge previously displayed contract documents with new retrieval results.
    Used in the show-more pagination flow (Step B1) to combine session history
    with fresh retrieval.
    
    Args:
        current_documents: Newly retrieved contract documents
        previously_displayed: Contract documents from session history already shown to user
        
    Returns:
        Combined list of current + previously displayed documents
    """
    try:
        # Deduplicate by contract_id using dict for O(1) lookup
        seen_ids = {doc.get('contract_id'): True for doc in current_documents if doc.get('contract_id')}
        
        merged = copy.deepcopy(current_documents)
        
        for prev_doc in previously_displayed:
            contract_id = prev_doc.get('contract_id')
            if contract_id and contract_id not in seen_ids:
                merged.append(prev_doc)
                seen_ids[contract_id] = True
        
        logging.info(f"[add_previously_displayed_documents_to_filtered_data] Merged {len(merged)} documents")
        return merged
        
    except Exception as e:
        logging.error(f"[add_previously_displayed_documents_to_filtered_data] Merge failed: {str(e)}")
        return current_documents

