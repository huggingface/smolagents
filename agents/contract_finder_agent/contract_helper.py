from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class FilterType(str, Enum):
    """Enum for contract-specific filter types."""
    CONTRACT_DETAILS = "contract_details"
    PRODUCT_TYPE = "product_type"
    PARTY_DETAILS = "party_details"
    IN_SCOPE_WORK_DETAILS = "in_scope_work_details"
    LICENSING_RIGHT_DETAILS = "licensing_right_details"
    TERRITORY_COUNTRY = "territory_country"
    LICENSING_PAYMENT_DETAILS = "licensing_payment_details"
    CLAUSE_DETAILS = "clause_details"


FILTER_TYPE_TO_TEXT_MAP = {
    FilterType.CONTRACT_DETAILS: "Contract Details",
    FilterType.PRODUCT_TYPE: "Product Type",
    FilterType.PARTY_DETAILS: "Party Details",
    FilterType.IN_SCOPE_WORK_DETAILS: "In-Scope Work Details",
    FilterType.LICENSING_RIGHT_DETAILS: "Licensing Right Details",
    FilterType.TERRITORY_COUNTRY: "Territory Country",
    FilterType.LICENSING_PAYMENT_DETAILS: "Licensing Payment Details",
    FilterType.CLAUSE_DETAILS: "Clause Details"
}


class FilterDetails(BaseModel):
    """Pydantic model for filter configuration."""
    table: str
    search_column: str
    return_column: str
    filter_title: str
    select_columns: List[str]
    primary_key: str
    mapping_type: str
    filter_location: Optional[str] = None
    filter_mode: str
    select_exclude_columns: Optional[List[str]] = None


FILTER_TABLE_MAPPING: Dict[str, FilterDetails] = {
    FilterType.CONTRACT_DETAILS.value: FilterDetails(
        table="contract_test_v2",
        search_column="contract_summary_embeddings",
        return_column="contract_id",
        filter_title="Contract Details",
        select_columns=["contract_name", "contract_summary", "contract_type"],
        primary_key="contract_id",
        mapping_type="single",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=["contract_name", "contract_summary"]
    ),
    FilterType.PRODUCT_TYPE.value: FilterDetails(
        table="contract_test_v2",
        search_column="product_types_embeddings",
        return_column="contract_id",
        filter_title="Product Type",
        select_columns=["product_types", "contract_name"],
        primary_key="contract_id",
        mapping_type="single",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=["product_types", "contract_name"]
    ),
    FilterType.PARTY_DETAILS.value: FilterDetails(
        table="contract_party_test_v2",
        search_column="party_name_embeddings",
        return_column="party_id",
        filter_title="Party Details",
        select_columns=["party_name", "party_role", "party_address"],
        primary_key="party_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="hard_filter",
        select_exclude_columns=[]
    ),
    FilterType.IN_SCOPE_WORK_DETAILS.value: FilterDetails(
        table="in_scope_work_test_v2",
        search_column="work_summary_embeddings",
        return_column="contract_id",
        filter_title="In-Scope Work Details",
        select_columns=["work_description", "work_type"],
        primary_key="work_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=[]
    ),
    FilterType.LICENSING_RIGHT_DETAILS.value: FilterDetails(
        table="licensing_right_test_v2",
        search_column="licensing_right_summary_embeddings",
        return_column="contract_id",
        filter_title="Licensing Right Details",
        select_columns=["right_description", "right_type"],
        primary_key="licensing_right_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=[]
    ),
    FilterType.TERRITORY_COUNTRY.value: FilterDetails(
        table="legal_licensing_right_territory_test_v2",
        search_column="country_embeddings",
        return_column="territory_id",
        filter_title="Territory Country",
        select_columns=["country_name", "region"],
        primary_key="territory_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=[]
    ),
    FilterType.LICENSING_PAYMENT_DETAILS.value: FilterDetails(
        table="licensing_payment_detail_test_v2",
        search_column="payment_conditions_embeddings",
        return_column="contract_id",
        filter_title="Licensing Payment Details",
        select_columns=["payment_amount", "payment_type", "payment_currency"],
        primary_key="payment_detail_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=[]
    ),
    FilterType.CLAUSE_DETAILS.value: FilterDetails(
        table="legal_clause_test_v2",
        search_column="source_clause_embeddings",
        return_column="contract_id",
        filter_title="Clause Details",
        select_columns=["clause_text", "clause_type", "clause_title"],
        primary_key="clause_id",
        mapping_type="multiple",
        filter_location="contract_metadata",
        filter_mode="soft_filter",
        select_exclude_columns=[]
    )
}


class ContractFilterType(BaseModel):
    """Pydantic model for metadata exact-match filter configuration."""
    table: str
    match_column: str
    return_column: str


METADATA_EXACT_FILTERS: Dict[str, ContractFilterType] = {
    "contract_name": ContractFilterType(
        table="contract_details_backup_2",
        match_column="contract_name",
        return_column="contract_id"
    )
}

