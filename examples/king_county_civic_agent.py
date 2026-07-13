"""
King County, WA civic data tools for smolagents.

Three tools that let an agent look up property and contractor information
for any address in King County, Washington State — no API keys required.

    pip install smolagents
    python king_county_civic_agent.py

Tools:
    king_county_address_to_parcel  — street address → 10-digit parcel number
    king_county_permit_status      — permits by address, parcel, or permit number
    wa_contractor_license          — WA contractor license status via L&I Verify

All three tools return JSON strings with an `action` field for agent routing:
    found / use  — result is ready to consume
    pick         — multiple candidates, agent should narrow query
    none         — no match found
    refine       — network issue, retry
    reject       — bad input
"""

import http.cookiejar
import json
import urllib.parse
import urllib.request

from smolagents import CodeAgent, InferenceClientModel, tool


# ---------------------------------------------------------------------------
# Tool 1: Address → Parcel Number
# ---------------------------------------------------------------------------


@tool
def king_county_address_to_parcel(address: str) -> str:
    """
    Convert a King County, WA street address to its 10-digit parcel number (PIN).

    Uses the King County ArcGIS geocoder. Returns a JSON string with:
      - action: "use" (exact match), "pick" (ambiguous), "refine" (no match), "reject" (bad input)
      - parcel_number: the 10-digit PIN when action is "use" or "pick"
      - matched_address: geocoder's canonical form of the address
      - score: match confidence 0–100 (>=90 is reliable)

    Args:
        address: Street address in King County, WA. Include house number and city.
                 Also accepts a bare 10-digit parcel number as pass-through.
                 Examples: "1817 Morris Ave S, Renton WA", "600 Grady Way, Renton", "7222000353"
    """
    address = address.strip()
    if not address:
        return json.dumps({"action": "reject", "message": "Address is required.", "parcel_number": None})

    # Bare 10-digit PIN passthrough
    import re
    if re.fullmatch(r"\d{10}", address):
        return json.dumps({"action": "use", "parcel_number": address, "matched_address": None, "score": 100,
                           "message": f"Input is already a parcel number: {address}"})

    geocoder_url = (
        "https://gismaps.kingcounty.gov/arcgis/rest/services"
        "/Address/KingCo_ParcelAddress_locator/GeocodeServer/findAddressCandidates"
    )
    params = urllib.parse.urlencode({"SingleLine": address, "outFields": "*", "maxLocations": 5, "f": "json"})
    try:
        req = urllib.request.Request(f"{geocoder_url}?{params}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        return json.dumps({"action": "refine", "message": f"Geocoder unavailable: {e}", "parcel_number": None})

    candidates = data.get("candidates", [])
    if not candidates:
        return json.dumps({"action": "refine", "message": f"No match for '{address}'.", "parcel_number": None})

    best = candidates[0]
    score = best.get("score", 0)
    attrs = best.get("attributes", {})
    pin = attrs.get("PIN", "")
    matched = attrs.get("Match_addr", "")

    if score >= 90 and pin and len(pin) == 10:
        return json.dumps({"action": "use", "parcel_number": pin, "matched_address": matched, "score": score,
                           "message": f"Matched: {matched} → parcel {pin}"})

    alts = [{"address": c.get("attributes", {}).get("Match_addr", ""),
              "parcel_number": c.get("attributes", {}).get("PIN", ""),
              "score": c.get("score", 0)}
             for c in candidates if len(c.get("attributes", {}).get("PIN", "")) == 10]
    if alts:
        return json.dumps({"action": "pick", "parcel_number": alts[0]["parcel_number"],
                           "matched_address": alts[0]["address"], "score": alts[0]["score"],
                           "candidates": alts, "message": f"Low confidence. Best guess: {alts[0]['address']}"})

    return json.dumps({"action": "refine", "message": f"Could not resolve '{address}' to a parcel.", "parcel_number": None})


# ---------------------------------------------------------------------------
# Tool 2: Permit Status
# ---------------------------------------------------------------------------


@tool
def king_county_permit_status(query: str) -> str:
    """
    Look up building permit history and status for any King County, WA property.

    Searches MyBuildingPermit.com, which covers Auburn, Bellevue, Bothell,
    Burien, Federal Way, Issaquah, Kenmore, King County (unincorporated),
    Kirkland, Mercer Island, Newcastle, Sammamish, Snoqualmie, and more.

    Returns a JSON string with:
      - action: "found" (permits returned), "none" (no permits), "refine" (connection issue)
      - permits: list of permit records when action is "found"
      - searched: which jurisdictions were queried

    Each permit includes: permit_number, type, status, description, address,
    jurisdiction, applied_date, issued_date, finaled_date, expires_date, portal.

    Args:
        query: Street address, 10-digit parcel number, or permit number.
               Examples: "1817 Morris Ave S, Renton WA", "7222000353", "B25000947"
    """
    query = query.strip()
    if not query:
        return json.dumps({"action": "reject", "message": "Query is required.", "permits": []})

    mbp_url = "https://permitsearch.mybuildingpermit.com/api/permits/search"
    payload = json.dumps({"searchType": "Address", "searchValue": query, "pageSize": 25, "pageNumber": 1}).encode()
    try:
        req = urllib.request.Request(mbp_url, data=payload,
                                     headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as e:
        return json.dumps({"action": "refine", "message": f"Could not reach permit database: {e}", "permits": []})

    raw_permits = data.get("permits") or data.get("results") or data.get("data") or []
    if not raw_permits and isinstance(data, list):
        raw_permits = data

    if not raw_permits:
        return json.dumps({"action": "none", "message": f"No permits found for '{query}'.",
                           "searched": ["King County (MyBuildingPermit)"], "permits": []})

    def _date(raw):
        if not raw:
            return None
        try:
            import re as _re
            m = _re.search(r"(\d{4}-\d{2}-\d{2})", str(raw))
            return m.group(1) if m else str(raw)[:10]
        except Exception:
            return None

    permits = []
    seen = set()
    for p in raw_permits:
        pn = str(p.get("PermitNumber") or p.get("permit_number") or "")
        if not pn or pn in seen:
            continue
        seen.add(pn)
        permits.append({
            "permit_number": pn,
            "type": p.get("PermitType") or p.get("permit_type") or "",
            "status": p.get("Status") or p.get("status") or "",
            "description": p.get("Description") or p.get("description") or "",
            "address": p.get("Address") or p.get("address") or "",
            "jurisdiction": p.get("Jurisdiction") or p.get("jurisdiction") or "",
            "applied_date": _date(p.get("AppliedDate") or p.get("applied_date")),
            "issued_date": _date(p.get("IssuedDate") or p.get("issued_date")),
            "finaled_date": _date(p.get("FinaledDate") or p.get("finaled_date")),
            "expires_date": _date(p.get("ExpiresDate") or p.get("expires_date")),
        })

    permits.sort(key=lambda x: x.get("applied_date") or "", reverse=True)
    return json.dumps({"action": "found", "permit_count": len(permits),
                       "searched": ["King County (MyBuildingPermit)"], "permits": permits})


# ---------------------------------------------------------------------------
# Tool 3: WA Contractor License
# ---------------------------------------------------------------------------


@tool
def wa_contractor_license(query: str) -> str:
    """
    Verify a Washington State contractor's license status via WA L&I Verify.

    Checks contractor registration, workers' comp status, and any safety or
    contractor violations on record. No API key required.

    Returns a JSON string with:
      - action: "found" (match), "pick" (multiple — narrow query), "none" (not found), "reject" (bad input)
      - results: list of matching contractor records
      - total_found: total matches in L&I database

    Each result includes: license_id, business_name, contractor_type, contractor_group,
    status (Active/Expired/Inactive), city, state, ubi, violations list, detail_url.

    Args:
        query: Business name, license ID, or 9-digit UBI number.
               Examples: "Acme Plumbing", "MORTESL763NR", "605417027"
    """
    query = query.strip()
    if len(query) < 2:
        return json.dumps({"action": "reject", "message": "Query must be at least 2 characters.", "results": []})

    import re as _re
    digits_only = _re.sub(r"[\s\-]", "", query)
    if _re.fullmatch(r"\d{9}", digits_only):
        search_cat, search_text = "Ubi", digits_only
    elif _re.fullmatch(r"[A-Z0-9*]{6,15}", query.upper()):
        search_cat, search_text = "LicenseId", query.upper()
    else:
        search_cat, search_text = "Name", query

    verify_base = "https://secure.lni.wa.gov/verify"
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    # Establish ASP.NET session (required before search)
    try:
        warmup_url = f"{verify_base}/Results.aspx#init"
        for url, data in [
            (f"{verify_base}/default.aspx", None),
            (warmup_url, None),
            (f"{verify_base}/SessionHandler.aspx", json.dumps({"hash": warmup_url}).encode()),
        ]:
            headers = {"User-Agent": ua}
            if data:
                headers.update({"Content-Type": "application/json; charset=UTF-8", "X-Requested-With": "XMLHttpRequest"})
            req = urllib.request.Request(url, data=data, headers=headers)
            with opener.open(req, timeout=15) as r:
                r.read()
    except Exception as e:
        return json.dumps({"action": "refine", "message": f"Could not reach WA L&I: {e}", "results": []})

    # Run the search
    search_dto = {
        "pageNumber": 0, "SearchType": 2, "SortColumn": "Rank", "SortOrder": "desc",
        "pageSize": 25, "ContractorTypeFilter": [], "SessionID": "", "SAW": "",
        "searchCat": search_cat, "searchText": search_text,
        search_cat: search_text,  # required: named field must match searchCat key
        "firstSearch": 1,
    }
    results_url = f"{verify_base}/Results.aspx#{urllib.parse.quote(json.dumps(search_dto))}"
    try:
        req = urllib.request.Request(
            f"{verify_base}/Controller.aspx/Search",
            data=json.dumps({"dtoSrch": search_dto}).encode(),
            headers={"User-Agent": ua, "Content-Type": "application/json; charset=UTF-8",
                     "X-Requested-With": "XMLHttpRequest", "Referer": results_url,
                     "Accept": "application/json, text/javascript, */*; q=0.01"},
        )
        with opener.open(req, timeout=15) as r:
            data = json.loads(r.read())["d"]
    except Exception as e:
        return json.dumps({"action": "refine", "message": f"Search failed: {e}", "results": []})

    total = data.get("TotalCount", 0)
    if total == 0:
        return json.dumps({"action": "none", "message": f"No WA contractor found for '{query}'.", "results": []})

    def _status(row):
        code = row.get("IrlStatusCode", "") or ""
        s = row.get("Status", "") or ""
        if code == "A" or s == "View Details":
            return "Active"
        if code in ("E", "X"):
            return "Expired"
        return "Inactive" if s.lower() == "inactive" else s or "Unknown"

    results = [{
        "license_id": r.get("LicenseId", ""),
        "business_name": r.get("BusinessName", ""),
        "contractor_type": r.get("ContractorType", ""),
        "contractor_group": r.get("ContractorGroup", ""),
        "status": _status(r),
        "city": r.get("City", ""),
        "state": r.get("State", ""),
        "ubi": r.get("Ubi", "") or None,
        "violations": (["safety"] if r.get("HasSafetyViolation") else []) +
                      (["contractor"] if r.get("HasContractorViolation") else []),
        "detail_url": f"https://secure.lni.wa.gov/verify/Detail.aspx"
                      f"?LicenseType={urllib.parse.quote(str(r.get('ContractorGroup') or ''))}"
                      f"&LicenseNumber={urllib.parse.quote(str(r.get('LicenseId') or ''))}",
    } for r in data.get("SearchResult", [])]

    if search_cat in ("LicenseId", "Ubi") and total == 1:
        r = results[0]
        return json.dumps({"action": "found", "total_found": total, "results": results,
                           "message": f"{r['business_name']} — {r['contractor_type']} — {r['status']}"})

    if search_cat == "Name":
        exact = [r for r in results if r["business_name"].upper() == query.upper()]
        if len(exact) == 1:
            r = exact[0]
            return json.dumps({"action": "found", "total_found": total, "results": exact,
                               "message": f"{r['business_name']} — {r['contractor_type']} — {r['status']}"})

    return json.dumps({"action": "pick", "total_found": total, "results": results,
                       "message": f"Found {total} matches for '{query}'. Showing {len(results)}. Use a license ID for an exact match."})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

model = InferenceClientModel()
# model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-5")  # or any other provider

agent = CodeAgent(
    tools=[king_county_address_to_parcel, king_county_permit_status, wa_contractor_license],
    model=model,
)

if __name__ == "__main__":
    # End-to-end example: look up permits at an address and verify a contractor
    result = agent.run(
        "For the property at 1817 Morris Ave S, Renton WA: "
        "(1) what is the parcel number, "
        "(2) are there any open or recently-issued permits, "
        "(3) is there an active WA contractor license for 'Mortenson Signs'?"
    )
    print(result)
