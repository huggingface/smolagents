"""Decentralized consensus protocol based on emergent behavior."""

import re
from typing import Dict, List, Optional, Tuple


class ConsensusProtocol:
    def __init__(self):
        self.local_state = {}
        self.confidence_threshold = 0.7
        self.adaptation_rate = 0.2

    def analyze_conversation(self, messages: List[Dict]) -> Tuple[bool, Optional[str], float]:
        """Analyze conversation to detect emerging consensus."""
        # Extract all proposals and their votes
        proposals = {}  # proposal_id -> {content, votes, confidence}

        for msg in messages:
            if msg.get("type") == "final_answer_proposal":
                pid = msg["id"]
                proposals[pid] = {
                    "content": msg["content"],
                    "votes": {"yes": 0, "no": 0},
                    "confidence": 0,
                    "timestamp": msg["timestamp"],
                }
            elif msg.get("type") == "vote":
                content = msg.get("content", {})
                pid = content.get("proposal_id")
                if pid in proposals:
                    vote = content.get("vote")
                    confidence = content.get("confidence", 0)
                    if vote in ("yes", "no"):
                        proposals[pid]["votes"][vote] += 1
                        proposals[pid]["confidence"] = max(proposals[pid]["confidence"], confidence)

        if not proposals:
            return False, None, 0

        # Analyze emerging patterns
        patterns = self._detect_patterns([p["content"] for p in proposals.values()])

        # Look for consensus candidates
        consensus_candidates = []
        for pid, prop in proposals.items():
            votes = prop["votes"]
            total_votes = votes["yes"] + votes["no"]
            if total_votes == 0:
                continue

            agreement = votes["yes"] / total_votes
            if agreement >= 0.5:  # At least 50% agreement
                score = self._calculate_consensus_score(
                    agreement, prop["confidence"], self._matches_pattern(prop["content"], patterns), prop["timestamp"]
                )
                consensus_candidates.append((score, prop["content"]))

        if not consensus_candidates:
            return False, None, 0

        # Return the strongest consensus candidate
        consensus_candidates.sort(reverse=True)
        best_score, content = consensus_candidates[0]

        # Need a minimum consensus strength
        if best_score > self.confidence_threshold:
            return True, content, best_score

        return False, None, best_score

    def _detect_patterns(self, proposals: List[str]) -> List[Dict]:
        """Detect emerging patterns in proposals."""
        patterns = []

        # Number format pattern
        number_formats = []
        for p in proposals:
            numbers = re.findall(r"\d+\.?\d*", p)
            if numbers:
                number_formats.extend(numbers)

        if number_formats:
            # Check if numbers tend to be integers or decimals
            decimals = len([n for n in number_formats if "." in n])
            if decimals > len(number_formats) / 2:
                patterns.append({"type": "number", "format": "decimal"})
            else:
                patterns.append({"type": "number", "format": "integer"})

        # Unit format pattern
        unit_formats = []
        for p in proposals:
            # Try to extract unit part
            parts = p.strip().split(" ")
            if len(parts) > 1:
                unit_formats.append(" ".join(parts[1:]))

        if unit_formats:
            # Find most common unit format
            from collections import Counter

            unit_counts = Counter(unit_formats)
            common_unit = unit_counts.most_common(1)[0][0]
            patterns.append({"type": "unit", "format": common_unit})

        return patterns

    def _matches_pattern(self, content: str, patterns: List[Dict]) -> float:
        """Calculate how well content matches detected patterns."""
        if not patterns:
            return 0.5  # Neutral when no patterns

        matches = 0
        total = len(patterns)

        for pattern in patterns:
            if pattern["type"] == "number":
                numbers = re.findall(r"\d+\.?\d*", content)
                if numbers:
                    num = numbers[0]
                    if pattern["format"] == "decimal" and "." in num:
                        matches += 1
                    elif pattern["format"] == "integer" and "." not in num:
                        matches += 1

            elif pattern["type"] == "unit":
                parts = content.strip().split(" ")
                if len(parts) > 1:
                    unit = " ".join(parts[1:])
                    if unit == pattern["format"]:
                        matches += 1

        return matches / total if total > 0 else 0.5

    def _calculate_consensus_score(
        self, agreement: float, confidence: float, pattern_match: float, timestamp: str
    ) -> float:
        """Calculate overall consensus score combining multiple factors."""
        # Weight recent proposals more heavily
        recency = 1.0  # Could factor in timestamp if needed

        # Combine factors with weights
        weights = {"agreement": 0.4, "confidence": 0.3, "pattern_match": 0.2, "recency": 0.1}

        score = (
            agreement * weights["agreement"]
            + confidence * weights["confidence"]
            + pattern_match * weights["pattern_match"]
            + recency * weights["recency"]
        )

        return score
