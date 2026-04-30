"""Ontology-backed semantic similarity."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path


class GoSimilarity:
    """GO Wang semantic similarity with a small ancestor fallback."""

    def __init__(self, obo_path: str | Path):
        from goatools.obo_parser import GODag

        self.dag = GODag(str(obo_path), optional_attrs={"relationship"})

    @lru_cache(maxsize=50000)
    def similarity(self, term_a: str, term_b: str) -> float:
        if term_a == term_b:
            return 1.0
        if term_a not in self.dag or term_b not in self.dag:
            return 0.0
        if self.dag[term_a].namespace != self.dag[term_b].namespace:
            return 0.0
        try:
            from goatools.semsim.termwise.wang import SsWang

            score = SsWang({term_a, term_b}, self.dag, relationships={"part_of"}).get_sim(
                term_a, term_b
            )
            return float(score or 0.0)
        except Exception:
            return self._ancestor_similarity(term_a, term_b)

    def _ancestor_similarity(self, term_a: str, term_b: str) -> float:
        ancestors_a = self._ancestors(term_a)
        ancestors_b = self._ancestors(term_b)
        if not ancestors_a or not ancestors_b:
            return 0.0
        return len(ancestors_a & ancestors_b) / len(ancestors_a | ancestors_b)

    def _ancestors(self, term_id: str) -> set[str]:
        visited: set[str] = set()
        stack = [term_id]
        while stack:
            current = stack.pop()
            if current in visited or current not in self.dag:
                continue
            visited.add(current)
            stack.extend(parent.id for parent in self.dag[current].parents)
        return visited


class AnatomySimilarity:
    """FlyBase anatomy Wang semantic similarity."""

    def __init__(self, obo_path: str | Path):
        from goatools.obo_parser import GODag

        self.filtered_obo = self._filter_anatomy_obo(Path(obo_path))
        self.dag = GODag(str(self.filtered_obo), optional_attrs={"relationship"})

    @lru_cache(maxsize=50000)
    def similarity(self, term_a: str, term_b: str) -> float:
        if term_a == term_b:
            return 1.0
        if term_a not in self.dag or term_b not in self.dag:
            return 0.0
        try:
            from goatools.semsim.termwise.wang import SsWang

            score = SsWang({term_a, term_b}, self.dag).get_sim(term_a, term_b)
            return float(score or 0.0)
        except Exception:
            return 0.0

    def _filter_anatomy_obo(self, path: Path) -> Path:
        """Remove GCI relationships that create cycles for similarity scoring."""

        lines = ["format-version: 1.2", "ontology: fbbt_filtered", ""]
        current: list[str] = []
        in_term = False
        is_obsolete = False

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "[Term]":
                if in_term and current and not is_obsolete and any("id: FBbt:" in x for x in current):
                    lines.extend(current)
                    lines.append("")
                current = ["[Term]"]
                in_term = True
                is_obsolete = False
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                if in_term and current and not is_obsolete and any("id: FBbt:" in x for x in current):
                    lines.extend(current)
                    lines.append("")
                current = []
                in_term = False
                is_obsolete = False
                continue
            if not in_term:
                continue
            if stripped.startswith("is_obsolete: true"):
                is_obsolete = True
            elif (
                stripped.startswith("id: FBbt:")
                or stripped.startswith("name:")
                or stripped.startswith("namespace:")
                or (stripped.startswith("is_a: FBbt:") and "{" not in stripped)
            ):
                current.append(line)

        if in_term and current and not is_obsolete and any("id: FBbt:" in x for x in current):
            lines.extend(current)
            lines.append("")

        tmp = tempfile.NamedTemporaryFile("w", suffix=".obo", delete=False, prefix="fbbt_filtered_")
        tmp.write("\n".join(lines))
        tmp.close()
        return Path(tmp.name)
