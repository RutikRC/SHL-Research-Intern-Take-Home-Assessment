"""
Generates deterministic comparison text between two or more SHL assessments.

Uses only catalog metadata — no LLM calls.
"""

from __future__ import annotations

from app.core.logging_ import get_logger
from app.database.models import Assessment

logger = get_logger(__name__)


class ComparisonService:
    """Generates a structured, deterministic comparison between assessments.

    Compares: description, categories, job levels, languages, remote,
    adaptive, and duration. Returns a plain-text comparison table.
    """

    async def compare(
        self,
        assessments: list[Assessment],
    ) -> str:
        """Build a deterministic comparison string for a list of assessments.

        Args:
            assessments: List of Assessment objects to compare.

        Returns:
            A formatted comparison string comparing all metadata fields.
        """
        if not assessments:
            return "No assessments found to compare."

        if len(assessments) == 1:
            return self._single_description(assessments[0])

        return self._multi_comparison(assessments)

    @staticmethod
    def _single_description(assessment: Assessment) -> str:
        """Generate a description for a single assessment."""
        lines: list[str] = []
        lines.append(f"Overview of {assessment.name}:")
        lines.append("")
        if assessment.description:
            lines.append(f"Description: {assessment.description}")
        lines.append(f"Categories: {', '.join(assessment.keys) if assessment.keys else 'General'}")
        lines.append(f"Job Levels: {', '.join(assessment.job_levels) if assessment.job_levels else 'All levels'}")
        lines.append(f"Languages: {', '.join(assessment.languages) if assessment.languages else 'English'}")
        lines.append(f"Remote Testing: {'Yes' if assessment.remote else 'No'}")
        lines.append(f"Adaptive/IRT: {'Yes' if assessment.adaptive else 'No'}")
        lines.append(f"Duration: {assessment.duration or 'Variable'}")
        return "\n".join(lines)

    @staticmethod
    def _multi_comparison(assessments: list[Assessment]) -> str:
        """Generate a structured comparison of multiple assessments."""
        lines: list[str] = []
        lines.append("Here is a comparison of the requested assessments:\n")

        header = f"{'Feature':<25}"
        for a in assessments:
            header += f" | {a.name[:30]:<30}"
        lines.append(header)
        lines.append("-" * (25 + 33 * len(assessments)))

        # Description (first 120 chars)
        desc = f"{'Description':<25}"
        for a in assessments:
            short_desc = (a.description[:117] + "...") if len(a.description) > 120 else a.description
            desc += f" | {short_desc:<30}"
        lines.append(desc)

        # Categories
        cats = f"{'Categories':<25}"
        for a in assessments:
            cat_str = ", ".join(a.keys)[:28] if a.keys else "General"
            cats += f" | {cat_str:<30}"
        lines.append(cats)

        # Job Levels
        jls = f"{'Job Levels':<25}"
        for a in assessments:
            jl_str = ", ".join(a.job_levels)[:28] if a.job_levels else "All levels"
            jls += f" | {jl_str:<30}"
        lines.append(jls)

        # Languages
        langs = f"{'Languages':<25}"
        for a in assessments:
            lang_str = ", ".join(a.languages)[:28] if a.languages else "English"
            langs += f" | {lang_str:<30}"
        lines.append(langs)

        # Remote
        rem = f"{'Remote Testing':<25}"
        for a in assessments:
            rem += f" | {'Yes' if a.remote else 'No':<30}"
        lines.append(rem)

        # Adaptive
        adp = f"{'Adaptive/IRT':<25}"
        for a in assessments:
            adp += f" | {'Yes' if a.adaptive else 'No':<30}"
        lines.append(adp)

        # Duration
        dur = f"{'Duration':<25}"
        for a in assessments:
            dur += f" | {(a.duration or 'Variable'):<30}"
        lines.append(dur)

        return "\n".join(lines)