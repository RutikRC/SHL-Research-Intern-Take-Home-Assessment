"""
Formats an Assessment ORM object into a semantic text document for embedding.

The generated document is a human-readable, structured text representation
that captures the assessment's key attributes. This improves semantic search
quality compared to embedding raw JSON.
"""

from __future__ import annotations

from app.database.models import Assessment


class EmbeddingDocumentFormatter:
    """Converts an Assessment into a semantic text document for embedding."""

    def format(self, assessment: Assessment) -> str:
        """Generate a semantic text document from an assessment.

        Args:
            assessment: The Assessment ORM object to format.

        Returns:
            A structured text string suitable for embedding.
        """
        lines: list[str] = []

        lines.append(f"Assessment Name:\n{assessment.name}")
        lines.append("")

        if assessment.description:
            lines.append(f"Description:\n{assessment.description}")
            lines.append("")

        if assessment.job_levels:
            lines.append("Job Levels:")
            for level in assessment.job_levels:
                lines.append(f"  {level}")
            lines.append("")

        if assessment.keys:
            lines.append("Categories:")
            for key in assessment.keys:
                lines.append(f"  {key}")
            lines.append("")

        if assessment.languages:
            lines.append("Languages:")
            for lang in assessment.languages:
                lines.append(f"  {lang}")
            lines.append("")

        lines.append(f"Remote Testing:\n{'Yes' if assessment.remote else 'No'}")
        lines.append("")

        lines.append(f"Adaptive/IRT:\n{'Yes' if assessment.adaptive else 'No'}")
        lines.append("")

        if assessment.duration:
            lines.append(f"Duration:\n{assessment.duration}")

        return "\n".join(lines).strip()