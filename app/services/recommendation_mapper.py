"""
Maps domain Assessment ORM objects to API Recommendation models.

Isolates the internal domain model from the public API schema so that
changes to the database model never leak into the API contract.
"""

from __future__ import annotations

from app.database.models import Assessment
from app.models.response import Recommendation


class RecommendationMapper:
    """Converts Assessment ORM objects into Recommendation API models.

    The mapping is:
        Assessment.name       → Recommendation.name
        Assessment.url        → Recommendation.url
        Assessment.keys[0]    → Recommendation.test_type (or "General" if empty)
    """

    def map_one(self, assessment: Assessment) -> Recommendation:
        """Convert a single Assessment into a Recommendation.

        Args:
            assessment: The Assessment ORM object to convert.

        Returns:
            A Recommendation Pydantic model populated from the assessment.
        """
        # Derive test_type from the first key/category if available
        test_type = "General"
        if assessment.keys and len(assessment.keys) > 0:
            test_type = str(assessment.keys[0])

        return Recommendation(
            name=assessment.name,
            url=assessment.url,
            test_type=test_type,
        )

    def map_many(self, assessments: list[Assessment]) -> list[Recommendation]:
        """Convert a list of Assessment objects into Recommendations.

        Args:
            assessments: The list of Assessment ORM objects.

        Returns:
            A list of Recommendation Pydantic models.
        """
        return [self.map_one(a) for a in assessments]