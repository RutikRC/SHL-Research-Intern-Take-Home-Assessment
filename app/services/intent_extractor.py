"""
Extracts a structured HiringIntent from a parsed conversation.

Uses deterministic rules and lightweight NLP patterns — no LLM calls.
The intent is the single source of truth for all downstream phases.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logging_ import get_logger
from app.models.intent import HiringIntent, JobLevelEnum, SkillArea
from app.services.conversation_parser import ParsedConversation

logger = get_logger(__name__)


class IntentExtractor:
    """Extracts structured hiring intent from conversation history.

    Uses keyword patterns, role detection, skill extraction, and context
    merging to produce a HiringIntent without calling any LLM.
    """

    # ── Patterns ─────────────────────────────────────────────────────────────

    # Refinement indicators
    REFINEMENT_PATTERNS = re.compile(
        r"\b(actually|instead|change|also\s+need|additionally|"
        r"remove|replace|add\s+on\s+top|another|different|"
        r"what\s+else|anything\s+else|more\s+like|"
        r"forget|ignore\s+previous|not\s+that|but\s+also|"
        r"in\s+addition|plus|also\s+want|also\s+require)\b",
        re.IGNORECASE,
    )

    # Comparison indicators
    COMPARISON_PATTERNS = re.compile(
        r"\b(compare|difference|versus|vs\.?|better|"
        r"which\s+one|how\s+does|similarities|"
        r"contrast|between\s+.+\s+and)\b",
        re.IGNORECASE,
    )

    # Comparison target extraction
    COMPARISON_TARGET_PATTERNS = re.compile(
        r"(?:compare|difference\s+between|vs\.?|versus)\s+"
        r"['\"]?([A-Za-z0-9\s&]+?)['\"]?\s*(?:and|vs\.?|versus|with)\s*"
        r"['\"]?([A-Za-z0-9\s&]+?)['\"]?",
        re.IGNORECASE,
    )

    # Off-topic detection patterns
    OFF_TOPIC_PATTERNS = re.compile(
        r"\b(salary\s*(negotiation|expectation|range)?|"
        r"interview\s*(questions|tips|preparation|advice)?|"
        r"legal\s*advice|resume\s*(review|writing|tips)?|"
        r"cover\s*letter|career\s*advice|career\s*counseling|"
        r"politics|election|medical\s*advice|diagnosis|"
        r"investment|stock|trading|tax\s*advice|"
        r"how\s+to\s+(get|ace|pass)\s+(a\s+)?interview|"
        r"negotiate|salary\s+increase|promotion)\b",
        re.IGNORECASE,
    )

    # Prompt injection patterns
    INJECTION_PATTERNS = re.compile(
        r"\b(ignore\s+(all\s+)?(previous|above|prior)\s+instructions|"
        r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|context)|"
        r"pretend\s+(you\s+are|to\s+be)|"
        r"you\s+are\s+now|"
        r"override\s+(your\s+)?(instructions|rules|prompt)|"
        r"new\s+instructions|"
        r"act\s+as\s+(if|though)|"
        r"disregard\s+(all\s+)?(previous|above)|"
        r"system\s+prompt|"
        r"jailbreak|"
        r"dan\s+|do\s+anything\s+now|"
        r"you\s+are\s+not\s+bound|"
        r"ignore\s+the\s+catalog)\b",
        re.IGNORECASE,
    )

    # Role detection keywords (common job titles)
    ROLE_PATTERNS = {
        "java developer": [r"\bjava\s*(developer|engineer|programmer|dev)\b"],
        "python developer": [r"\bpython\s*(developer|engineer|programmer)\b"],
        "data scientist": [r"\bdata\s*scientist\b"],
        "data analyst": [r"\bdata\s*analyst\b"],
        "software engineer": [r"\bsoftware\s*(engineer|developer)\b"],
        "frontend developer": [r"\bfront\s*end\s*(developer|engineer)\b"],
        "backend developer": [r"\bback\s*end\s*(developer|engineer)\b"],
        "full stack developer": [r"\bfull\s*stack\s*(developer|engineer)\b"],
        "devops engineer": [r"\bdevops\s*(engineer|developer)\b"],
        "product manager": [r"\bproduct\s*manager\b"],
        "project manager": [r"\bproject\s*manager\b"],
        "sales representative": [r"\bsales\s*(representative|associate|manager)\b"],
        "customer service": [r"\bcustomer\s*service\b"],
        "call center": [r"\bcall\s*center\b"],
        "manager": [r"\bmanager\b"],
        "executive": [r"\bexecutive\b|\bceo\b|\bcto\b|\bcfo\b"],
        "director": [r"\bdirector\b"],
    }

    # Skill detection patterns
    SKILL_PATTERNS = [
        (r"\b(java|j2ee|jee)\b", "Java"),
        (r"\bpython\b", "Python"),
        (r"\bjavascript\b", "JavaScript"),
        (r"\btypescript\b", "TypeScript"),
        (r"\breact\b", "React"),
        (r"\bangular\b", "Angular"),
        (r"\bvue\b", "Vue.js"),
        (r"\bnode\.?(js)?\b", "Node.js"),
        (r"\bspring\b", "Spring"),
        (r"\bdjango\b", "Django"),
        (r"\bflask\b", "Flask"),
        (r"\bsql\b", "SQL"),
        (r"\b(mongodb|nosql)\b", "MongoDB"),
        (r"\baws\b", "AWS"),
        (r"\b(azure|ms\s*azure)\b", "Azure"),
        (r"\bgcp|google\s*cloud\b", "GCP"),
        (r"\bdocker\b", "Docker"),
        (r"\bkubernetes\b", "Kubernetes"),
        (r"\bci/cd\b", "CI/CD"),
        (r"\bgcp\b", "GCP"),
        (r"\bc#|csharp\b", "C#"),
        (r"\bc\+\+\b", "C++"),
        (r"\bgolang|go\s*lang\b", "Go"),
        (r"\brust\b", "Rust"),
        (r"\bkotlin\b", "Kotlin"),
        (r"\bswift\b", "Swift"),
        (r"\bphp\b", "PHP"),
        (r"\bruby\b", "Ruby"),
        (r"\bscala\b", "Scala"),
        (r"\br\s*(programming|language)?\b", "R"),
        (r"\b\.net\b", ".NET"),
        (r"\bhtml\b", "HTML"),
        (r"\bcss\b", "CSS"),
        (r"\bsql\b", "SQL"),
        (r"\b(machine\s*learning|ml|ai)\b", "Machine Learning"),
        (r"\bdeep\s*learning\b", "Deep Learning"),
        (r"\bnlp\b", "NLP"),
        (r"\bdata\s*engineering\b", "Data Engineering"),
        (r"\bdata\s*science\b", "Data Science"),
        (r"\b(statistics|statistical)\b", "Statistics"),
        (r"\bleadership\b", "Leadership"),
        (r"\bcommunication\b", "Communication"),
        (r"\bstakeholder\s*(management|communication)?\b", "Stakeholder Management"),
        (r"\bproject\s*management\b", "Project Management"),
        (r"\bagile\b", "Agile"),
        (r"\bscrum\b", "Scrum"),
        (r"\bsales\b", "Sales"),
        (r"\bcustomer\s*service\b", "Customer Service"),
        (r"\b(support|helpdesk)\b", "Technical Support"),
        (r"\banalytics?\b", "Analytics"),
        (r"\bproblem\s*solving\b", "Problem Solving"),
        (r"\bteamwork\b", "Teamwork"),
        (r"\bcollaboration\b", "Collaboration"),
        (r"\baccounting\b", "Accounting"),
        (r"\bfinance\b", "Finance"),
        (r"\bmarketing\b", "Marketing"),
        (r"\bhr\b", "HR"),
        (r"\brecruiting\b", "Recruiting"),
    ]

    # Experience level patterns
    EXPERIENCE_PATTERNS = [
        (r"\bentry\s*level\b", "Entry-Level"),
        (r"\bjunior\b", "Junior"),
        (r"\bmid\s*level\b|\bmid\b[\s-]*professional", "Mid-Level"),
        (r"\bsenior\b", "Senior"),
        (r"\blead\b", "Lead"),
        (r"\bprincipal\b", "Principal"),
        (r"\bfresher\b|\bgraduate\b", "Graduate/Fresher"),
        (r"\b(\d+)\+?\s*(years?\s*)?(of\s*)?experience\b", None),  # Dynamic
    ]

    # Job level mapping
    JOB_LEVEL_MAP: dict[str, JobLevelEnum] = {
        "entry": JobLevelEnum.ENTRY_LEVEL,
        "junior": JobLevelEnum.ENTRY_LEVEL,
        "fresher": JobLevelEnum.ENTRY_LEVEL,
        "graduate": JobLevelEnum.GRADUATE,
        "mid": JobLevelEnum.MID_PROFESSIONAL,
        "senior": JobLevelEnum.PROFESSIONAL,
        "lead": JobLevelEnum.MANAGER,
        "principal": JobLevelEnum.MANAGER,
        "manager": JobLevelEnum.MANAGER,
        "director": JobLevelEnum.DIRECTOR,
        "executive": JobLevelEnum.EXECUTIVE,
    }

    # Assessment category detection
    CATEGORY_PATTERNS: list[tuple[re.Pattern, SkillArea]] = [
        (re.compile(r"\b(technical|programming|coding|software|development|engineering|it)\b", re.IGNORECASE), SkillArea.TECHNICAL),
        (re.compile(r"\b(personality|behavior|behaviour|opq|work\s*style)\b", re.IGNORECASE), SkillArea.PERSONALITY),
        (re.compile(r"\b(cognitive|ability|aptitude|reasoning|numerical|verbal|abstract)\b", re.IGNORECASE), SkillArea.COGNITIVE),
        (re.compile(r"\b(knowledge|skill|domain|subject)\b", re.IGNORECASE), SkillArea.KNOWLEDGE),
        (re.compile(r"\b(simulation|simulated|role[\s-]play|interactive)\b", re.IGNORECASE), SkillArea.SIMULATION),
        (re.compile(r"\b(competenc|behavioral|behavioural|job\s*fitness)\b", re.IGNORECASE), SkillArea.COMPETENCY),
        (re.compile(r"\b(biodata|situational|scenario|judgment)\b", re.IGNORECASE), SkillArea.BIODATA),
        (re.compile(r"\b(development|360|feedback|coaching)\b", re.IGNORECASE), SkillArea.DEVELOPMENT),
        (re.compile(r"\b(exercise|assessment\s*center|ac)\b", re.IGNORECASE), SkillArea.ASSESSMENT_EXERCISES),
    ]

    # Clarification indicators
    CLARIFICATION_VAGUE_PATTERNS = re.compile(
        r"^(i\s+need\s+(an?\s+)?(assessment|test)|"
        r"i\s+want\s+(an?\s+)?(assessment|test)|"
        r"looking\s+for\s+(an?\s+)?(assessment|test)|"
        r"hire|hiring|recruit|job)$",
        re.IGNORECASE,
    )

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(self, parsed: ParsedConversation) -> HiringIntent:
        """Extract a structured HiringIntent from a parsed conversation.

        Args:
            parsed: A validated ParsedConversation.

        Returns:
            A HiringIntent with all detected fields populated.
        """
        latest = parsed.latest_user_message

        intent = HiringIntent(
            raw_user_query=latest,
            message_count=parsed.message_count,
            turn_count=parsed.turn_count,
        )

        # Safety checks first
        if self._is_prompt_injection(latest):
            intent.prompt_injection_detected = True
            logger.warning("prompt_injection_detected", query_preview=latest[:100])
            return intent

        if self._is_off_topic(latest):
            intent.off_topic = True
            intent.off_topic_reason = "Query appears unrelated to SHL assessment selection."
            logger.warning("off_topic_detected", query_preview=latest[:100])
            return intent

        # Core intent extraction from ALL user messages (for constraint merging)
        all_user_text = " ".join(parsed.user_messages)

        # Detect role
        intent.role = self._detect_role(all_user_text)

        # Detect skills
        intent.skills = self._detect_skills(all_user_text)

        # Detect experience
        intent.experience = self._detect_experience(all_user_text)
        intent.job_level = self._detect_job_level(all_user_text, intent.experience)

        # Detect assessment categories
        intent.assessment_categories = self._detect_assessment_categories(all_user_text)

        # Detect languages
        intent.languages = self._detect_languages(all_user_text)

        # Detect remote and adaptive
        intent.remote_required = self._detect_remote(all_user_text)
        intent.adaptive_required = self._detect_adaptive(all_user_text)

        # Detect refinement
        intent.refinement_requested = self._detect_refinement(parsed)

        # Detect comparison
        comparison_result = self._detect_comparison(latest)
        intent.comparison_requested = comparison_result["detected"]
        intent.comparison_targets = comparison_result["targets"]

        # Detect clarification need
        intent.clarification_needed = self._detect_clarification_needed(parsed)

        # Merge constraints from previous turns if refinement detected
        if intent.refinement_requested and len(parsed.user_messages) > 1:
            intent = self._merge_constraints(intent, parsed)

        logger.info(
            "intent_extracted",
            role=intent.role or "none",
            skills=len(intent.skills),
            categories=[c.value for c in intent.assessment_categories],
            clarification_needed=intent.clarification_needed,
            refinement=intent.refinement_requested,
            comparison=intent.comparison_requested,
            off_topic=intent.off_topic,
            injection=intent.prompt_injection_detected,
        )

        return intent

    # ── Detection methods ────────────────────────────────────────────────────

    @staticmethod
    def _detect_role(text: str) -> str:
        """Detect the target role from user text."""
        for role_name, patterns in IntentExtractor.ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return role_name.title()

        # Broader role detection — look for "hiring a <role>" or "<role> developer" etc.
        broad_match = re.search(
            r"(?:hiring|looking\s+for|need|want|require)\s+(?:a|an|some)?\s*"
            r"['\"]?([A-Za-z\s]+?)['\"]?\s*(?:developer|engineer|architect|lead|"
            r"manager|specialist|analyst|scientist|consultant|associate|officer)?",
            text, re.IGNORECASE,
        )
        if broad_match:
            candidate = broad_match.group(1).strip().title()
            if len(candidate) < 100:
                return candidate

        return ""

    @staticmethod
    def _detect_skills(text: str) -> list[str]:
        """Detect skills mentioned in user text."""
        skills: list[str] = []
        seen: set[str] = set()
        normalized = text.lower()

        for pattern, skill_name in IntentExtractor.SKILL_PATTERNS:
            if re.search(pattern, normalized):
                if skill_name not in seen:
                    skills.append(skill_name)
                    seen.add(skill_name)

        return skills

    @staticmethod
    def _detect_experience(text: str) -> str:
        """Detect experience level from user text."""
        for pattern, label in IntentExtractor.EXPERIENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if label:
                    return label
                # Dynamic years-of-experience extraction
                years = match.group(1)
                return f"{years}+ Years"

        return ""

    @staticmethod
    def _detect_job_level(text: str, experience: str) -> JobLevelEnum:
        """Map detected experience to a JobLevelEnum."""
        text_lower = text.lower()

        for keyword, level in IntentExtractor.JOB_LEVEL_MAP.items():
            if keyword in text_lower:
                return level

        # Fall back to experience-based mapping
        if "entry" in experience.lower() or "junior" in experience.lower() or "fresher" in experience.lower():
            return JobLevelEnum.ENTRY_LEVEL
        if "graduate" in experience.lower():
            return JobLevelEnum.GRADUATE
        if "mid" in experience.lower():
            return JobLevelEnum.MID_PROFESSIONAL
        if "senior" in experience.lower() or "lead" in experience.lower() or "principal" in experience.lower():
            return JobLevelEnum.PROFESSIONAL

        return JobLevelEnum.UNKNOWN

    @staticmethod
    def _detect_assessment_categories(text: str) -> list[SkillArea]:
        """Detect requested assessment categories from user text."""
        categories: list[SkillArea] = []
        seen: set[SkillArea] = set()

        for pattern, area in IntentExtractor.CATEGORY_PATTERNS:
            if pattern.search(text):
                if area not in seen:
                    categories.append(area)
                    seen.add(area)

        return categories

    @staticmethod
    def _detect_languages(text: str) -> list[str]:
        """Detect language requirements from user text."""
        languages: list[str] = []
        lang_patterns = {
            "English": r"\benglish\b",
            "Spanish": r"\bspanish\b",
            "French": r"\bfrench\b",
            "German": r"\bgerman\b",
            "Chinese": r"\b(mandarin|chinese)\b",
            "Japanese": r"\bjapanese\b",
            "Portuguese": r"\bportuguese\b",
        }
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                languages.append(lang)
        return languages

    @staticmethod
    def _detect_remote(text: str) -> bool | None:
        """Detect remote testing requirement."""
        if re.search(r"\bremote\b", text, re.IGNORECASE):
            return True
        if re.search(r"\bon[\s-]?site\b", text, re.IGNORECASE):
            return False
        return None

    @staticmethod
    def _detect_adaptive(text: str) -> bool | None:
        """Detect adaptive/IRT testing requirement."""
        if re.search(r"\b(adaptive|irt)\b", text, re.IGNORECASE):
            return True
        return None

    @staticmethod
    def _detect_refinement(parsed: ParsedConversation) -> bool:
        """Detect whether the latest message is a refinement of previous intent."""
        if parsed.turn_count <= 1:
            return False
        latest = parsed.latest_user_message
        return bool(IntentExtractor.REFINEMENT_PATTERNS.search(latest))

    @staticmethod
    def _detect_comparison(text: str) -> dict[str, Any]:
        """Detect comparison requests and extract target assessments."""
        result: dict[str, Any] = {"detected": False, "targets": []}

        if not IntentExtractor.COMPARISON_PATTERNS.search(text):
            return result

        result["detected"] = True

        # Extract comparison targets
        match = IntentExtractor.COMPARISON_TARGET_PATTERNS.search(text)
        if match:
            target1 = match.group(1).strip()
            target2 = match.group(2).strip()
            if target1:
                result["targets"].append(target1)
            if target2:
                result["targets"].append(target2)

        return result

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        """Check if the user query is off-topic for SHL assessment selection."""
        return bool(IntentExtractor.OFF_TOPIC_PATTERNS.search(text))

    @staticmethod
    def _is_prompt_injection(text: str) -> bool:
        """Check if the user query contains prompt injection attempts."""
        return bool(IntentExtractor.INJECTION_PATTERNS.search(text))

    @staticmethod
    def _detect_clarification_needed(parsed: ParsedConversation) -> bool:
        """Determine if more information is needed before retrieval.

        Returns True if the query is too vague (e.g., "I need an assessment").
        """
        latest = parsed.latest_user_message.strip().lower()

        # If already on a later turn with assistant having provided info, clarification not needed
        if parsed.turn_count > 1:
            return False

        # Check if query is too vague
        if IntentExtractor.CLARIFICATION_VAGUE_PATTERNS.match(latest):
            return True

        # If query has fewer than 3 meaningful words, clarification may be needed
        words = [w for w in latest.split() if len(w) > 2]
        if len(words) < 2:
            return True

        # If we detected a role, we have enough to start retrieval
        role = IntentExtractor._detect_role(latest)
        if role:
            return False

        # If we detected specific skills, we have enough
        skills = IntentExtractor._detect_skills(latest)
        if skills:
            return False

        # Generic fallback
        return True

    @staticmethod
    def _merge_constraints(intent: HiringIntent, parsed: ParsedConversation) -> HiringIntent:
        """Merge constraints from previous turns into the current intent.

        When refinement is detected, accumulate constraints from all user
        messages rather than only using the latest.
        """
        if len(parsed.user_messages) < 2:
            return intent

        # Collect all user text except the latest (that's the refinement)
        previous_text = " ".join(parsed.user_messages[:-1])

        # Save current state as "previous"
        intent.previous_role = intent.role
        intent.previous_skills = list(intent.skills)
        intent.previous_categories = list(intent.assessment_categories)

        # Merge skills from previous turns
        prev_skills = IntentExtractor._detect_skills(previous_text)
        for skill in prev_skills:
            if skill not in intent.skills:
                intent.skills.append(skill)

        # Merge assessment categories from previous turns
        prev_categories = IntentExtractor._detect_assessment_categories(previous_text)
        for cat in prev_categories:
            if cat not in intent.assessment_categories:
                intent.assessment_categories.append(cat)

        # Use previous role if current is empty
        if not intent.role:
            prev_role = IntentExtractor._detect_role(previous_text)
            if prev_role:
                intent.role = prev_role

        return intent