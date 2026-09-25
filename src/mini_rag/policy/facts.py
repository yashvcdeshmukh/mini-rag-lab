from __future__ import annotations

from mini_rag.policy.models import PolicyChunk, PolicyFact, section_covers

# Figures copied from the source policies. Ingest will not invent these.
CHECKED_FACTS: tuple[PolicyFact, ...] = (
    PolicyFact("hr", "1.0", "pet_leave_days", "5", "5"),
    PolicyFact("hr", "1.0", "pet_leave_days_dog", "7", "5"),
    PolicyFact("hr", "1.0", "boss_error_grace_minutes", "30", "6"),
    PolicyFact("hr", "2.0", "pet_leave_days", "5", "5"),
    PolicyFact("hr", "2.0", "pet_leave_days_dog", "7", "5"),
    PolicyFact("hr", "2.0", "boss_error_grace_minutes", "30", "6"),
    PolicyFact("time-usage", "1.0", "video_game_minutes", "45", "3"),
    PolicyFact("time-usage", "1.0", "foosball_minutes", "20", "4"),
    PolicyFact("time-usage", "1.0", "token_allocation", "1000000", "5"),
    PolicyFact("time-usage", "2.0", "video_game_minutes", "45", "3"),
    PolicyFact("time-usage", "2.0", "foosball_minutes", "20", "4"),
    PolicyFact("time-usage", "2.0", "token_allocation", "500000", "6"),
    PolicyFact("preparedness", "1.0", "nuclear_shelter_hours", "2", "4"),
    PolicyFact("preparedness", "2.0", "nuclear_shelter_weeks", "2", "4"),
    PolicyFact("health-wellness", "1.0", "gym_sessions_per_week", "3", "3"),
    PolicyFact("health-wellness", "1.0", "caffeine_mg_per_day", "400", "5"),
)

CHOICE_CRITERIA: dict[str, str] = {
    "pet_leave_days": "Days of pet leave for a pet other than a dog",
    "pet_leave_days_dog": "Days of pet leave for a dog",
    "boss_error_grace_minutes": "Minutes of grace when the boss makes an error",
    "video_game_minutes": "Minutes allowed for video games",
    "foosball_minutes": "Minutes allowed for foosball",
    "token_allocation": "How many tokens an employee receives each cycle",
    "nuclear_shelter_hours": "Hours allowed to reach the nuclear shelter",
    "nuclear_shelter_weeks": "Weeks of supplies stored for the nuclear shelter",
    "gym_sessions_per_week": "Required gym sessions each week",
    "caffeine_mg_per_day": "Daily caffeine limit in milligrams",
    "none": "The question is not a lookup of one checked figure",
    "current": "The question selects the latest edition over the earlier one",
    "previous": "The question selects the earlier edition over the latest one",
    "unspecified": "The question does not select one edition over the other",
    "yes": "The excerpts contain enough specific information to answer fully",
    "no": "More sections are needed to answer the question",
}


def validate_facts(
    facts: tuple[PolicyFact, ...] | list[PolicyFact],
    chunks: list[PolicyChunk],
) -> None:
    missing = [
        fact
        for fact in facts
        if not any(
            chunk.document_slug == fact.document_slug
            and chunk.version == fact.version
            and section_covers(chunk.section, fact.section)
            for chunk in chunks
        )
    ]
    if not missing:
        return
    described = ", ".join(
        f"{fact.document_slug} v{fact.version} {fact.fact_key} section {fact.section}"
        for fact in missing
    )
    raise ValueError(f"Fact section was not chunked: {described}")
