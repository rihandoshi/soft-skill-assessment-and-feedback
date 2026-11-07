"""Stage 0, step 1: hand-labelled question types for the 76 RecruitView questions.

Writes Datasets/question_types.csv with columns: question_id, question, question_type.

Label definitions
-----------------
intro            "Introduce yourself" / "Tell me about yourself": open self-presentation.
behavioural      Asks for a concrete past episode ("Tell me about a time...", "Give an example...",
                 "Describe a situation/project...", "Share a story...", biggest achievement/failure,
                 toughest decision). Expected answer: a narrative (situation -> action -> result).
self_assessment  Asks the candidate to characterise their own traits, skills or habits in general
                 ("How do you handle stress?", "Are you reliable?", "Rate yourself", strengths/weaknesses,
                 "Why should we hire you?"). Expected answer: claim + justification.
opinion          Values, preferences, motivations, aspirations or hypotheticals ("What motivates you?",
                 "Dream company", "Where do you see yourself in 5 years?", lottery question, hobbies).

The labels are a judgement call; edit LABELS below and re-run if the team disagrees with any.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "Datasets" / "RecruitView" / "metadata.jsonl"
OUTPUT = ROOT / "Datasets" / "question_types.csv"

I, B, S, O = "intro", "behavioural", "self_assessment", "opinion"

LABELS: dict[int, str] = {
    1: I,   # Introduce yourself
    2: S,   # greatest strengths and weaknesses
    3: S,   # handle changes or unexpected situations
    4: B,   # biggest achievement so far
    5: B,   # time you went above and beyond
    6: B,   # example of your creativity
    7: S,   # work under pressure
    8: O,   # Rs.10-crore lottery
    9: O,   # what motivates you
    10: B,  # implemented a solution to improve a process
    11: B,  # stepped into a role outside expertise
    12: S,  # how do you respond to change
    13: B,  # toughest decision
    14: O,  # greatest fear
    15: B,  # led a new strategy/process
    16: B,  # identified a problem early
    17: S,  # rate yourself 1-10
    18: S,  # handle stress and anxiety
    19: B,  # not satisfied with your performance
    20: B,  # managed competing priorities
    21: O,  # where in 5 years
    22: S,  # why should a company hire you
    23: S,  # are you reliable
    24: O,  # what makes you angry
    25: B,  # persuaded others
    26: B,  # took initiative without being asked
    27: S,  # open to risks / experimenting
    28: O,  # dream company
    29: S,  # good work ethic
    30: B,  # capitalised on team members' strengths
    31: B,  # resolved a long-standing issue
    32: S,  # how do you improve your knowledge
    33: O,  # hobbies
    34: B,  # led by example
    35: B,  # encouraged open communication
    36: S,  # what makes you different
    37: S,  # time management skills
    38: B,  # overcame a challenge in a team
    39: S,  # ensure task completed effectively
    40: B,  # demonstrated leadership
    41: O,  # three things most important in a job
    42: B,  # how did you handle disagreements (past)
    43: B,  # difficulty while working on a project
    44: O,  # what makes you happy
    45: B,  # mentored a junior colleague
    46: O,  # passionate about
    47: O,  # motivates you at work
    48: B,  # sought opportunities to develop skills
    49: B,  # leveraged technology/automation
    50: B,  # cross-functional collaboration
    51: B,  # adapted to a change in work environment
    52: B,  # most proud of
    53: O,  # ideal work environment
    54: B,  # project you initiated or led
    55: B,  # resolved a disagreement in a team
    56: S,  # deal with criticism
    57: B,  # failed to meet a goal
    58: B,  # greatest failure
    59: B,  # conflict with a coworker
    60: B,  # led team to cost savings/revenue
    61: B,  # project outside comfort zone
    62: S,  # are you organised
    63: O,  # regrets
    64: B,  # turned a struggling project around
    65: B,  # motivated a disengaged team member
    66: S,  # how do you learn new skills
    67: B,  # prioritised under tight deadlines
    68: S,  # adapt to new technology
    69: B,  # facilitated a team meeting
    70: B,  # collaborated with stakeholders
    71: O,  # dream job
    72: S,  # weaknesses
    73: B,  # proposed an innovative idea
    74: B,  # coached or mentored a colleague
    75: B,  # rallied team during a crisis
    76: I,  # Tell me about yourself
}


def main() -> None:
    questions: dict[int, str] = {}
    with METADATA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            questions.setdefault(int(row["question_id"]), row["question"])
    missing = sorted(set(questions) - set(LABELS))
    extra = sorted(set(LABELS) - set(questions))
    if missing or extra:
        raise SystemExit(f"Label mismatch. Unlabelled ids: {missing}; unknown ids: {extra}")
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "question", "question_type"])
        for qid in sorted(questions):
            writer.writerow([qid, questions[qid], LABELS[qid]])
    counts: dict[str, int] = {}
    for label in LABELS.values():
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {OUTPUT} ({len(questions)} questions): {counts}")


if __name__ == "__main__":
    main()
