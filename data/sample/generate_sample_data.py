"""
generate_sample_data.py
------------------------
Generates a small, offline BBC-style sample news dataset so that the NewsBot
notebooks, tests, and Django app all run end-to-end WITHOUT requiring a
Kaggle download.

This is a *fallback/demo* dataset only. For the real assignment dataset,
download the BBC News (learn-ai-bbc) dataset from Kaggle and place
'BBC News Train.csv' in data/raw/ -- the DataLoader in
src/data_processing prefers the real file automatically when present.

Run: python generate_sample_data.py
Produces: sample_news.csv (5 categories x 40 articles = 200 rows)
"""
import random
import csv
import os

random.seed(42)

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

TEMPLATES = {
    "business": [
        "{company} reported {change} in quarterly profits, citing {reason}. Analysts said the results {reaction} expectations for the {sector} sector.",
        "Shares of {company} {movement} after the firm announced {event}. The {sector} sector has been under pressure amid {macro}.",
        "The central bank held interest rates steady, saying inflation pressures tied to {macro} remain a key concern for the {sector} industry.",
        "{company} announced a merger with a rival firm worth {amount}, aiming to expand its presence in the {sector} market.",
        "Unemployment figures released this week showed {change}, with economists pointing to {reason} as a major factor.",
    ],
    "entertainment": [
        "The film starring {celebrity} topped the box office this weekend, earning {amount} in its opening frame.",
        "{celebrity} announced a new album set for release later this year, describing the project as {desc}.",
        "Critics gave mixed reviews to the new series, praising its {desc} while questioning the pacing of later episodes.",
        "The awards ceremony saw {celebrity} take home the top honor, in a night dominated by {desc} performances.",
        "A biopic about {celebrity} is currently in production, with filming expected to wrap by the end of the year.",
    ],
    "politics": [
        "The government unveiled a new policy on {issue}, drawing {reaction} from opposition lawmakers.",
        "Lawmakers debated the proposed bill on {issue} for hours before adjourning without a vote.",
        "The prime minister addressed concerns over {issue} during a press conference, promising further reforms.",
        "Polling data released this week suggests public opinion on {issue} remains sharply divided.",
        "A parliamentary committee began an inquiry into {issue}, with witnesses expected to testify next month.",
    ],
    "sport": [
        "{team} secured a dramatic victory in the final minutes, moving up in the league table.",
        "The manager of {team} said the squad's performance reflected weeks of hard work in training.",
        "{team} announced the signing of a new player ahead of the upcoming season.",
        "Injuries continue to hamper {team}, with two more players ruled out for the next match.",
        "Fans celebrated as {team} clinched the title after a hard-fought campaign.",
    ],
    "tech": [
        "{company} unveiled its latest device, featuring {desc} that the company says will redefine the {sector} market.",
        "A new report warned that {issue} poses a growing risk to {sector} companies worldwide.",
        "{company} announced layoffs affecting hundreds of employees as it restructures its {sector} division.",
        "Researchers demonstrated a new AI system capable of {desc}, raising both excitement and ethical questions.",
        "Regulators are examining {company} over concerns related to {issue} in the {sector} sector.",
    ],
}

FILL = {
    "company": ["TechCorp", "Meridian Bank", "Orion Motors", "NovaSoft", "Global Retail Group", "Atlas Energy", "Pinnacle Airlines", "BrightWave Media"],
    "change": ["a sharp rise", "a modest decline", "a record increase", "an unexpected drop", "steady growth"],
    "reason": ["rising costs", "strong consumer demand", "supply chain disruption", "currency fluctuations", "improved efficiency"],
    "reaction": ["beat", "missed", "matched", "exceeded", "fell short of"],
    "sector": ["retail", "banking", "manufacturing", "energy", "technology", "airline", "media"],
    "movement": ["rose sharply", "fell", "were largely unchanged", "surged", "slipped"],
    "event": ["a major restructuring", "a new product line", "an expansion into overseas markets", "a leadership change"],
    "macro": ["global supply shortages", "rising energy prices", "shifting trade policy", "labor market tightness"],
    "amount": ["$2.3 billion", "$450 million", "$18 million", "$1.1 billion", "$75 million"],
    "celebrity": ["a leading actress", "a popular singer", "an award-winning director", "a chart-topping band", "a veteran actor"],
    "desc": ["striking visuals", "an experimental sound", "strong performances", "groundbreaking special effects", "unprecedented processing power", "advanced natural language capabilities"],
    "team": ["the home side", "the visiting club", "the national squad", "the defending champions", "the newly promoted team"],
    "issue": ["healthcare reform", "immigration policy", "data privacy", "climate regulation", "tax reform", "cybersecurity", "trade tariffs"],
}


def fill_template(t):
    out = t
    for key, options in FILL.items():
        placeholder = "{" + key + "}"
        while placeholder in out:
            out = out.replace(placeholder, random.choice(options), 1)
    return out


def make_title(category, sentence):
    words = sentence.split()[:8]
    return " ".join(words).rstrip(".,") 


def generate(n_per_category=40):
    rows = []
    article_id = 1000
    for category in CATEGORIES:
        templates = TEMPLATES[category]
        for _ in range(n_per_category):
            n_sentences = random.randint(3, 5)
            sentences = [fill_template(random.choice(templates)) for _ in range(n_sentences)]
            content = " ".join(sentences)
            title = make_title(category, sentences[0])
            rows.append({
                "ArticleId": article_id,
                "Text": content,
                "Category": category,
                "title": title,
            })
            article_id += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate(40)
    out_path = os.path.join(os.path.dirname(__file__), "sample_news.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ArticleId", "Text", "Category", "title"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} sample articles -> {out_path}")
