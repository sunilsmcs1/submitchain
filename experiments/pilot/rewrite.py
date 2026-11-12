"""Rule-based rewriter for the light and medium duplicate levels of the pilot benchmark.

light  : about 35% of dictionary words replaced by a synonym; sentence order kept; title edited the same way.
medium : about 80% of dictionary words replaced; sentences after the first shuffled; one sentence dropped
         when the abstract has four or more; two adjacent sentences merged; title edited.
Heavy duplicates are not produced here: they were written by a language model (data/heavy.jsonl).
"""
from __future__ import annotations

import random
import re

SYN = {
 "propose": ["present", "introduce", "put forward"], "proposed": ["presented", "introduced", "suggested"],
 "proposes": ["presents", "introduces"], "novel": ["new", "original"], "method": ["approach", "technique"],
 "methods": ["approaches", "techniques"], "approach": ["method", "strategy"], "approaches": ["methods", "strategies"],
 "technique": ["method", "procedure"], "techniques": ["methods", "procedures"], "model": ["framework", "architecture"],
 "models": ["frameworks", "architectures"], "use": ["employ", "utilise"], "used": ["employed", "utilised", "applied"],
 "uses": ["employs", "utilises"], "using": ["employing", "utilising", "with"], "improve": ["enhance", "boost"],
 "improves": ["enhances", "boosts"], "improved": ["enhanced", "better"], "improving": ["enhancing", "boosting"],
 "performance": ["effectiveness", "results"], "results": ["findings", "outcomes"], "result": ["outcome", "finding"],
 "show": ["demonstrate", "indicate"], "shows": ["demonstrates", "indicates"], "shown": ["demonstrated", "established"],
 "important": ["significant", "crucial"], "significant": ["considerable", "substantial"], "challenge": ["difficulty", "problem"],
 "challenges": ["difficulties", "problems"], "challenging": ["difficult", "demanding"], "problem": ["issue", "difficulty"],
 "problems": ["issues", "difficulties"], "issue": ["problem", "concern"], "issues": ["problems", "concerns"],
 "however": ["nevertheless", "yet"], "However": ["Nevertheless", "Yet"], "moreover": ["furthermore", "in addition"],
 "Moreover": ["Furthermore", "In addition"], "therefore": ["thus", "hence"], "Therefore": ["Thus", "Hence"],
 "paper": ["study", "work", "article"], "study": ["work", "investigation"], "studies": ["works", "investigations"],
 "research": ["work", "investigation"], "existing": ["current", "prior"], "current": ["present", "existing"],
 "various": ["several", "diverse"], "different": ["distinct", "diverse"], "several": ["multiple", "a number of"],
 "many": ["numerous", "a large number of"], "large": ["big", "sizeable"], "small": ["little", "limited"],
 "high": ["elevated", "strong"], "low": ["reduced", "limited"], "accuracy": ["precision", "correctness"],
 "efficient": ["effective", "economical"], "effective": ["efficient", "successful"], "effectively": ["efficiently", "successfully"],
 "framework": ["architecture", "scheme"], "frameworks": ["architectures", "schemes"], "system": ["platform", "framework"],
 "systems": ["platforms", "frameworks"], "data": ["information", "records"], "information": ["data", "knowledge"],
 "provide": ["offer", "supply"], "provides": ["offers", "supplies"], "provided": ["offered", "supplied"],
 "achieve": ["attain", "reach"], "achieves": ["attains", "reaches"], "achieved": ["attained", "reached"],
 "evaluate": ["assess", "examine"], "evaluated": ["assessed", "examined"], "evaluation": ["assessment", "analysis"],
 "analysis": ["examination", "study"], "analyze": ["examine", "study"], "analyse": ["examine", "study"],
 "develop": ["build", "create"], "developed": ["built", "created"], "development": ["growth", "advancement"],
 "increase": ["rise", "growth"], "increasing": ["growing", "rising"], "increased": ["greater", "raised"],
 "reduce": ["lower", "decrease"], "reduces": ["lowers", "decreases"], "reduced": ["lowered", "decreased"],
 "detect": ["identify", "spot"], "detecting": ["identifying", "spotting"], "detection": ["identification", "recognition"],
 "attack": ["threat", "intrusion"], "attacks": ["threats", "intrusions"], "secure": ["safe", "protected"],
 "security": ["protection", "safety"], "privacy": ["confidentiality", "data protection"], "sharing": ["exchange", "distribution"],
 "share": ["exchange", "distribute"], "network": ["net", "infrastructure"], "networks": ["nets", "infrastructures"],
 "task": ["job", "assignment"], "tasks": ["jobs", "assignments"], "learning": ["training", "learning"],
 "learn": ["acquire", "capture"], "features": ["attributes", "characteristics"], "feature": ["attribute", "characteristic"],
 "recent": ["latest", "new"], "recently": ["lately", "in recent years"], "widely": ["broadly", "extensively"],
 "major": ["main", "key"], "key": ["central", "main"], "critical": ["crucial", "vital"], "crucial": ["critical", "vital"],
 "demonstrate": ["show", "illustrate"], "experiments": ["tests", "trials"], "experimental": ["empirical", "test"],
 "dataset": ["data set", "corpus"], "datasets": ["data sets", "corpora"], "users": ["individuals", "people"],
 "user": ["individual", "person"], "allow": ["enable", "permit"], "allows": ["enables", "permits"],
 "enable": ["allow", "permit"], "enables": ["allows", "permits"], "require": ["need", "demand"], "requires": ["needs", "demands"],
 "address": ["tackle", "handle"], "solve": ["resolve", "address"], "consider": ["examine", "take into account"],
 "focus": ["concentrate", "centre"], "focuses": ["concentrates", "centres"], "obtain": ["acquire", "get"],
 "based": ["built", "founded"], "combine": ["merge", "integrate"], "combining": ["merging", "integrating"],
 "extract": ["derive", "pull out"], "investigate": ["explore", "examine"], "accurate": ["precise", "exact"],
 "accurately": ["precisely", "exactly"], "traditional": ["conventional", "classical"], "potential": ["possible", "prospective"],
 "advantages": ["benefits", "strengths"], "limitations": ["shortcomings", "drawbacks"], "limited": ["restricted", "constrained"],
 "applications": ["uses", "deployments"], "application": ["use", "deployment"], "design": ["devise", "create"],
 "designed": ["devised", "created"], "medical": ["clinical", "healthcare"], "patients": ["individuals receiving care", "people"],
 "images": ["scans", "pictures"], "image": ["scan", "picture"], "segmentation": ["delineation", "partitioning"],
 "lesion": ["abnormality", "lesion"], "lesions": ["abnormalities", "lesions"], "recommendation": ["suggestion", "recommendation"],
 "graph": ["network graph", "graph"], "node": ["vertex", "node"], "nodes": ["vertices", "nodes"],
 "computing": ["computation", "processing"], "computation": ["processing", "computing"], "resource": ["capacity", "resource"],
 "resources": ["capacities", "assets"], "allocation": ["assignment", "distribution"], "devices": ["equipment", "units"],
 "server": ["host", "server"], "servers": ["hosts", "servers"], "clients": ["participants", "clients"],
 "classification": ["categorisation", "labelling"], "sentiment": ["opinion", "sentiment"], "text": ["writing", "textual content"],
 "fine-tune": ["adapt", "refine"], "pre-trained": ["pretrained", "previously trained"], "language": ["linguistic", "language"],
}
PHRASES = [
 ("In this paper,", ["In this work,", "This study", "Here,"]), ("In this paper", ["In this work", "Here"]),
 ("this paper", ["this work", "this study"]), ("we propose", ["we present", "we introduce"]),
 ("We propose", ["We present", "We introduce"]), ("deep learning", ["deep neural learning", "deep learning"]),
 ("state-of-the-art", ["leading", "cutting-edge"]), ("a variety of", ["a range of", "diverse"]),
 ("in order to", ["to", "so as to"]), ("as well as", ["and", "along with"]),
]


def _sub_words(text: str, p: float, rng: random.Random) -> str:
    def rep(m):
        w = m.group(0)
        if w in SYN and rng.random() < p:
            return rng.choice(SYN[w])
        lw = w.lower()
        if lw in SYN and rng.random() < p:
            s = rng.choice(SYN[lw]); return s[0].upper() + s[1:] if w[0].isupper() else s
        return w
    return re.sub(r"[A-Za-z][A-Za-z\-]*", rep, text)


def _phrases(text: str, p: float, rng: random.Random) -> str:
    for a, alts in PHRASES:
        if a in text and rng.random() < p:
            text = text.replace(a, rng.choice(alts), 1)
    return text


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s]


def rewrite(title: str, abstract: str, level: str, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    p = {"light": 0.35, "medium": 0.8}[level]
    sents = sentences(abstract)
    if level == "medium" and len(sents) >= 3:
        head, rest = sents[0], sents[1:]
        rng.shuffle(rest)
        if len(sents) >= 4:
            rest.pop(rng.randrange(len(rest)))
        if len(rest) >= 2:
            i = rng.randrange(len(rest) - 1)
            rest[i:i + 2] = [rest[i].rstrip(".") + "; " + rest[i + 1][0].lower() + rest[i + 1][1:]]
        sents = [head] + rest
    out = " ".join(_sub_words(_phrases(s, p, rng), p, rng) for s in sents)
    return _sub_words(title, p, rng), out
