import json
import re

import requests
from requests import Response


def detect_entities(text: str, confidence: float = 0.5, support: float = 0, format: str = "application/json"):
    if format not in ["application/json", "text/html", "application/n-triples", "application/ld+json", "text/turtle"]:
        raise ValueError(f"Parameter 'format' set to {format}, should be one of the following strings: 'application/json', 'text/html', 'application/n-triples', 'application/ld+json', 'text/turtle'")

    filter_q = """PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT DISTINCT ?a WHERE {
        {?a a dbo:Year}
        UNION
        {?a a dbo:Film .
         MINUS { ?a dbo:genre dbpedia-fr:Documentaire . }}
    }"""

    data = {
        "text": text,
        "confidence": confidence,
        "support": support,
        "sparql": filter_q,
        "types": "DBpedia:Film",
        "policy": "blacklist"
    }

    response = requests.post("https://api.dbpedia-spotlight.org/fr/annotate", data=data, headers={"accept": format, "content-type": "application/x-www-form-urlencoded"})
    response.encoding = "utf-8"

    if format == "text/html":
        response_processed = response.content.decode("utf-8")
        r = '(<a href="http://fr\\.dbpedia\\.org/resource/([1-2][0-9]|3[0-1]|1er|[2-9])_(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)".*?>)(.*?)(<\\/a>)'
        response_processed = re.sub(r, "\\4", response_processed)

    elif format == "application/json":
        response_processed = response.json()
        if "Resources" in response_processed:
            r = 'http:\\/\\/fr.dbpedia.org\\/resource\\/([1-2][0-9]|3[0-1]|1er|[2-9])_(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'
            response_processed["Resources"] = list(filter(lambda resource: re.match(r, resource["@URI"]) is None,
                                                          response_processed["Resources"]))
    else:
        response_processed = response.content

    return response_processed
