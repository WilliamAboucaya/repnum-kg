import re
from pprint import pprint

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import detect_entities, disambiguate_acronym


def build_stats(corrected: bool = True) -> pd.DataFrame:
    consultation = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                               parse_dates=["Création", "Modification"], index_col=0,
                               dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
    consultation["Lié.à.."] = consultation["Lié.à.."].fillna("Unknown")
    consultation["Type.de.profil"] = consultation["Type.de.profil"].fillna("Unknown")
    proposals = consultation.loc[consultation["Type.de.contenu"] == "Proposition"]
    proposals["Contenu"] = proposals["Contenu"].apply(lambda proposal: re.sub(
        "Éléments de contexte\r?\nExplication de l'article :\r?\n", "", re.sub("(\r?\n)+", "\n", proposal)))
    proposals["full_contribution"] = proposals[["Titre", "Contenu"]].agg(". \n\n".join, axis=1)

    stats_raw = {}
    stats_unique = {}

    re_acronym = re.compile(r"^([A-Z]\.?)+$")

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    for idx, proposal in tqdm(proposals.iterrows(), total=proposals.shape[0]):
        response_json = detect_entities(proposal["full_contribution"], confidence=0.6)

        if "Resources" not in response_json:
            continue

        resources = response_json["Resources"]
        unique_resources = []

        for resource in resources:
            if all(map(lambda x: resource["@URI"] != x["@URI"], unique_resources)):
                unique_resources.append(resource)

        for resource in unique_resources:
            if re_acronym.match(resource["@surfaceForm"]) and corrected:
                entity = disambiguate_acronym(resource["@surfaceForm"], proposal["full_contribution"], model)
            else:
                entity = resource["@URI"]

            if entity in stats_unique.keys():
                stats_unique[entity] += 1
                stats_raw[entity] += sum(map(lambda x: x["@URI"] == entity, resources))
            else:
                stats_unique[entity] = 1
                stats_raw[entity] = sum(map(lambda x: x["@URI"] == entity, resources))

    stats = pd.DataFrame(data={
        "resource": list(stats_unique.keys()),
        "unique_count": list(stats_unique.values()),
        "raw_count": list(stats_raw.values())
    })

    return stats
    # pprint(stats)
    #
    # top_stats = list(sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20])
    # pprint(top_stats)
