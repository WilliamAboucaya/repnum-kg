import re
import sys
from pprint import pprint

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import detect_entities, disambiguate_acronym

if __name__ == "__main__":
    corrected = "--corrected" in sys.argv

    consultation = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                               parse_dates=["Création", "Modification"], index_col=0,
                               dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
    consultation["Lié.à.."] = consultation["Lié.à.."].fillna("Unknown")
    consultation["Type.de.profil"] = consultation["Type.de.profil"].fillna("Unknown")
    proposals = consultation.loc[consultation["Type.de.contenu"] == "Proposition"]
    proposals["Contenu"] = proposals["Contenu"].apply(lambda proposal: re.sub(
        "Éléments de contexte\nExplication de l'article :\n", "", re.sub("\n+", "\n", proposal)))
    proposals["full_contribution"] = proposals[["Titre", "Contenu"]].agg(". \n\n".join, axis=1)

    stats = {}

    re_acronym = re.compile(r"^([A-Z]\.?)+$")

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    for idx, proposal in tqdm(proposals.iterrows(), total=proposals.shape[0]):
        response_json = detect_entities(proposal["full_contribution"], confidence=0.6)

        if "Resources" not in response_json:
            continue

        unique_resources = []

        for resource in response_json["Resources"]:
            if all(map(lambda x: resource["@URI"] != x["@URI"], unique_resources)):
                unique_resources.append(resource)

        for resource in unique_resources:
            if re_acronym.match(resource["@surfaceForm"]) and corrected:
                entity = disambiguate_acronym(resource["@surfaceForm"], proposal["full_contribution"], model)
            else:
                entity = resource["@URI"]

            if entity in stats.keys():
                stats[entity] += 1
            else:
                stats[entity] = 1

    pprint(stats)

    top_stats = list(sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20])
    pprint(top_stats)
