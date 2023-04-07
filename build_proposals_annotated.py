import csv
import re

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import detect_entities, disambiguate_acronym

def build(corrected: bool = True):
    consultation = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                               parse_dates=["Création", "Modification"], index_col=0,
                               dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
    consultation["Lié.à.."] = consultation["Lié.à.."].fillna("Unknown")
    consultation["Type.de.profil"] = consultation["Type.de.profil"].fillna("Unknown")
    proposals = consultation.loc[consultation["Type.de.contenu"] == "Proposition"]
    proposals["Contenu"] = proposals["Contenu"].apply(lambda proposal: re.sub(
        "Éléments de contexte\r?\nExplication de l'article :\r?\n", "", re.sub("(\r?\n)+", "\n", proposal)))
    proposals["full_contribution"] = proposals[["Titre", "Contenu"]].agg(". \n\n".join, axis=1)

    print("Building the annotated file...")

    with open(f"./results/proposals_annotated{'_corrected' if corrected else ''}.csv", "w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["prop_idx", "proposal_html", "num_annotations"],
                                delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        re_annotation = re.compile(r"<a href=.+?</a>")
        re_acronym = re.compile(r"^([A-Z]\.?)+$")

        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        for idx, proposal in tqdm(proposals.iterrows(), total=proposals.shape[0]):
            response_html = detect_entities(proposal["full_contribution"], confidence=0.6, format="text/html")
            response_html = response_html.split("<body>")[1]
            response_html = response_html.split("</body>")[0]

            annotations = re_annotation.findall(response_html)
            if annotations:
                num_annotations = len(annotations)

                if corrected:
                    for annotation in set(annotations):
                        term = annotation.split('">')[1].split("</")[0]
                        if re_acronym.match(term):
                            entity_corrected = disambiguate_acronym(term, proposal["full_contribution"], model)
                            annotation_corrected = f'<a href="{entity_corrected}" title="{entity_corrected}" target="_blank">{term}</a>'

                            response_html = re.sub(annotation, annotation_corrected, response_html)
            else:
                num_annotations = 0

            writer.writerow({"prop_idx": idx, "proposal_html": response_html, "num_annotations": num_annotations})
        print("Done!")
