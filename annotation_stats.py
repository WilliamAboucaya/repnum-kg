import re

import pandas as pd

import recognized_entity_stats


def compute_stats() -> dict:
    consultation = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                               parse_dates=["Création", "Modification"], index_col=0,
                               dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
    consultation["Lié.à.."] = consultation["Lié.à.."].fillna("Unknown")
    consultation["Type.de.profil"] = consultation["Type.de.profil"].fillna("Unknown")
    proposals = consultation.loc[consultation["Type.de.contenu"] == "Proposition"]
    proposals["Contenu"] = proposals["Contenu"].apply(lambda proposal: re.sub(
        "Éléments de contexte\r?\nExplication de l'article :\r?\n", "", re.sub("(\r?\n)+", "\n", proposal)))
    proposals["full_contribution"] = proposals[["Titre", "Contenu"]].agg(". \n\n".join, axis=1)

    proposals_annotated_corrected = pd.read_csv("./results/proposals_annotated_corrected.csv", index_col=0,
                                                dtype={"proposal_html": str, "num_annotations": int})

    entities_data = recognized_entity_stats.build_stats()

    stats = dict()

    stats["Number of proposals"] = len(proposals.index)
    stats["Raw number of annotations"] = proposals_annotated_corrected["num_annotations"].sum()
    stats["Number of different entities"] = len(entities_data)
    stats["Number of entities linked with exactly 1 proposal"] = len(entities_data.loc[entities_data["unique_count"] == 1].index)
    over_5anno_entities = entities_data.loc[entities_data["unique_count"] >= 5]
    stats["Number of entities linked with 5 or more proposals"] = len(over_5anno_entities.index)
    stats["Number of annotations for entities linked with 5 or more proposals"] = over_5anno_entities["raw_count"].sum()
    stats["Number of proposals without any annotation"] = len(proposals_annotated_corrected.loc[
        proposals_annotated_corrected["num_annotations"] == 0].index)
    stats["Number of proposals with 5 or more annotations"] = len(proposals_annotated_corrected.loc[
        proposals_annotated_corrected["num_annotations"] >= 5].index)
    stats["Median number of annotations per proposal"] = int(proposals_annotated_corrected["num_annotations"].median())

    return stats
