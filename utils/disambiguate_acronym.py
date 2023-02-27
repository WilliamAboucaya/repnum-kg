import re

import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer, util
from wikipedia import wikipedia

from utils.queries import *


def disambiguate_acronym(acronym: str, proposal: str, model: SentenceTransformer):
    sparql = SPARQLWrapper("http://fr.dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery(node_exists_q.format(re.escape(acronym)))
    ret = sparql.queryAndConvert()

    if ret["boolean"]:  # The node exists
        sparql.setQuery(node_disambiguates_q.format(re.escape(acronym)))
        ret = sparql.queryAndConvert()

        if len(ret["results"]["bindings"]) > 0:  # The node is a disambiguation
            similarity_scores = [0 for _ in ret["results"]["bindings"]]
            proposal_emb = model.encode(proposal)

            for i in range(len(ret["results"]["bindings"])):
                row = ret["results"]["bindings"][i]

                row_abstract_emb = model.encode(row["abstract"]["value"])
                similarity_scores[i] = util.cos_sim(proposal_emb, row_abstract_emb).numpy()[0][0]

            return ret["results"]["bindings"][np.argmax(similarity_scores)]["disambiguates"]["value"]

        else:
            sparql.setQuery(node_redirects_q.format(re.escape(acronym)))
            ret = sparql.queryAndConvert()
            if len(ret["results"]["bindings"]) > 0:  # The node redirects to a precise Wikipedia page
                return ret["results"]["bindings"][0]["redirects"]["value"]
            else:  # The node is a precise Wikipedia page
                return f"http://fr.dbpedia.org/resource/{acronym}"
    else:  # The node does not exist
        wikipedia.set_lang("fr")
        wiki_pages_titles = wikipedia.search(acronym, results=5)

        similarity_scores = [0 for _ in wiki_pages_titles]
        proposal_emb = model.encode(proposal)

        for i in range(len(wiki_pages_titles)):
            page_abstract = wikipedia.summary(wiki_pages_titles[i])
            page_abstract_emb = model.encode(page_abstract)
            similarity_scores[i] = util.cos_sim(proposal_emb, page_abstract_emb).numpy()[0][0]

        best_page = wiki_pages_titles[np.argmax(similarity_scores)]
        sparql.setQuery(node_from_fr_title.format(best_page))
        ret = sparql.queryAndConvert()

        return ret["results"]["bindings"][0]["page"]["value"]
