node_exists_q = """PREFIX dbo: <http://dbpedia.org/ontology/>

ASK {{
    dbpedia-fr:{0} dbo:wikiPageID ?id
}}"""

node_disambiguates_q = """PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?disambiguates ?abstract WHERE {{
    dbpedia-fr:{0} dbo:wikiPageDisambiguates ?disambiguates.
    ?disambiguates dbo:abstract ?abstract
}}"""

node_redirects_q = """PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?redirects ?abstract WHERE {{
    dbpedia-fr:{0} dbo:wikiPageRedirects ?redirects.
    ?redirects dbo:abstract ?abstract
}}"""

rua_abstract_q = """PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?abstract WHERE {
    dbpedia-fr:Revenu_universel_d\\'activité dbo:abstract ?abstract
}"""

node_from_fr_title = """PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?page WHERE {{
    ?page rdfs:label \"{0}\"@fr
}}
"""