import math
import re
from time import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None


def plot_top_words(model, feature_names, n_top_words, topic_labels=None):
    row_size = 5

    fig, axes = plt.subplots(math.ceil(len(model.components_) / row_size), row_size, figsize=(30, 15), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        if (topic_labels == None) or (len(topic_labels) != len(model.components_)):
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        else:
            ax.set_title(f"Topic {topic_idx + 1}\n{topic_labels[topic_idx]}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.tight_layout()
    plt.savefig(f"./results/figures/LDA_clusters_words_{len(model.components_)}_topics_repnum.eps", format="eps")
    plt.show()

n_features = 1000
n_components = 5
n_top_words = 20

print("Loading dataset...")
t0 = time()

consultation = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                           parse_dates=["Création", "Modification"], index_col=0,
                           dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
consultation["Lié.à.."] = consultation["Lié.à.."].fillna("Unknown")
consultation["Type.de.profil"] = consultation["Type.de.profil"].fillna("Unknown")
proposals = consultation.loc[consultation["Type.de.contenu"] == "Proposition"]
proposals["Contenu"] = proposals["Contenu"].apply(lambda proposal: re.sub(
        "Éléments de contexte\r?\nExplication de l'article :\r?\n", "", re.sub("(\r?\n)+", "\n", proposal)))
proposals["full_contribution"] = proposals[["Titre", "Contenu"]].agg(". \n\n".join, axis=1)

n_samples = len(proposals.index)

french_stopwords = urllib.request.urlopen("https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt").read().decode('utf-8').splitlines()

print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
t0 = time()
tf_vectorizer = CountVectorizer(min_df=2, max_features=n_features, stop_words=french_stopwords)
tf = tf_vectorizer.fit_transform(proposals["full_contribution"])
print("done in %0.3fs." % (time() - t0))
print()

print(
    "\n" * 2,
    "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
    % (n_samples, n_features),
)
lda = LatentDirichletAllocation(
    n_components=n_components,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))
tf_feature_names = tf_vectorizer.get_feature_names_out()

plot_top_words(lda, tf_feature_names, n_top_words, ["Legal vocabulary", "Technical specification", "Internet access", "Data protection", "Open science"])
