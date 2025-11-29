import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def text_rank_summarize(articles, top_sentences=3):
    if not isinstance(articles, list):
        articles = [articles]

    summaries = []
    for article in articles:
        sentences = sent_tokenize(article)
        if len(sentences) <= top_sentences:
            summaries.append(" ".join(sentences))
            break

        sentence_embeddings = TfidfVectorizer().fit_transform(sentences)
        similarity_matrix = (sentence_embeddings * sentence_embeddings.T).toarray()

        sentence_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(sentence_graph)

        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True
        )
        selected_sentences = sorted(
            ranked_sentences[:top_sentences], key=lambda x: x[2]
        )
        summary = " ".join([s for (_, s, _) in selected_sentences])
        summaries.append(summary)

    return summaries
