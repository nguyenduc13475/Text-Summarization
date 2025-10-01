import networkx as nx
import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:100]")
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


if __name__ == "__main__":
    for i in range(2):
        article = dataset[i]["article"]
        target_summary = dataset[i]["highlights"]

        text_rank_summary = text_rank_summarize(article, top_sentences=3)[0]

        print("=" * 80)
        print("Article:\n", article[:700], "...")
        print("\nTarget summary:\n", target_summary)
        print("\nText Rank summary:\n", text_rank_summary)
        print("=" * 80)
