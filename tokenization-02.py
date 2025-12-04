import nltk
from numpy import number
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> list[str]:
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    return [word for word in tokens if word.isalnum()]


def perform_preprocess_and_join(words: list[str]) -> list[str]:
    return [" ".join(preprocess(word)) for word in words]


documents = [
    "Machine Learning é legal",
    "Aprender sobre machine learning pode ser essencial",
    "Jamais se viu uma revolução como aquela que a IA está promovendo",
    "Conta-se que mesmo a internet não impactou o mundo como a IA",
    "Machine learning é um dos campos que está incluso em inteligência artificial",
    "Em adição a isso, pode-se citar natural language processing, robotics, deep learning, dentre outros",
    "Machine learning é uma área da inteligência artificial que se concentra em formas de gerar aprendizado à um computador",
    "Deep learning é a ciência cujo objeto de estudo é o aprendizado profundo",
]
# Preprocessing step
preprocessed_docs = perform_preprocess_and_join(documents)
# Starting Vector Space Model
vectorizer = TfidfVectorizer()
# Creating Vector Space Model for the documents
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
query = "o que é machine learning"


# Now we create a vector to represent the query in that n-dimension space
# where the documents live
def generate_tfidf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    # Calculating relevance from document vectors
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    return sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)


similarities_ordered = generate_tfidf(query, vectorizer, tfidf_matrix)
print("########## TOP 5 Most Similar documents to the query ##########")
for index, similarity in similarities_ordered[:5]:
    print(index, similarity)
