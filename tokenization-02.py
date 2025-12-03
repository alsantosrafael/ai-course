import nltk
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
    "Conta-se que mesmo a internet não impactou o mundo como o ML",
]
# Preprocessing step
preprocessed_docs = perform_preprocess_and_join(documents)
# Starting Vector Space Model
vectorizer = TfidfVectorizer()
# Creating Vector Space Model for the documents
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# Now we create a vector to represent the query in that n-dimension space
# where the documents live
query = "machine learning"
query_vector = vectorizer.transform([query])
print(preprocessed_docs)
print(tfidf_matrix)
print(query_vector)
# Calculating relevance from document vectors
similarity = cosine_similarity(tfidf_matrix, query_vector).flatten()
print(similarity)
