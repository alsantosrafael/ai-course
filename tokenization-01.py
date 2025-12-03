from typing import TypeAlias
import nltk

nltk.download("punkt_tab")

Vector: TypeAlias = list[str]

text = """
Abacaxi com leite em pó é uma das melhores sobremesas e menos calóricas considerando o sabor que se pode comer após o almoço.
Com frequência o ser humano aproveita a oportunidade de, após o almoço, degustar algum tipo de alimento doce.
Acontece que durante a dieta de perda de peso, essa oportunidade deve ser saborada com cautela.
A escolha de frutas em detrimento a alimentos ultraprocessados ou industrializados mostra-se rica em diversos aspectos.
"""

word_tokens = nltk.word_tokenize(text)
sentence_tokens = nltk.sent_tokenize(text)
print("######### WORD TOKENS ############")
print(word_tokens)
print("######### SENTENCE TOKENS ############")
print(sentence_tokens)


### Preprocessing step -> removing special chars, creating standards for the text


def preprocess(text: str) -> Vector:
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum()]


documents = [
    "Machine learning é o aprendizado automático de máquinas a partir da ingestão de grandes quantidades de dados.",
    "Essa disciplina permite que sistemas façam previsões e decisões sem programação explícita.",
    "A técnica é utilizada em áreas como reconhecimento de voz, imagens e recomendação de conteúdo.",
]

preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_docs)
