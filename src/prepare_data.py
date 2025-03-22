import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from sklearn.decomposition import PCA


def vectorize_texts(texts):
    """
    Векторизует список текстов при помощи модели RoBERTa.
    Получает: texts - список текстов (строк)
    Возвращает: numpy array эмбеддингов текстов
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # загружаем предобученную модель и токенизатор
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()
    # токенизация текстов
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    # получение выходов модели
    with torch.no_grad():
        outputs = model(**inputs)
    # получение векторов для классификации (CLS token)
    cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_vectors


def reduce_dimensions(vectors):
    """
    Уменьшает размерность векторов при помощи PCA.
    Получает:
        vectors: векторы, которые надо уменьшить
        n_components: сколько компонент оставить
    Возвращает: уменьшенные векторы
    """
    # сначала определяем оптимальное о компонент
    pca_start = PCA()
    pca_start.fit(vectors)
    explained_variance = np.cumsum(pca_start.explained_variance_ratio_)
    # объясняющих 95% дисперсии, их будет 192
    n_components_95 = np.argmax(explained_variance >= 0.95) + 1
    # а теперь уменьшаем размерность до оптимальной
    pca = PCA(n_components=n_components_95)
    reduced_vectors = pca.fit_transform(vectors)

    return reduced_vectors


def prepare_data():
    """
    Готовит данные для рекомендательной системы.
    Возвращает: кортеж датафреймой (users, posts, feeds)
    """
    # загрузка таблиц
    users_df = pd.read_csv('./data/users.csv')
    feeds_df = pd.read_csv('./data/feeds.csv')
    posts_df = pd.read_csv('./data/posts.csv')
    # считаем рейтинг постов
    likes = feeds_df.groupby('post_id')['target'].sum()
    actions = feeds_df.groupby('post_id')['target'].count()
    posts_df['rating'] = likes / actions
    posts_df['rating'].fillna(0, inplace=True)
    # векторизуем посты и уменьшаем размерность
    texts = posts_df['text'].tolist()
    vectors = vectorize_texts(texts)
    reduced_vectors = reduce_dimensions(vectors)
    # добавляем уменьшенные векторы к дакафрейму постов
    reduced_vector_columns = pd.DataFrame(
        reduced_vectors,
        columns=[f'pca_{i+1}' for i in range(reduced_vectors.shape[1])],
        index=posts_df.index
    )
    posts_df = pd.concat([posts_df, reduced_vector_columns], axis=1)

    return users_df, posts_df, feeds_df
