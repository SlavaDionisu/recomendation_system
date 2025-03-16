from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import func, create_engine, text
import os
from catboost import CatBoostClassifier
import pandas as pd
from datetime import datetime
from src.schema import PostGet
from src.prepare_data import prepare_data

app = FastAPI()

def load_model():
    model_path = "./recommendation_system/catboost_model_with_vectors"
    from_file = CatBoostClassifier() 
    return from_file.load_model(model_path)

# загрузка и предобработка данных
print("Загружаем данные. Пожалуйста, подождите, это может занять некоторое время...")
df_users, df_posts, df_feeds = prepare_data()

best_posts = df_posts.sort_values(by='rating', ascending=False).head(5)[['post_id', 'text', 'topic']].rename(columns={'post_id': 'id'}).to_dict('records')
model = load_model() # загружаем обученную модель

def recomendation_posts(user_id: int, df_users: pd.DataFrame, df_posts: pd.DataFrame, df_feeds: pd.DataFrame, limit: int = 5) -> List[dict]:
    df = df_users[df_users['user_id']==user_id] # находим пользователя
    if df.empty: # если пользователь не найден, возвращаем лучшие посты
        return best_posts
    df_feeds_curr = df_feeds[df_feeds['user_id']==user_id] # находим посты которые юзер видел
    df_posts_new = df_posts[~df_posts['post_id'].isin(df_feeds_curr['post_id'])] # убираем посты которые юзер видел
    df = df.merge(df_posts_new.drop(columns='text'), how='cross') #готовим данные для модельки
    df = df.drop(columns = 'user_id')
    df = df.set_index('post_id') #это чтоб потом найти
    #зафиксировали столбцы на всякий случай
    df = df[['topic','rating','pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10','pca_11','pca_12','pca_13','pca_14','pca_15','pca_16','pca_17','pca_18','pca_19','pca_20','pca_21','pca_22','pca_23','pca_24','pca_25','pca_26','pca_27','pca_28','pca_29','pca_30','pca_31','pca_32','pca_33','pca_34','pca_35','pca_36','pca_37','pca_38','pca_39','pca_40','pca_41','pca_42','pca_43','pca_44','pca_45','pca_46','pca_47','pca_48','pca_49','pca_50','pca_51','pca_52','pca_53','pca_54','pca_55','pca_56','pca_57','pca_58','pca_59','pca_60','pca_61','pca_62','pca_63','pca_64','pca_65','pca_66','pca_67','pca_68','pca_69','pca_70','pca_71','pca_72','pca_73','pca_74','pca_75','pca_76','pca_77','pca_78','pca_79','pca_80','pca_81','pca_82','pca_83','pca_84','pca_85','pca_86','pca_87','pca_88','pca_89','pca_90','pca_91','pca_92','pca_93','pca_94','pca_95','pca_96','pca_97','pca_98','pca_99','pca_100','pca_101','pca_102','pca_103','pca_104','pca_105','pca_106','pca_107','pca_108','pca_109','pca_110','pca_111','pca_112','pca_113','pca_114','pca_115','pca_116','pca_117','pca_118','pca_119','pca_120','pca_121','pca_122','pca_123','pca_124','pca_125','pca_126','pca_127','pca_128','pca_129','pca_130','pca_131','pca_132','pca_133','pca_134','pca_135','pca_136','pca_137','pca_138','pca_139','pca_140','pca_141','pca_142','pca_143','pca_144','pca_145','pca_146','pca_147','pca_148','pca_149','pca_150','pca_151','pca_152','pca_153','pca_154','pca_155','pca_156','pca_157','pca_158','pca_159','pca_160','pca_161','pca_162','pca_163','pca_164','pca_165','pca_166','pca_167','pca_168','pca_169','pca_170','pca_171','pca_172','pca_173','pca_174','pca_175','pca_176','pca_177','pca_178','pca_179','pca_180','pca_181','pca_182','pca_183','pca_184','pca_185','pca_186','pca_187','pca_188','pca_189','pca_190','pca_191','pca_192','gender','age','country','city','exp_group','os','source']] 
    df['predict_proba'] = model.predict_proba(df)[:,1] # вероятности для ервого класса, т е лайка
    df = df.sort_values(by='predict_proba', ascending=False).head(limit)
    df = df.merge(df_posts[['post_id', 'text']], left_index=True, right_on='post_id', how='left').reset_index() # вернули сам текст поста, а то ж мы его убирали
    df = df[['post_id','text','topic']] # оставили только нужные столбцы
    df = df.rename(columns={'post_id': 'id'}) # переименовали, чтоб класс смог понять
    list_of_dicts = df.to_dict('records')
    return list_of_dicts

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int, 
		limit: int = 5,
        ) -> List[PostGet]:
    recommended_posts_list = recomendation_posts(id, df_users, df_posts, df_feeds, limit)
    return [PostGet(**row) for row in recommended_posts_list] # превратили словарь в объект класса