import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    """Префильтрация товаров"""
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    year_data = data[data['week_no'] > data['week_no'].max() - 52]
    
    no_quantity_year_items = year_data[year_data['quantity'] == 0].item_id.tolist()
    
    data = data[~data['item_id'].isin(no_quantity_year_items)]
    
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]
    
    #создаем признак с ценой
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
        
    # Уберем слишком дешевые товары (на них не заработаем)
    data = data[data['price'] > 1]

    # Уберем слишком дорогие товары
    data = data[data['price'] <= 100]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    #удаляем признак с ценой
    #data = data.drop(['price'], axis=1)

    return data

def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""
    
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    
    recs = popular.head(n).item_id
    
    return recs.tolist()

def postfilter_items(recommendations, data, item_features, N=5):
    """Постфильтрация товаров"""
    # Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]
    
    # Разные категории
    categories_used = []
    final_recommendations = []
    
    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
        
        if category not in categories_used:
            final_recommendations.append(item)
            
        unique_recommendations.remove(item)
        categories_used.append(category)
    
    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(popularity_recommendation(data, n=5)[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]
    
    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations
