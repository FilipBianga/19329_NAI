"""
Autor: Filip Bianga
======================================
Movies recommendation system
======================================
* Code inspired by the app from the class

To run the program, install the following tools(if you dont have):
pip install numpy

python/python3 main.py
...
"""

"""
A system that compares users and selects the best fit, 
so you can choose which can be recommended and not recommended.
"""

import json
from scores import euclidean_score


def recommendationMovies(dataset, user_dict, user):
    if user not in dataset:
        raise TypeError("Cannot find " + user + " in the databases")

    # dict items are presented in key:value pairs
    new_dict = {}

    for key, value in user_dict.items():
        if key not in data[user].keys():
            new_dict[key] = value
    # Movies of the best matched user
    # print(new_dict)

    # print(sort_dict)
    # Sort movies
    sort_dict = dict(sorted(new_dict.items(), key=lambda element: element[1]))

    # The best 5 movies
    bestRecommendation = dict(list(sort_dict.items())[-5:])

    # The worst 5 movies
    worstRecommandation = dict(list(sort_dict.items())[:5])

    movieList = []
    i: int = 0

    print('\033[92m'+'\nThe best recommendation movies:')
    for movie in bestRecommendation:
        print(movie)
        i += 1
        movieList.append(movie)

    print('\033[91m'+'\nThe worst recommendation movies:')
    for movie in worstRecommandation:
        print(movie)
        i += 1
        movieList.append(movie)

    return movieList


if __name__ == '__main__':
    # Choose user
    user = 'Paweł Czapiewski'

    # Load dataset in json
    ratings_file = 'rating.json'
    with open(ratings_file, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())

    scoreList = []
    for item in data:
        if item != user:
            scoreList.append({'score': euclidean_score(data, user, item), 'user': item})

    print("\nUżytkownik: " + user)
    bestScore = max(scoreList, key=lambda x: x['score'])
    user_dict = data[bestScore['user']]
    movieList = recommendationMovies(data, user_dict, user)
