import json
from scores import euclidean_score


# Funkcja wyświetla polecane / nie polecane filmy ,dla podanego użytkownika
def getRecommendation(dataset, user_dict, user):
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
    recommendedMovies = dict(list(sort_dict.items())[-5:])

    # The worst 5 movies
    notRecommededMovies = dict(list(sort_dict.items())[:5])

    movieList = []
    i: int = 0

    print('\nThe best recommendation movies:')
    for movie in recommendedMovies:
        print(movie)
        i += 1
        movieList.append(movie)

    print('\nThe worst recommendation movies:')
    for movie in notRecommededMovies:
        print(movie)
        i += 1
        movieList.append(movie)

    return movieList


if __name__ == '__main__':

    user = 'Paweł Czapiewski'

    ratings_file = 'rating.json'
    with open(ratings_file, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())

    scoreList = []
    for item in data:
        if item != user:
            scoreList.append({'score': euclidean_score(data, user, item), 'user': item})

    bestScore = max(scoreList, key=lambda x: x['score'])
    user_dict = data[bestScore['user']]
    movieList = getRecommendation(data, user_dict, user)
