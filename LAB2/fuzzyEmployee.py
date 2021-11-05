"""
Autor: Filip Bianga
======================================
System przyznawania premi pracownikowi
======================================

Aby uruchomić program zainstaluj poniższe narzędzia:

pip install scikit-fuzzy
pip install matplotlib
"""
"""
Naszym systemem będzie oszacowanie przyznanej premi dla pracownika oceniając jego 4 atrybuty, które
podane są poniżej jako "Wejścia".

Wyjściem będzie oszacowana premia na podstawie podanych przez nas danych.

Wejście:
Punktualność - Punctuality -> Skala od 0 do 5
Realizacja zadan - Implementation of tasks -> Skala od 0 do 5
Kultura osobista - Personal culture -> Skala od 0 do 5
Zaangażowanie w projektach - Engagement -> Skala od 0 do 5


Wyjście:
Premia - Bonus -> Maksymalna premia może wynieść 30% wynagrodzenia danego pracownika.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


#Wejścia do systemu
punctuality = ctrl.Antecedent(np.arange(0,6,1), 'punctuality')
tasks = ctrl.Antecedent(np.arange(0,6,1), 'tasks')
culture = ctrl.Antecedent(np.arange(0,6,1), 'culture')
engagement = ctrl.Antecedent(np.arange(0,6,1), 'engagement')

#Wyjście z systemu
bonus = ctrl.Consequent(np.arange(0,31,1), 'bonus')

#Membership function
punctuality.automf(3)
tasks.automf(3)
culture.automf(3)
engagement.automf(3)

#Budowanie trójkątnej funkcji przynależności (składająca się z 3 parametrów a,b,c) - trimf
#Funkcja przynależności - uogólniona postać funkcji charakterystycznej zbioru, określona na zbiorach rozmytych.
bonus['low'] = fuzz.trimf(bonus.universe, [0,0,15.5])
bonus['medium'] = fuzz.trimf(bonus.universe, [0,15.5,31])
bonus['high'] = fuzz.trimf(bonus.universe, [15.5,31,31])

"""
Tworzymy/Definiujemy role.

1. Poor - słaba/niska premia
2. Average - przeciętna/średnia premia
3. Good - dobra/wysoka premia

Dla każdych z tych 3 ról tworzymy pewne zależności.
"""
rule1 = ctrl.Rule(punctuality['poor'] & tasks['poor'] & engagement['poor'] & culture['poor']
                  , bonus['low'])

rule2 = ctrl.Rule(punctuality['average'] & tasks['average']
                  | culture['average'] & engagement['average']
                  | punctuality['poor'] & tasks['average'] & engagement['average']
                  | punctuality['average'] & tasks['average'] & engagement['average'] & culture['average']
                  , bonus['medium'])

rule3 = ctrl.Rule(punctuality['good'] & tasks['good']
                  | culture['good'] & engagement['good']
                  | punctuality['average'] & tasks['good'] & engagement['good']
                  | punctuality['average'] & tasks['good'] & engagement['good'] & culture['good']
                  | punctuality['good'] & tasks['good'] & engagement['average'] & culture['good']
                  | punctuality['good'] & tasks['good'] & engagement['good'] & culture['good']
                  , bonus['high'])

"""
Teraz gdy mamy nasze role możemy stworzyć ControlSystem
"""

bonus_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

"""
W kolejnym etapie trzeba stworzyć naszą symulacje
"""

bonusing = ctrl.ControlSystemSimulation(bonus_ctrl)

"""
Aby móc zasymulować nasz system trzeba podać przykładowe dane wejściowe w określonym przez nas przedziale
"""

bonusing.input['punctuality'] = 5.8
bonusing.input['tasks'] = 3
bonusing.input['culture'] = 4
bonusing.input['engagement'] = 1.8

#Obliczane wyjście
bonusing.compute()

"""
Na końcu wypisywany jest nasz wynik oraz jego wizualizacja
"""
print(bonusing.output['bonus'])
bonus.view(sim=bonusing)