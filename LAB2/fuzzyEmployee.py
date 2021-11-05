import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

"""
Filip Bianga

Wejście:
Punktualność - Punctuality
Realizacja zadan - Implementation of tasks
Kultura osobista - Personal culture

Skala - przedział od 0 do 5
poor, average, good


Wyjście:
Premia - Bonus

Maksymalna premia może wynieść 30% wynagrodzenia danego pracownika.
low, medium, high

"""

punctuality = ctrl.Antecedent(np.arange(0,6,1), 'punctuality')
tasks = ctrl.Antecedent(np.arange(0,6,1), 'tasks')
culture = ctrl.Antecedent(np.arange(0,6,1), 'culture')


bonus = ctrl.Consequent(np.arange(0,31,1), 'bonus')

punctuality.automf(3)
tasks.automf(3)
culture.automf(3)

bonus['low'] = fuzz.trimf(bonus.universe, [0,0,15.5])
bonus['medium'] = fuzz.trimf(bonus.universe, [0,15.5,31])
bonus['high'] = fuzz.trimf(bonus.universe, [15.5,31,31])

punctuality.view()
tasks.view()
culture.view()
bonus.view()

rule1 = ctrl.Rule(punctuality['poor'] | tasks['poor'] | culture['poor'], bonus['low'])
rule2 = ctrl.Rule(tasks['average'] | culture['average'], bonus['medium'])
rule3 = ctrl.Rule(punctuality['good'] | tasks['good'] | culture['good'], bonus['high'])

bonus_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

bonusing = ctrl.ControlSystemSimulation(bonus_ctrl)

bonusing.input['punctuality'] = 4.5
bonusing.input['tasks'] = 4.8
bonusing.input['culture'] = 5.1

bonusing.compute()

print(bonusing.output['bonus'])
bonus.view(sim=bonusing)

plt.show()

