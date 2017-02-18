#!/usr/bin/env python2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def println():
    print "\n"
    return

df_seeds = pd.read_csv("data/TourneySeeds.csv")
df_tour = pd.read_csv("data/TourneyCompactResults.csv")

print "Seeds and results"
print df_seeds.head()
print df_tour.head()
println()

df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis = 1) # premove unnecessary shit

def seed_to_int(seed):
        s_int = int(seed[1:3])
        return s_int

df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)

print "Modified seeds"
print df_seeds.head()
println()

df_winseeds = df_seeds.rename(columns={'Team': 'Wteam', 'n_seed':'win_seed'})
df_lossseeds = df_seeds.rename(columns={'Team': 'Lteam', 'n_seed':'loss_seed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'Wteam'])
df_concat = pd.merge(left = df_dummy, right = df_lossseeds, on=['Season', 'Lteam'])
df_concat['seed_diff'] = df_concat.win_seed - df_concat.loss_seed

print "Win seeds, Loss seeds, dummy, and concat"
print df_winseeds.head()
print df_lossseeds.head()
print df_dummy.head()
print df_concat.head()
println()

df_wins = pd.DataFrame()
df_wins['seed_diff'] = df_concat['seed_diff']
df_wins['result'] = 1

df_losses = pd.DataFrame()
df_losses['seed_diff'] = -df_concat['seed_diff']
df_losses['result'] = 0

df_for_predictions = pd.concat((df_wins, df_losses))
print df_for_predictions.head()
println()

X_train = df_for_predictions.seed_diff.values.reshape(-1,1)
y_train = df_for_predictions.result.values
X_train, y_train = shuffle(X_train, y_train)

logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
