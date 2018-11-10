import pandas as pd 
import numpy as np
from IPython import embed
#  ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
#        'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
#        'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace',
#        'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
#        'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
#        'weaponsAcquired', 'winPoints', 'winPlacePerc']
# DBNOs - Number of enemy players knocked.
# assists - Number of enemy players this player damaged that were killed by teammates.
# boosts - Number of boost items used.
# damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# headshotKills - Number of enemy players killed with headshots.
# heals - Number of healing items used.
# killPlace - Ranking in match of number of enemy players killed.
# killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# killStreaks - Max number of enemy players killed in a short amount of time.
# kills - Number of enemy players killed.
# longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# matchId - ID to identify match. There are no matches that are in both the training and testing set.
# matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# revives - Number of times this player revived teammates.
# rideDistance - Total distance traveled in vehicles measured in meters.
# roadKills - Number of kills while in a vehicle.
# swimDistance - Total distance traveled by swimming measured in meters.
# teamKills - Number of times this player killed a teammate.
# vehicleDestroys - Number of vehicles destroyed.
# walkDistance - Total distance traveled on foot measured in meters.
# weaponsAcquired - Number of weapons picked up.
# winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# numGroups - Number of groups we have data for in the match.
class PreProcessing(object):
    def __init__(self, path):
        data = pd.read_csv(path)
        y = data['winPlacePerc']
        del data['Id']
        del data['matchDuration']
        colums = data.columns
        m,n = data.shape
        #max_each_match = data.groupby('matchId').max()


        embed()
        



if __name__ == '__main__':
    dataProcessing = PreProcessing('/Users/hy/Downloads/all/train_V2.csv')
