FEATURES_GROUP = {

"target" : ["total_points"],

"id_cols" : ["name", "element", "kickoff_time", "gw", "code", "season"],

"static_cols" : ["position", "was_home", "starts"],

"fpl_cols" : ['bonus', 'bps', 'ict_index', 'influence', 'value', 'threat',  'selected', 'transfers_out', 'transfers_balance', 'transfers_in', "transfers_trend"],

"perf_cols" : [
 'yellow_cards',
 'xg_conceded_per_90',
 'clean_sheets',
 'minutes',
 'xg_involvements_per_90',
 'xg_per_90',
 'xa_per_90',
 'own_goals',
 'assists',
 'xg',
 'saves',
 'penalties_saved',
 'xp',
 'penalties_missed',
 'goals_scored',
 'team_h_score',
 'xa',
 'xg_involvements',
 'team_a_score',
 'goals_conceded',
 'creativity',
 'xg_conceded',
 'pca_cluster',
 'dist_to_centroid',
 'red_cards'],

"pre_game_cols" : ["name", "position", "element", "was_home", "was_home", "gw", "code", "season", "kickoff_time", "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend"]
}




players_x_stats = [
    'xg_per_90',
    'xa_per_90',
    'shots',
    'key_passes',
    'ict_index_per_90',
    'xg_involvements_per_90'
]

teams_x_stats = [
    'team_xg_per_90',
    'opp_xg_per_90',
    'team_np_xg_difference_per_90',
    'team_ppda',
    'team_match_np_xg_diff_per_90'
]
