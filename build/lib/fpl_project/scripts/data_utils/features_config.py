from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesConfig:
    target: list = field(default_factory=lambda: ["total_points"])

    id_cols: list = field(default_factory=lambda: [
        "name", "element", "code", "season", "gw", "kickoff_time",
        "team", "opponent_team"
    ])

    static_cols: list = field(default_factory=lambda: [
        "position", "was_home", "starts"
    ])

    fpl_cols: list = field(default_factory=lambda: [
        "bonus", "bps", "ict_index", "influence", "threat", "creativity",
        "value", "selected", "transfers_in", "transfers_out",
        "transfers_balance", "transfers_trend"
    ])

    perf_cols: list = field(default_factory=lambda: [
        "goals_scored", "assists", "own_goals", "penalties_missed",
        "xg", "xa", "xg_per_90", "xa_per_90",
        "xg_involvements", "xg_involvements_per_90",
        "goals_conceded", "xg_conceded", "xg_conceded_per_90",
        "clean_sheets", "saves", "penalties_saved",
        "minutes", "yellow_cards", "red_cards",
        "team_h_score", "team_a_score",
        "pca_cluster", "dist_to_centroid", "xp",
        "prob_win", "prob_draw", "prob_lose",
        "prob_over_2.5", "prob_under_2.5",
        "elo", "elo_opp", "elo_diff", "fixture_per_90",
        "prob_win_elo", "player_match_xg_expected", "player_match_xa_expected",
        "match_value_efficiency", "hype_vs_odds_ratio"
    ])

    pre_game_cols: list = field(default_factory=lambda: [
        "name", "element", "code", "season", "gw", "kickoff_time", "team", "opponent_team",
        "position", "was_home",
        "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend",
        "prob_win", "prob_draw", "prob_lose", "prob_over_2.5", "prob_under_2.5",
        "elo", "elo_opp", "elo_diff", "prob_win_elo", "player_match_xg_expected",
        "player_match_xa_expected", "match_value_efficiency", "hype_vs_odds_ratio"
    ])



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
