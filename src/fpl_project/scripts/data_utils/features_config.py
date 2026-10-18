from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesConfig:
    ### LEAKAGE COLUMNS - just for ema
    target: list = field(default_factory=lambda: [
        "total_points"
    ])

    id_cols: list = field(default_factory=lambda: [
        "name", "element", "code", "season", "gw", "kickoff_time",  "team", "opponent_team", "is_played"
    ])

    static_cols: list = field(default_factory=lambda: [
        "position", "was_home"
    ])
    ### LEAKAGE COLUMNS
    fpl_cols: list = field(default_factory=lambda: [
        "total_points", "xp", "bonus", "bps", "ict_index", "influence", "creativity", "threat"
    ])

    ### LEAKAGE COLUMNS
    market_cols: list = field(default_factory=lambda: [
        "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend"
    ])

    ### LEAKAGE COLUMNS
    match_context_cols: list = field(default_factory=lambda: [
        "starts", "minutes", "goals_scored", "assists", "own_goals", "clean_sheets", "goals_conceded", 
        "saves", "penalties_missed", "penalties_saved", "yellow_cards", "red_cards", "team_h_score", "team_a_score",
        "xg", "xa", "xg_involvements", "xg_conceded", "xg_per_90", "xa_per_90", "xg_involvements_per_90", "xg_conceded_per_90",
        "player_match_xg_expected", "player_match_xa_expected", "match_value_efficiency", "hype_vs_odds_ratio"
    ])

    betting_cols: list = field(default_factory=lambda: [
        "prob_win", "prob_draw", "prob_lose", "ahh", "prob_over_2.5", "prob_under_2.5"
    ])

    @property
    def LEAKAGE_COLS(self) -> list:
        return list(set(self.fpl_cols + self.match_context_cols + self.market_cols + self.target))




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
