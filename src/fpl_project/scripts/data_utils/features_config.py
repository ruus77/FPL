from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesConfig:
    # Prediction target
    target: list = field(default_factory=lambda: [
        "total_points"
    ])

    id_cols: list = field(default_factory=lambda: [
        "name", "element", "code", "season", "gw", "kickoff_time",  "team", "opponent_team", "is_played"
    ])

    static_cols: list = field(default_factory=lambda: [
        "position", "was_home"
    ])
    # Post-match columns (fpl, market, match context): used only as shifted features (ema/lag/std)
    fpl_cols: list = field(default_factory=lambda: [
        "total_points", "xp", "bonus", "bps", "ict_index", "influence", "creativity", "threat"
    ])

    market_cols: list = field(default_factory=lambda: [
        "value", "selected", "transfers_in", "transfers_out", "transfers_balance", "transfers_trend"
    ])

    match_context_cols: list = field(default_factory=lambda: [
        "minutes", "starts", "goals_scored", "assists", "own_goals", "clean_sheets", "goals_conceded", 
        "saves", "penalties_missed", "penalties_saved", "yellow_cards", "red_cards", "team_h_score", "team_a_score",
        "xg", "xa", "xg_involvements", "xg_conceded", "xg_per_90", "xa_per_90", "xg_involvements_per_90", "xg_conceded_per_90"
    ])

    betting_cols: list = field(default_factory=lambda: [
        "prob_win", "prob_draw", "prob_lose", "ahh", "prob_over_2.5", "prob_under_2.5"
    ])

    @property
    def LEAKAGE_COLS(self) -> list:
        return list(dict.fromkeys(self.fpl_cols + self.market_cols + self.match_context_cols + self.target))
