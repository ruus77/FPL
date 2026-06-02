import matplotlib.pyplot as plt
import seaborn as sns
from fpl_project.scripts.data_utils.colors_config import team_colors  # <-- TUTAJ DODANO .scripts
import plotly.io as pio
pio.renderers.default = "png"
import pandas as pd
import plotly.express as px
import textwrap


def best_players_plots(df: pd.DataFrame, num_players: int, season: int) -> None:
    best_df = (df[df.season == season]
               .groupby(["name", "team_name"])["total_points"].sum()
               .reset_index()
               .sort_values("total_points", ascending=False)
               .head(num_players))

    plt.figure(figsize=(12, 6))
    ax1 = sns.barplot(data=best_df, x="total_points", y="name", hue="team_name",
                      palette=team_colors, dodge=False, legend=False)
    for container in ax1.containers:
        ax1.bar_label(container)
    plt.title(f"Top {num_players} zawodników - Suma punktów ({season})")
    plt.show()

    plt.figure(figsize=(12, 6))

    line_data = (df[(df.season == season) & (df.name.isin(best_df["name"]))]
                 .sort_values(["name", "gw"])
                 .assign(cum_pts=lambda x: x.groupby("name")["total_points"].cumsum()))

    player_to_team = dict(zip(best_df.name, best_df.team_name))
    player_palette = {name: team_colors.get(player_to_team[name], "#808080") for name in best_df.name}

    sns.lineplot(data=line_data, x="gw", y="cum_pts", hue="name",
                 palette=player_palette,
                 marker="o")

    plt.title(f"Skumulowana suma punktów najlepszych piłkarzy ({season})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Zawodnik")
    plt.grid(True, alpha=0.2)
    plt.show()


def expected_stats_vs_actuals(df: pd.DataFrame, teams: pd.DataFrame, season: int,
                              expected_stat: str, actual_stat: str, position: str | list[str]) -> plt.Figure:
    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex=True)
    pos_list = position if isinstance(position, list) else [position]

    for i in range(3):
        for j, team in enumerate([teams.iloc[i, 0], teams.iloc[-(i + 1), 0]]):
            plot_data = df[(df.team_name == team) & (df.season == season) & (df.position.isin(pos_list))]
            melted_data = plot_data[[expected_stat, actual_stat]].melt()

            sns.barplot(data=melted_data,
                        y="variable",
                        x="value",
                        estimator="sum",
                        errorbar=None,
                        ax=ax[i, j],
                        palette=["#d3d3d3", "#ff7f0e"]
                        )

            ax[i, j].set_title(f"{team}", fontweight='bold')
            ax[i, j].set_ylabel("")

            ax[i, j].set_xlabel("Suma statystyki" if i == 2 else "")

            for container in ax[i, j].containers:
                ax[i, j].bar_label(container, fmt='%.1f', padding=5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"{expected_stat} vs {actual_stat} dla pozycji {', '.join(pos_list)} (Sezon: {season})", fontsize=16)

    return fig





def radar_plot_x_stats(df: pd.DataFrame, players: list[str], stats: list[str], season: int|list[int]=2526):
    if isinstance(season, int) and season:
        season = [season]
    if not players:
        return df

    df_season = df[(df.season.isin(season)) & (df.name.isin(players))]

    last_gw = df_season['gw'].max()
    df_filtered = df_season[df_season['gw'] == last_gw][['name'] + stats].reset_index(drop=True)

    df_melted = pd.melt(
        df_filtered,
        id_vars=['name'],
        value_vars=stats,
        var_name="stats",
        value_name='value'
    )

    main_title = "STATYSTYKI OCZEKIWANE"
    players_str = ', '.join(players)
    wrapped_players = "<br>".join(textwrap.wrap(players_str, width=60))
    season_str = f"Sezon: {', '.join(map(str, season))} | GW: {last_gw}"
    final_title_text = f"{main_title}<br><span style='font-size:16px; color:#CCCCCC;'>{wrapped_players}<br>{season_str}</span>"

    neon_colors = ["#00F0FF", "#FF0055", "#FFE600", "#00FF66", "#B400FF"]

    fig = px.line_polar(
        df_melted,
        r='value',
        theta='stats',
        color='name',
        line_close=True,
        markers=True,
        template=None,
        color_discrete_sequence=neon_colors
    )

    fig.update_traces(
        fill='toself',
        opacity=0.35,
        line=dict(width=3.5),
        marker=dict(size=9)
    )

    fig.update_layout(
        width=950,
        height=850,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#F0F0F0", family="Arial, sans-serif", size=14),
        title=dict(
            text=final_title_text,
            x=0.5,
            xanchor='center',
            y=0.97,
            yanchor='top',
            font=dict(size=26)
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(color="#F0F0F0", size=15)
        ),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                showline=False,
                gridcolor="#555555",
                tickfont=dict(color='#A0A0A0', size=12)
            ),
            angularaxis=dict(
                gridcolor="#555555",
                linecolor="#EEEEEE",
                tickfont=dict(color='#FFFFFF', size=15)
            )
        ),
        margin=dict(t=160, b=100, l=80, r=80)
    )

    return fig