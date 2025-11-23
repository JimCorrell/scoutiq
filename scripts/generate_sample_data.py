"""
Generate sample data for testing the pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set random seed and create generator
rng = np.random.default_rng(42)
random.seed(42)

# Configuration
NUM_PLAYERS = 500
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_player_stats():
    """Generate sample player statistics"""

    levels = ["A", "A+", "AA", "AAA"]

    data = []

    for i in range(NUM_PLAYERS):
        player_id = f"P{i+1:04d}"

        # Demographics
        age = rng.integers(18, 28)
        level = random.choice(levels)

        # Batting stats
        pa = rng.integers(200, 600)
        ab = int(pa * 0.85)

        # Generate correlated batting stats
        true_talent = rng.normal(0.260, 0.030)
        h = int(ab * true_talent)

        doubles = int(h * 0.20)
        triples = int(h * 0.02)
        hr = rng.poisson(15)

        bb = int(pa * rng.uniform(0.08, 0.15))
        k = int(pa * rng.uniform(0.15, 0.30))
        sb = rng.poisson(10)
        cs = rng.poisson(3)

        avg = h / ab if ab > 0 else 0
        obp = (h + bb) / pa if pa > 0 else 0
        slg = (h + doubles + 2 * triples + 3 * hr) / ab if ab > 0 else 0
        ops = obp + slg

        # Position
        positions = ["C", "1B", "2B", "3B", "SS", "OF", "P"]
        position = random.choice(positions)

        # MLB projections (ground truth for training)
        # These would normally come from actual MLB performance data
        aging_factor = 1 - (age - 22) * 0.02
        mlb_avg = min(max(avg * 0.95 * aging_factor, 0.200), 0.320)
        mlb_obp = min(max(obp * 0.93 * aging_factor, 0.280), 0.420)
        mlb_slg = min(max(slg * 0.92 * aging_factor, 0.320), 0.600)
        mlb_hr = int(hr * 0.9 * aging_factor)
        mlb_sb = int(sb * 0.85 * aging_factor)
        mlb_war = rng.normal(2.0, 1.5)

        data.append(
            {
                "player_id": player_id,
                "name": f"Player {i+1}",
                "age": age,
                "level": level,
                "position": position,
                "pa": pa,
                "ab": ab,
                "h": h,
                "avg": round(avg, 3),
                "2b": doubles,
                "3b": triples,
                "hr": hr,
                "bb": bb,
                "k": k,
                "sb": sb,
                "cs": cs,
                "obp": round(obp, 3),
                "slg": round(slg, 3),
                "ops": round(ops, 3),
                # MLB projections (targets)
                "mlb_avg": round(mlb_avg, 3),
                "mlb_obp": round(mlb_obp, 3),
                "mlb_slg": round(mlb_slg, 3),
                "mlb_hr": mlb_hr,
                "mlb_sb": mlb_sb,
                "mlb_war": round(mlb_war, 1),
            }
        )

    return pd.DataFrame(data)


def generate_scouting_reports(player_stats):
    """Generate sample scouting reports"""

    tool_descriptors = {
        80: ["elite", "special", "generational"],
        70: ["plus-plus", "excellent", "outstanding"],
        60: ["plus", "above-average", "strong"],
        55: ["above-average", "solid", "good"],
        50: ["average", "adequate", "serviceable"],
        45: ["fringe-average", "borderline"],
        40: ["below-average", "limited", "weak"],
        30: ["poor", "well below-average"],
    }

    strength_words = ["displays", "shows", "exhibits", "demonstrates", "possesses"]
    concern_words = [
        "needs to improve",
        "struggles with",
        "lacks",
        "has issues with",
        "concerns about",
    ]

    reports = []

    for _, player in player_stats.iterrows():
        # Generate tool grades based on stats
        hit_grade = int(50 + (player["avg"] - 0.260) * 200)
        hit_grade = max(30, min(70, hit_grade))

        power_grade = int(40 + player["hr"] * 1.5)
        power_grade = max(30, min(70, power_grade))

        speed_grade = int(40 + player["sb"] * 1.5)
        speed_grade = max(30, min(70, speed_grade))

        # Round to nearest 5
        hit_grade = round(hit_grade / 5) * 5
        power_grade = round(power_grade / 5) * 5
        speed_grade = round(speed_grade / 5) * 5

        # Generate report text
        hit_desc = random.choice(tool_descriptors[hit_grade])
        power_desc = random.choice(tool_descriptors[power_grade])
        speed_desc = random.choice(tool_descriptors[speed_grade])

        strength = random.choice(strength_words)

        report_parts = [
            f"{player['age']}-year-old {player['position']} at {player['level']}.",
            f"{strength.capitalize()} {hit_desc} hit tool with good bat-to-ball skills.",
            f"Power is {power_desc} with {power_grade}-grade raw power.",
            f"Speed is {speed_desc} on the bases.",
        ]

        # Add concerns for weaker tools
        if hit_grade < 50:
            concern = random.choice(concern_words)
            report_parts.append(f"However, {concern} contact consistency.")

        if power_grade < 45:
            concern = random.choice(concern_words)
            report_parts.append(f"Also {concern} power development.")

        # Add positive outlook
        if player["age"] <= 22:
            report_parts.append("Still young with significant upside potential.")

        # OFP (Overall Future Potential)
        ofp = int((hit_grade + power_grade + speed_grade) / 3)
        report_parts.append(f"Overall future potential: {ofp}-grade.")

        report_text = " ".join(report_parts)

        reports.append(
            {
                "player_id": player["player_id"],
                "report_date": "2024-06-15",
                "scout_name": f"Scout {rng.integers(1, 20)}",
                "report_text": report_text,
            }
        )

    return pd.DataFrame(reports)


def main():
    """Generate and save sample data"""

    print("Generating sample player statistics...")
    player_stats = generate_player_stats()
    stats_file = OUTPUT_DIR / "prospect_stats.csv"
    player_stats.to_csv(stats_file, index=False)
    print(f"Saved {len(player_stats)} player records to {stats_file}")

    print("\nGenerating sample scouting reports...")
    scouting_reports = generate_scouting_reports(player_stats)
    reports_file = OUTPUT_DIR / "scouting_reports.csv"
    scouting_reports.to_csv(reports_file, index=False)
    print(f"Saved {len(scouting_reports)} scouting reports to {reports_file}")

    print("\nSample data generation complete!")
    print(f"\nPlayer Stats Shape: {player_stats.shape}")
    print(f"Scouting Reports Shape: {scouting_reports.shape}")

    print("\nSample Player Stats:")
    print(player_stats.head())

    print("\nSample Scouting Report:")
    print(scouting_reports.iloc[0]["report_text"])


if __name__ == "__main__":
    main()
