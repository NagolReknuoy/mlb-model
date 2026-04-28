import json

with open("mlb_odds_2026.json") as f:
    data = json.load(f)

date = list(data.keys())[0]
print(f"Date: {date}")
for game in data[date][:3]:
    gv = game["gameView"]
    print(f"\n{gv.get('awayTeam',{}).get('fullName')} @ {gv.get('homeTeam',{}).get('fullName')}")
    for bet_type, bets in game.get("odds", {}).items():
        if bets:
            print(f"  {bet_type}: {bets[0].get('currentLine')}")