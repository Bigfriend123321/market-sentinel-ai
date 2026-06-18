# 📈 Market Sentinel AI

Plateforme modulaire d'**aide à la décision** pour l'analyse des marchés financiers :
suivi des cours, scoring fondamental + technique, analyse de sentiment des actualités,
moteur prédictif **probabiliste**, alertes, service d'arrière-plan et tableau de bord.

> ⚠️ **Avertissement.** Les analyses fournies sont des **aides à la décision** et ne
> constituent **pas des conseils financiers**. Toutes les estimations sont
> **probabilistes** et ne représentent jamais des certitudes. Vous restez seul
> responsable de vos décisions d'investissement.

---

## 1. Ce que le projet fait réellement (et ses limites)

Pour rester honnête et professionnel, voici la part de **réel** et la part de
**simplification** par rapport au cahier des charges :

| Fonctionnalité demandée | État dans ce projet | Remarque |
|---|---|---|
| Cours « temps réel » multi-bourses (NYSE, NASDAQ, Euronext, LSE) | ✅ via `yfinance` | Données souvent **différées (~15 min)**. Le vrai tick temps réel exige un fournisseur **payant** (Polygon, Alpaca, IEX…). La source est isolée et remplaçable. |
| Analyse fondamentale (CA, marge, P/E, P/S, dette, FCF, EPS, capitalisation, croissance) | ✅ | Données issues de Yahoo Finance, parfois incomplètes selon la valeur. |
| Score global /100 + risque + potentiel + horizon | ✅ | Moteur **transparent et explicable** (pas une boîte noire). |
| Analyse de sentiment des actualités | ✅ | Lexique financier intégré par défaut ; **FinBERT** (transformers) en option. |
| IA prédictive probabiliste (hausse / stabilité / baisse) | ✅ | Heuristique transparente par défaut ; modèle **scikit-learn** entraînable en option. |
| TOP 10 des opportunités | ✅ | Classement par score, avec justification. |
| Exécution 24h/24 en arrière-plan + démarrage Windows | ✅ | Service + tâche planifiée (`pythonw`, sans fenêtre). |
| Alertes intelligentes (notifications) | ✅ | Notifications bureau via `plyer`, repli sur logs. |
| Tableau de bord moderne | ✅ | Streamlit + Plotly (graphiques, tables, portefeuille virtuel). |
| Lecture intégrale de Reuters/Bloomberg/FT/WSJ | ⚠️ partiel | Ces sites sont **payants / sous licence**. On utilise les titres agrégés par Yahoo + (option) NewsAPI / flux RSS. Le scraping de contenus protégés n'est pas inclus. |

---

## 2. Structure des dossiers

```
market_sentinel_ai/
├── README.md
├── requirements.txt              # dépendances cœur
├── requirements-optional.txt     # FinBERT, NewsAPI, RSS… (facultatif)
├── .gitignore
│
├── config/
│   ├── config.yaml               # paramètres (intervalle, poids, seuils, logs…)
│   └── watchlist.yaml            # liste des tickers surveillés
│
├── src/market_sentinel/          # paquet Python principal
│   ├── __init__.py               # version + AVERTISSEMENT
│   ├── config.py                 # chargement de la config (accès pointé)
│   ├── logging_setup.py          # logs console + fichier rotatif
│   │
│   ├── data/
│   │   ├── market_data.py        # cours, variations, pics de volume (yfinance)
│   │   └── fundamentals.py       # indicateurs fondamentaux
│   │
│   ├── analysis/
│   │   ├── technical.py          # RSI, SMA, MACD, tendance, volatilité
│   │   ├── scoring.py            # moteur de score /100 (transparent)
│   │   └── opportunities.py      # orchestration + construction du TOP N
│   │
│   ├── news/
│   │   ├── fetcher.py            # récupération des actualités
│   │   └── sentiment.py          # sentiment (lexique ou FinBERT)
│   │
│   ├── ai/
│   │   ├── predictor.py          # scénarios probabilistes (heuristique)
│   │   └── ml_model.py           # modèle scikit-learn optionnel
│   │
│   ├── alerts/
│   │   └── notifier.py           # règles d'alerte + notifications bureau
│   │
│   ├── storage/
│   │   └── database.py           # persistance SQLite
│   │
│   └── service/
│       └── runner.py             # boucle d'arrière-plan 24h/24
│
├── dashboard/
│   └── app.py                    # tableau de bord Streamlit
│
├── scripts/
│   ├── run_service.py            # lance le service
│   ├── run_analysis_once.py      # un cycle d'analyse en console (test rapide)
│   └── install_autostart_windows.ps1  # démarrage auto Windows
│
└── tests/
    └── test_scoring.py           # tests unitaires (sans réseau)
```

---

## 3. Installation

> Prérequis : **Python 3.10+**. Sur Windows, installez-le depuis
> [python.org](https://www.python.org/downloads/) en cochant **« Add Python to PATH »**.

```powershell
# Depuis le dossier market_sentinel_ai

# 1. Créer et activer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # (Windows PowerShell)

# 2. Installer les dépendances cœur
pip install -r requirements.txt

# 3. (Optionnel) fonctionnalités avancées : FinBERT, NewsAPI, RSS
pip install -r requirements-optional.txt
```

---

## 4. Utilisation

**Le plus simple (Windows) : double-cliquer sur les lanceurs `.bat`**
- `Lancer_Dashboard.bat` — ouvre le tableau de bord dans le navigateur
- `Lancer_Analyse_Console.bat` — TOP 10 en console
- `Entrainer_IA.bat` — (ré)entraîne l'IA locale

En ligne de commande :

```powershell
# A. Test rapide : un cycle d'analyse + TOP 10 dans la console
python scripts\run_analysis_once.py

# B. Tableau de bord interactif (recommandé)
streamlit run dashboard\app.py

# C. Entraîner l'IA locale (Gradient Boosting) — à faire une fois
python scripts\train_ai.py

# D. Service d'arrière-plan 24h/24
python scripts\run_service.py

# E. Démarrage automatique avec Windows (tâche planifiée silencieuse)
.\scripts\install_autostart_windows.ps1
#   désinstallation : .\scripts\install_autostart_windows.ps1 -Remove

# F. Tests unitaires
pip install pytest
pytest -q
```

**Onglets du tableau de bord** : 🏆 Opportunités · 💎 Pépites (actions peu
chères à fort potentiel) · 🔍 Analyse entreprise (prix réel d'une action,
objectif analystes, plage 52 semaines, graphiques techniques, jauge IA) ·
📰 Actualités · 🔔 Alertes · 💼 Portefeuille · 🕓 Historique.

Personnalisez la **watchlist** dans `config/watchlist.yaml` et les **paramètres**
(intervalle, poids du score, seuils d'alerte) dans `config/config.yaml` — sans
toucher au code.

---

## 5. Explication détaillée de chaque module

- **`config.py`** — Charge `config.yaml` et `watchlist.yaml`. Accès par chemin
  pointé (`cfg.get("scoring.weights")`) et résolution de chemins relatifs.
- **`logging_setup.py`** — Logger applicatif : sortie console + fichier rotatif
  (`logs/market_sentinel.log`, 5 Mo × 5).
- **`data/market_data.py`** — Cotations (`get_quote`), historique OHLCV
  (`get_history`), détection de variations fortes et de pics de volume. Source
  yfinance **isolée** pour pouvoir brancher un fournisseur temps réel payant.
- **`data/fundamentals.py`** — Récupère les indicateurs fondamentaux et les
  normalise dans une `dataclass` (champs Optionnels, gérés en aval).
- **`analysis/technical.py`** — RSI, SMA 50/200, MACD, tendance et volatilité
  annualisée ; produit un **sous-score technique /100**. Code pur, testable.
- **`analysis/scoring.py`** — **Cœur du système.** Chaque critère → sous-score
  via des seuils lisibles, puis **moyenne pondérée renormalisée** (les données
  manquantes sont ignorées). Renvoie score global, **risque**, **potentiel**,
  **horizon** et le **détail des sous-scores** (explicabilité).
- **`news/sentiment.py`** — Sentiment d'un texte : score [-1, 1], label
  (5 niveaux), confiance, impact. Backend **lexique** (par défaut, léger) ou
  **FinBERT** (option `transformers`).
- **`news/fetcher.py`** — Récupère les actualités (yfinance), les analyse et
  **agrège** le sentiment pondéré par confiance × impact.
- **`ai/predictor.py`** — Combine score fondamental + technique + sentiment en
  un signal composite, puis **softmax à température pilotée par la volatilité**
  → probabilités **hausse / stabilité / baisse** + confiance. Toujours
  probabiliste, jamais certain.
- **`ai/ml_model.py`** — *(optionnel, scikit-learn)* régression logistique
  entraînée sur des indicateurs techniques historiques → probabilité de hausse.
  Démonstration pédagogique ; les performances passées ne préjugent pas du futur.
- **`alerts/notifier.py`** — Règle d'alerte (score + confiance) et notification
  bureau (`plyer`), avec repli sur les logs.
- **`storage/database.py`** — SQLite (standard) : tables `analyses`, `alerts`,
  `portfolio` (portefeuille virtuel).
- **`service/runner.py`** — Boucle 24h/24 : analyse la watchlist, enregistre,
  déclenche les alertes, dort entre les cycles (faible CPU) et s'arrête proprement.
- **`dashboard/app.py`** — Tableau de bord Streamlit : TOP opportunités, analyse
  détaillée (chandeliers Plotly, scénarios), actualités, alertes, portefeuille
  virtuel, historique.
- **`analysis/opportunities.py`** — Orchestrateur : enchaîne tous les modules
  pour produire un objet `Analysis` complet et construire le TOP N.

---

## 6. API gratuites recommandées

| Besoin | Service | Clé requise ? | Notes |
|---|---|---|---|
| Cours & fondamentaux | **Yahoo Finance** (`yfinance`) | Non | Utilisé par défaut. Données différées. |
| Actualités générales | **NewsAPI.org** | Oui (gratuite) | 100 req/jour en plan gratuit. |
| Cours temps réel (US) | **Alpaca Markets** | Oui (gratuite) | Flux temps réel pour titres US, compte gratuit. |
| Cours / agrégats | **Polygon.io** | Oui (gratuite limitée) | Plan gratuit limité en débit. |
| Fondamentaux | **Financial Modeling Prep** | Oui (gratuite limitée) | États financiers détaillés. |
| Cours & FX | **Alpha Vantage** | Oui (gratuite) | 25 req/jour, simple à intégrer. |
| Flux d'actualités | **RSS** (`feedparser`) | Non | Flux publics des éditeurs (titres). |

> Les contenus intégraux de Reuters, Bloomberg, FT et WSJ sont **sous licence
> payante** : ne scrapez pas ces sites. Préférez leurs API officielles ou les
> titres/flux autorisés.

Stockez vos clés dans `config/config.yaml` (ou un `config/secrets.yaml` ignoré
par git) — **ne les committez jamais**.

---

## 7. Améliorations futures possibles

1. **Vrai temps réel** : brancher Alpaca/Polygon (WebSocket) derrière l'interface
   de `market_data.py`.
2. **Modèles avancés** : LSTM/Transformers (PyTorch/TensorFlow) pour les séries
   temporelles ; backtesting rigoureux avant toute mise en production.
3. **FinBERT activé** par défaut pour un sentiment plus fin (GPU recommandé).
4. **Backtesting & métriques** : Sharpe, drawdown, hit-rate, walk-forward.
5. **Alertes multi-canaux** : e-mail, Telegram, webhook Discord/Slack.
6. **API REST** (FastAPI) pour exposer les analyses à d'autres clients.
7. **Conteneurisation** (Docker) et planification (APScheduler/cron).
8. **Cache & limites de débit** pour respecter les quotas d'API.
9. **Internationalisation** de l'interface et fuseaux horaires des marchés.
10. **Tests d'intégration** avec données mockées + CI.

---

## 8. Licence & responsabilité

Projet fourni à des fins **éducatives**. Aucune garantie de performance.
**Les analyses sont des aides à la décision et ne constituent pas des conseils
financiers.** Vérifiez les conditions d'utilisation de chaque source de données.
