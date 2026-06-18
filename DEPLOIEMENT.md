# 📱 Mettre Market Sentinel AI sur ton iPhone (comme une app)

Objectif : une **icône sur l'écran d'accueil** qui ouvre l'app **24/7, même PC
éteint, même en 4G/5G**. Pour ça, on héberge l'app gratuitement sur
**Streamlit Community Cloud**, puis on l'**ajoute à l'écran d'accueil**.

Le code est déjà prêt et committé en local (dépôt git initialisé). Il te reste
3 étapes.

---

## Étape 1 — Mettre le code sur GitHub (gratuit)

1. Crée un compte sur **https://github.com** (si pas déjà fait).
2. Clique sur **New repository** (bouton vert).
   - Nom : `market-sentinel-ai`
   - Visibilité : **Private** (recommandé) ou Public.
   - **Ne coche RIEN** (pas de README, pas de .gitignore) — le projet en a déjà.
   - Clique **Create repository**.
3. GitHub affiche une page « …or push an existing repository ». Copie les 2-3
   lignes proposées et colle-les dans **PowerShell**, dans le dossier du projet :

```powershell
cd "C:\Users\ardau\OneDrive\Documents\site_web\market_sentinel_ai"
git remote add origin https://github.com/TON_PSEUDO/market-sentinel-ai.git
git branch -M main
git push -u origin main
```

> Au premier `push`, une fenêtre te demandera de te connecter à GitHub
> (navigateur). Accepte → le code part sur GitHub.

---

## Étape 2 — Déployer sur Streamlit Community Cloud (gratuit)

1. Va sur **https://share.streamlit.io** → **Sign in with GitHub** → autorise.
2. Clique **Create app** → **Deploy a public app from GitHub** (ou « I have an app »).
3. Renseigne :
   - **Repository** : `TON_PSEUDO/market-sentinel-ai`
   - **Branch** : `main`
   - **Main file path** : `dashboard/app.py`
4. Clique **Deploy**. La première construction prend 3-6 min (installation des
   bibliothèques). Ensuite tu obtiens une **adresse publique** du type :
   `https://market-sentinel-ai-xxxx.streamlit.app`

---

## Étape 3 — Ajouter à l'écran d'accueil de l'iPhone

1. Ouvre cette adresse dans **Safari** sur ton iPhone.
2. Touche le bouton **Partager** (carré avec flèche vers le haut).
3. Choisis **« Sur l'écran d'accueil »** → **Ajouter**.
4. 🎉 Une icône apparaît : tu l'ouvres comme n'importe quelle app, en plein écran.

---

## ⚠️ À savoir (honnêtement)

- **URL publique sans mot de passe** : toute personne ayant le lien peut ouvrir
  l'app. Aucune donnée sensible n'y est (pas de clé API, pas de compte), et on
  ne passe aucun ordre de bourse — mais garde le lien pour toi.
- **Portefeuille & historique non permanents** : sur l'offre gratuite, le
  stockage est éphémère et peut se réinitialiser quand l'app redémarre.
  L'analyse en direct, elle, fonctionne toujours (données fraîches).
- **1er chargement lent** : l'app scanne ~60 valeurs ; compte ~30-60 s la
  première fois, puis c'est mis en cache.
- **yfinance** peut occasionnellement être limité par Yahoo depuis un serveur
  cloud (données momentanément vides) — réessayer suffit en général.
- **Réentraîner l'IA** : pour mettre à jour le modèle déployé, relance
  `Entrainer_IA.bat` en local puis `git add data/predictor.joblib && git commit
  -m "maj IA" && git push` — Streamlit redéploie tout seul.

---

## Alternative rapide (si tu veux juste tester depuis ton iPhone à la maison)

Sans cloud, sur le **même Wi-Fi**, PC allumé et serveur lancé :
1. Trouve l'IP locale du PC : `ipconfig` → ligne « Adresse IPv4 » (ex. 192.168.0.159).
2. Sur l'iPhone (même Wi-Fi), ouvre `http://192.168.0.159:8501`.
3. La 1re fois, autorise l'app dans le **pare-feu Windows** si demandé.
