# VisionAI 🧠🦀

**VisionAI** est un projet académique développé dans le cadre du **Projet Annuel (PA)**.  
L’objectif est de concevoir **un moteur de modèles de Machine Learning from scratch en Rust**, sans utiliser de bibliothèques ML existantes, puis de le rendre exploitable via **Python** et une **API REST**.

Le projet met l’accent sur :
- la compréhension mathématique des modèles
- une architecture logicielle propre et modulaire
- la performance et la sécurité offertes par Rust

---

## 🎯 Objectifs du projet

- Implémenter des modèles de Machine Learning à partir de zéro
- Séparer clairement mathématiques, modèles et algorithmes d’optimisation
- Fournir une interface Python pour l’expérimentation et la visualisation
- Exposer les modèles via une API REST pour des cas d’usage concrets

---

## 🧱 Architecture du projet

Le projet est organisé sous forme de **workspace Rust**, composé de plusieurs crates indépendantes mais interconnectées.

```txt
VisionAI/
├── Cargo.toml              # Workspace Rust
├── README.md
├── .gitignore
│
├── core_lib/               # Cœur du moteur ML (Rust)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── math/           # Outils mathématiques
│       │   ├── mod.rs
│       │   ├── vector.rs
│       │   └── matrix.rs
│       ├── models/         # Modèles de Machine Learning
│       │   ├── mod.rs
│       │   ├── linear.rs   # Modèle linéaire
│       │   ├── mlp.rs      # Perceptron Multi-Couches
│       │   └── rbf.rs      # Réseau à fonctions de base radiale
│       └── optim/          # Algorithmes d’optimisation
│           ├── mod.rs
│           └── gradient_descent.rs
│
├── python_binding/         # Wrapper Python (PyO3)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs
│
├── api_server/             # API REST (Rust)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── notebooks/              # Jupyter Notebooks (tests & analyses)
│   ├── analyse_dataset.ipynb
│   └── test_linear.ipynb
│
├── datasets/               # Jeux de données (non versionnés)
│
└── client_app/             # Application cliente (web / mobile / Unity)
