# Lab 3 — NLP : Classification de textes et génération avec les modèles séquentiels et Transformers

## Présentation du projet

Ce TP a pour objectif de mettre en pratique les techniques de Deep Learning appliquées au traitement automatique du langage naturel (NLP). Il porte principalement sur l'utilisation des réseaux de neurones récurrents pour la classification de textes ainsi que sur l'adaptation d'un modèle Transformer pré-entraîné pour la génération de contenu textuel.

Les expérimentations ont été réalisées sur des textes en langue arabe traitant de sujets environnementaux. Le travail est divisé en deux parties complémentaires : la classification des textes à l'aide de différentes architectures récurrentes et la génération de texte à partir d'un modèle GPT-2 adapté à notre corpus.

---

# Partie 1 — Classification des textes arabes

## 1. Description du problème

L'objectif de cette première partie est de développer des modèles capables d'analyser des paragraphes arabes et de déterminer leur catégorie en fonction de leur contenu.

Le dataset utilisé comporte trois classes :

| Classe | Signification |
|---|---|
| 0 | Texte sans rapport avec le sujet environnemental |
| 1 | Texte traitant de l'environnement de manière générale |
| 2 | Texte consacré à la pollution |

Quatre architectures de réseaux de neurones récurrents ont été expérimentées afin d'étudier leurs capacités à traiter des séquences textuelles et à reconnaître les caractéristiques pertinentes de chaque catégorie.

## 2. Préparation des données

Avant l'entraînement, les textes ont subi plusieurs transformations pour faciliter leur exploitation par les modèles.

Les opérations de prétraitement comprennent :

- Nettoyage des textes et suppression des éléments inutiles.
- Normalisation des caractères arabes.
- Tokenisation des paragraphes.
- Élimination des mots fréquents peu informatifs (stop words).
- Application de techniques de stemming et de lemmatisation.
- Conversion des mots en représentations numériques.
- Padding des séquences pour obtenir des longueurs uniformes.

Les données ont ensuite été réparties en ensembles d'entraînement, de validation et de test, afin de mesurer la capacité des modèles à traiter des textes qui n'ont pas été utilisés pendant l'apprentissage.

## 3. Architectures étudiées

### 3.1 SimpleRNN

Le SimpleRNN constitue une architecture récurrente de base permettant de traiter les mots d'une séquence dans leur ordre d'apparition.

Le modèle obtient une accuracy de **50 % sur l'ensemble de test**. Il reconnaît relativement bien les textes appartenant à la classe « Non pertinent », mais rencontre des difficultés pour distinguer les textes de la classe « Environnement général ».

Ces observations montrent que cette architecture reste limitée lorsqu'il s'agit de mémoriser des informations éloignées dans une séquence, particulièrement lorsque le volume de données d'entraînement est réduit.

### 3.2 Bidirectional RNN

Le réseau récurrent bidirectionnel a été introduit afin d'exploiter le contexte d'un texte dans les deux sens de lecture.

Cette architecture permet de prendre en compte les informations précédentes et suivantes lors de l'analyse d'une séquence. Malgré cet avantage, elle a obtenu une accuracy de **50 % sur le test**, ce qui ne représente pas une amélioration significative par rapport au SimpleRNN.

Le résultat obtenu indique que l'utilisation d'un contexte bidirectionnel ne garantit pas nécessairement de meilleures performances, notamment lorsque les données disponibles sont limitées.

### 3.3 GRU (Gated Recurrent Unit)

Le GRU utilise des mécanismes de contrôle appelés portes afin de mieux gérer les informations transmises au cours du traitement d'une séquence.

Parmi les quatre modèles évalués, il présente le meilleur résultat sur l'ensemble de test, avec une accuracy de **70,83 %**.

Cette architecture a montré de meilleures capacités de reconnaissance pour les classes « Non pertinent » et « Pollution ». Son mécanisme de gestion de l'information permet de conserver les éléments utiles tout en réduisant l'influence des informations moins importantes.

### 3.4 LSTM (Long Short-Term Memory)

Le LSTM repose sur une cellule mémoire et différents mécanismes de portes qui facilitent l'apprentissage des dépendances à long terme.

Il a obtenu **66,67 % d'accuracy sur le test** et **78,95 % sur la validation**, soit la meilleure performance de validation parmi les architectures expérimentées.

Contrairement aux autres modèles, le LSTM a réussi à reconnaître certains exemples appartenant à la catégorie « Environnement général ». Cela suggère que sa structure mémoire peut être utile pour identifier des informations contextuelles plus complexes.

## 4. Comparaison des performances

Les résultats obtenus durant les expérimentations sont résumés dans le tableau suivant :

| Architecture | Accuracy validation | Accuracy test |
|---|---:|---:|
| SimpleRNN | 68,42 % | 50,00 % |
| Bidirectional RNN | 52,63 % | 50,00 % |
| GRU | 73,68 % | **70,83 %** |
| LSTM | **78,95 %** | 66,67 % |

Le GRU se distingue par son meilleur résultat sur l'ensemble de test, tandis que le LSTM présente la meilleure accuracy sur l'ensemble de validation.

Cette différence montre que le choix d'un modèle ne doit pas se baser uniquement sur les performances de validation. L'évaluation sur des données indépendantes est également indispensable pour apprécier sa capacité de généralisation.

## 5. Analyse des résultats

Les expériences mettent en évidence des différences de comportement entre les architectures étudiées.

Le SimpleRNN et le Bidirectional RNN présentent des performances relativement limitées. Le GRU améliore les résultats sur le test, tandis que le LSTM obtient de meilleurs résultats sur la validation et parvient à détecter certains exemples d'une classe plus difficile.

Néanmoins, les performances restent influencées par plusieurs facteurs :

- Le nombre limité d'exemples dans le dataset.
- La distribution inégale des classes.
- La diversité du vocabulaire utilisé dans les textes arabes.
- La difficulté à distinguer certains paragraphes environnementaux.
- Le risque de surapprentissage pendant l'entraînement.

Les résultats doivent donc être interprétés en tenant compte de ces contraintes.

## 6. Compétences acquises

Cette première partie m'a permis de découvrir et de mettre en œuvre plusieurs techniques liées au NLP et au Deep Learning :

- Préparation et nettoyage d'un corpus de textes arabes.
- Normalisation linguistique et tokenisation.
- Utilisation des techniques de stemming et de lemmatisation.
- Conversion des textes en séquences numériques.
- Gestion des séquences de longueurs différentes avec le padding.
- Création de modèles neuronaux à l'aide de Keras.
- Utilisation des couches Embedding, SimpleRNN, Bidirectional, GRU et LSTM.
- Entraînement, validation et test de modèles de classification.
- Utilisation du Dropout et de l'EarlyStopping pour limiter le surapprentissage.
- Évaluation des performances avec des métriques de classification.
- Comparaison des architectures et interprétation des résultats expérimentaux.

---

# Partie 2 — Génération de texte avec GPT-2

## 1. Objectif

La deuxième partie du TP vise à adapter un modèle de langage pré-entraîné afin de produire du texte arabe à partir d'une phrase initiale.

Contrairement à la classification, où le modèle doit attribuer une catégorie à un texte, la génération consiste à prédire successivement les tokens suivants pour construire une continuation cohérente.

Pour cette expérimentation, nous avons utilisé un modèle GPT-2 pré-entraîné sur la langue arabe, puis réalisé un fine-tuning sur notre corpus environnemental.

## 2. Constitution du corpus

Le corpus de cette partie provient des textes arabes utilisés précédemment pour la classification.

Après vérification et nettoyage, les données retenues sont réparties de la manière suivante :

| Élément | Valeur |
|---|---:|
| Nombre total de paragraphes | 117 |
| Données d'entraînement | 105 |
| Données de validation | 12 |
| Proportion d'entraînement | 90 % |
| Proportion de validation | 10 % |

Seuls les paragraphes textuels ont été conservés. Les catégories de classification et les données issues du stemming ou de la lemmatisation ont été exclues, car le modèle de génération doit apprendre à partir de textes naturels.

Le preprocessing utilisé pour cette étape repose sur AraBERT Preprocessing afin d'adapter les textes aux particularités de la langue arabe.

## 3. Modèle pré-entraîné

Le modèle utilisé est :

**`aubmindlab/aragpt2-base`**

Il s'agit d'un modèle GPT-2 adapté à la langue arabe, basé sur l'architecture Transformer Decoder.

Caractéristiques principales de l'environnement utilisé :

| Caractéristique | Valeur |
|---|---|
| Architecture | GPT-2 |
| Modèle | AraGPT2 Base |
| Nombre de paramètres | Environ 135 millions |
| Taille du vocabulaire | 64 001 tokens après ajout du token de padding |
| Bibliothèque | Hugging Face Transformers |
| Accélération | GPU CUDA |

L'utilisation d'un modèle pré-entraîné permet de bénéficier de connaissances linguistiques déjà acquises, puis de les spécialiser sur un domaine particulier grâce au fine-tuning.

## 4. Tokenisation et préparation des séquences

Les paragraphes ont été transformés en séquences de tokens grâce au tokenizer associé au modèle AraGPT2.

Les étapes appliquées sont les suivantes :

1. Prétraitement des textes arabes.
2. Encodage des paragraphes en identifiants numériques.
3. Limitation de la longueur des séquences à 255 tokens.
4. Ajout du token de fin de séquence (EOS).
5. Gestion du padding pour faciliter le traitement par batch.

Le `DataCollatorForLanguageModeling` a été configuré avec `mlm=False`, car GPT-2 repose sur le principe du *Causal Language Modeling*.

Le modèle apprend donc à prédire le token suivant à partir des tokens précédents, sans avoir recours à un mécanisme de masquage de mots comme dans le MLM.

## 5. Configuration du fine-tuning

L'entraînement a été effectué à l'aide de l'API `Trainer` de Hugging Face.

Les paramètres utilisés sont présentés ci-dessous :

| Paramètre | Configuration |
|---|---:|
| Nombre d'epochs | 3 |
| Batch size entraînement | 2 |
| Batch size validation | 2 |
| Gradient accumulation | 4 |
| Learning rate | 0,00005 |
| Weight decay | 0,01 |
| Longueur maximale | 256 tokens |
| Precision | FP16 |
| Évaluation | À chaque epoch |
| Sélection du checkpoint | Plus faible validation loss |

La sauvegarde du meilleur modèle a été activée afin de conserver la version présentant la meilleure performance sur l'ensemble de validation.

## 6. Résultats de l'entraînement

L'évolution de la loss pendant le fine-tuning est présentée dans le tableau suivant :

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 8,7011 | **3,8676** |
| 2 | 7,4467 | 3,8949 |
| 3 | 6,4467 | 3,8977 |

La loss d'entraînement diminue progressivement, ce qui indique que le modèle s'adapte aux textes du corpus.

Cependant, la loss de validation atteint sa valeur minimale dès la première epoch, puis augmente légèrement au cours des epochs suivantes. Cette évolution peut être un signe de surapprentissage : le modèle améliore ses performances sur les données d'entraînement sans progresser sur les données de validation.

Le checkpoint de la première epoch a donc été retenu comme meilleur modèle selon le critère de validation utilisé.

## 7. Génération de texte

Après le fine-tuning, le modèle a été testé avec différentes phrases initiales en arabe.

Les paramètres de génération utilisés sont :

| Paramètre | Valeur |
|---|---:|
| `max_new_tokens` | 150 |
| `temperature` | 0,8 |
| `top_k` | 50 |
| `top_p` | 0,9 |
| `repetition_penalty` | 1,2 |
| `no_repeat_ngram_size` | 3 |

Ces paramètres permettent de contrôler la longueur du texte généré, la diversité des tokens sélectionnés et la répétition des expressions.

La température permet notamment de modifier le degré de variation des sorties : une valeur plus faible favorise des continuations plus prévisibles, tandis qu'une valeur plus élevée augmente la diversité des générations.

## 8. Observations sur les textes générés

Les tests réalisés montrent que le modèle est capable de prolonger des phrases en arabe et de produire des paragraphes contenant du vocabulaire lié à l'environnement.

Les générations obtenues présentent plusieurs caractéristiques :

- Production de continuations en langue arabe.
- Utilisation de termes et expressions associés aux problématiques environnementales.
- Construction de phrases pouvant être grammaticalement plausibles.
- Capacité à produire plusieurs phrases à partir d'un prompt court.

Certaines sorties restent toutefois imparfaites. Le modèle peut s'éloigner du sujet initial, répéter certaines expressions ou produire des informations qui ne sont pas nécessairement exactes.

Ces observations s'expliquent notamment par la taille réduite du corpus et par le fait que l'objectif du fine-tuning est d'adapter un modèle de langage, et non de lui fournir une base de connaissances vérifiée.

## 9. Limites de l'expérimentation

Le dataset de 117 paragraphes constitue une ressource limitée pour adapter un modèle possédant plusieurs millions de paramètres.

Les principales limitations rencontrées sont :

- Risque de surapprentissage.
- Faible diversité des exemples disponibles.
- Capacité de généralisation limitée.
- Possibilité de produire des répétitions.
- Difficulté à garantir la pertinence factuelle des textes générés.

Pour améliorer les résultats, il serait intéressant d'utiliser un corpus plus volumineux, d'augmenter la diversité des sources et de tester différentes stratégies d'entraînement et de génération.

---

# Conclusion générale

Ce TP m'a permis d'explorer deux approches importantes du traitement automatique du langage naturel : l'analyse de textes à travers les réseaux récurrents et la génération de contenu à l'aide d'un modèle Transformer pré-entraîné.

La première partie a montré que les architectures GRU et LSTM peuvent offrir des performances intéressantes pour la classification de textes arabes, notamment grâce à leurs mécanismes de gestion de l'information séquentielle. Le GRU a obtenu la meilleure accuracy sur l'ensemble de test, avec 70,83 %, tandis que le LSTM a atteint 78,95 % sur la validation.

La seconde partie a permis d'adapter AraGPT2 à un corpus spécialisé dans les thématiques environnementales. Malgré la taille limitée du dataset, le modèle a pu produire des continuations de textes en arabe et reproduire certains motifs linguistiques liés au domaine étudié.

Au-delà des résultats numériques, ce travail m'a permis de mieux comprendre le fonctionnement des architectures récurrentes, les différentes étapes de préparation des données textuelles, les principes du fine-tuning et les méthodes d'évaluation des modèles de Deep Learning.

Il m'a également montré l'importance de la qualité et de la quantité des données dans les performances des modèles NLP, ainsi que la nécessité d'analyser leurs résultats avec un esprit critique.

---

## Technologies utilisées

- Python
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- AraGPT2
- AraBERT Preprocessing
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Google Colab

## Structure du projet

```text
Lab-3-NLP-Classification-Task-Transformer-Text-generation/
│
├── Part1_Classification/
│   ├── Dataset
│   ├── Preprocessing
│   ├── SimpleRNN
│   ├── Bidirectional_RNN
│   ├── GRU
│   └── LSTM
│
├── Part2_Text_Generation/
│   ├── Dataset_Preparation
│   ├── GPT2_Fine_Tuning
│   └── Text_Generation
│
├── Results/
│   ├── Classification_Metrics
│   ├── Training_Curves
│   └── Generated_Text
│
└── README.md
```

## Réalisation

TP réalisé dans le cadre du module Natural Language Processing (NLP), dans le but d'étudier les modèles séquentiels et les architectures Transformer appliqués aux textes arabes.
