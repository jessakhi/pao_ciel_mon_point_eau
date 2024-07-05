# PAO_Ciel_Mon_Point_Deau

## Modèle
Le notebook U-net2.ipynb extrait les données, prépare le modèle, lance l'apprentissage et les prédictions, ainsi que le seuillage des résultats.
Tout est paramètrable au maximum, les fonctions utilisées dans le notebook sont définies dans le module unet2.py

### archive
Ce dossier contient les essais précédents du modèle

### runs
Ce dossier contient certains résultats de runs effectués, ainsi que le meilleur modèle pour un apprentissage.

## Données

Il manque les données dans ce git
Pour fonctionner il faut en local une arborescence des data telles que :

- data_folder
    - data_folder
        - masks
        - S1
        - S2

## Résultats
En local, les résultats sont enregistrés dans 
- results
    - graphs(nImagesTrain, nEpoch, batch_size, '%H-%M')
        - accuracy.png
        - loss.png
        - histograms.png
        - Y_pred.png
        - Y_pred_treshold.png

## Rapport

Le rapport est disponible ici :
**Lien overleaf**