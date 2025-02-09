import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf





def preprocess_and_display_images(dossier):
    """
    Pré-traite et affiche toutes les images d'un dossier en utilisant Matplotlib.
    
    Parameters:
    dossier (str): Chemin du dossier contenant les images.
    """
    def preprocess_image(img_path):
        """
        Pré-traite une image pour la reconnaissance faciale avec VGG19 et TensorFlow/Keras.
        
        Parameters:
        img_path (str): Chemin de l'image à traiter.
        
        Returns:
        tuple: Image redimensionnée et normalisée prête pour l'entrée dans VGG19.
        """
        # Lire l'image avec OpenCV
        img = cv2.imread(img_path, 1)
        
        # Vérifier que l'image a été chargée correctement
        if img is None:
            print(f"L'image {img_path} n'a pas pu être chargée.")
            return None, None
        
        # Convertir l'image en RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convertir l'image en tensor TensorFlow
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        
        # Redimensionner l'image à 224x224 pixels
        resizing_layer = tf.keras.layers.Resizing(224, 224)
        resized_img = resizing_layer(img_tensor)
        print("Image redimensionnée avec succès !")
        
        # Normaliser les valeurs des pixels (mettre à l'échelle dans la plage [-1, 1])
        rescaling_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        normalized_img = rescaling_layer(resized_img)
        print("Image normalisée dans une plage d'indices comprise entre -1 et 1 avec succès !")
        
        return resized_img, normalized_img
    
    # Lister tous les fichiers du dossier
    fichiers = os.listdir(dossier)
    
    # Filtrer pour garder uniquement les fichiers d'images
    images = [f for f in fichiers if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return
    else:
        print(f"Les images sont bien présentes dans le dossier {dossier}")
        print("Voici le type de données :", type(images))  # Vérification du type de données
    
    print("Combien d'images souhaitez-vous voir ?")
    reponse2 = input("Total / Plusieurs / Aucune\n:").lower()
    if reponse2 == "total":
        for i, image in enumerate(images):
            img_path = os.path.join(dossier, image)
            preprocessed_img_affichage, preprocess_img = preprocess_image(img_path)
            
            if preprocessed_img_affichage is None:
                continue
            plt.figure(figsize=(6, 6))
            plt.imshow(preprocessed_img_affichage.numpy().astype("uint8"))
            plt.axis('off')
            plt.title(image)
            plt.show()
    
    elif reponse2 == "plusieurs":
        print("Quel nombre exactement ?")
        nb = int(input())
        for i in range(nb):
            print(images)
            print("Quelle image souhaitez-vous afficher ?")
            reponse3 = input()
            img_path = os.path.join(dossier, reponse3)
            preprocessed_img_affichage, preprocess_img = preprocess_image(img_path)
            
            if preprocessed_img_affichage is None:
                continue
            
            plt.figure(figsize=(6, 6))
            plt.imshow(preprocessed_img_affichage.numpy().astype("uint8"))
            plt.axis('off')
            plt.title(reponse3)
            plt.show()
            
    elif reponse2 == "aucune":
        print("Êtes-vous certain de ne pas vouloir afficher d'images ?")
        reponse4 = input("Oui / Non ?\n:").lower()
        if reponse4 == "oui":
            for i, image in enumerate(images):
                img_path = os.path.join(dossier, image)
                preprocessed_img_affichage, preprocess_img = preprocess_image(img_path)
                print(f"Image ({i}) traité")
    
    return preprocess_img


# Exemple d'utilisation
dossier_images = input("Chemin vers le dossier contenant les images :\n")  # r"c:\Users\oreni\Desktop\deeplearning\dp3\Protocole\img_align_celeba"
Data = preprocess_and_display_images(dossier_images)
print("Données prêtes pour le VGG-19 :", Data)
if Data is not None:
    print("Taille du jeu de données :", len(Data))
