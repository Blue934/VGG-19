import os
import cv2
import tensorflow as tf
import numpy as np

def preprocess_and_save_images(dossier, batch_size=10):
    def preprocess_image(img_path):
        # Lire l'image avec OpenCV
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"L'image {img_path} n'a pas pu être chargée.")
            return None
        # Convertir l'image en RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convertir l'image en tensor TensorFlow
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        # Redimensionner l'image à 224x224 pixels
        resizing_layer = tf.keras.layers.Resizing(224, 224)
        resized_img = resizing_layer(img_tensor)
        # Normaliser les valeurs des pixels (mettre à l'échelle dans la plage [-1, 1])
        rescaling_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        normalized_img = rescaling_layer(resized_img)
        return normalized_img

    fichiers = os.listdir(dossier)
    images = [f for f in fichiers if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    print("Nb d'images:", len(images))
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return None
    else:
        print(f"Les images sont bien présentes dans le dossier {dossier}")
    
    preprocess_images = []
    for i, image in enumerate(images):
        img_path = os.path.join(dossier, image)
        print(f"Traitement de l'image {img_path}...")
        preprocess_img = preprocess_image(img_path)
        preprocess_images.append(preprocess_img)

        if (i + 1) % batch_size == 0 or i == len(images) - 1:
            # Compresser les images de 10 %
            compressed_images = [tf.image.resize(img, [int(img.shape[0] * 0.9), int(img.shape[1] * 0.9)]) for img in preprocess_images]
            
            batch_filename = os.path.join(dossier, f'preprocessed_batch_{i // batch_size + 1}.npy')
            print(f"Sauvegarde du mini-lot dans {batch_filename}...")
            np.save(batch_filename, compressed_images)
            preprocess_images = []
    
    print("Images redimensionnées avec succès !")
    print("Images normalisées dans une plage d'indices comprise entre -1 et 1 avec succès !")
    print("Étape de prétraitement accomplie avec succès !") 
    return preprocess_images
    


def charger_images_par_morceaux(dossier, batch_size=10):
    fichiers = [f for f in os.listdir(dossier) if f.startswith('preprocessed_batch_') and f.endswith('.npy')]
    toutes_les_images_traite = []

    for fichier in fichiers:
        chemin_fichier = os.path.join(dossier, fichier)
        print(f"Chargement du fichier {chemin_fichier}...")
        images_batch = np.load(chemin_fichier, allow_pickle=True)
        
        # Convertir le batch d'images en une liste
        images_batch_list = images_batch.tolist()
        
        # Ajouter les images du batch à la liste totale
        toutes_les_images_traite.extend(images_batch_list)
    
    return toutes_les_images_traite

# Dossier où sont stockées les images
dossier = 'Deepenv\img_align_celeba' 




print("Voulez-vous prétraiter vos images ou charger les images préetratitées ?")
reponse = input("Pretraiter ou Charger\n:").lower()
if reponse == "pretraiter" or reponse == "prétraiter":
    print("Pretraitement")
    Data = preprocess_and_save_images(dossier, batch_size=10)
    print("Data:\n", Data)
    print("Taille de Data :\n", len(Data))
else:
    print("Chargement")
    images = charger_images_par_morceaux(dossier, batch_size=10)
    print(f"Nombre total d'images chargées : {len(images)}")

