import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder


def preprocess_and_save_images_MiniLotCompresse(dossier, batch_size=10):
    def preprocess_image(img_path):
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"L'image {img_path} n'a pas pu être chargée.")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        resizing_layer = tf.keras.layers.Resizing(224, 224)
        resized_img = resizing_layer(img_tensor)
        rescaling_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        normalized_img = rescaling_layer(resized_img)
        print(f"Image {img_path} dimensions: {normalized_img.shape}")
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
        preprocess_img = preprocess_image(img_path)
        preprocess_images.append(preprocess_img)

        if (i + 1) % batch_size == 0 or i == len(images) - 1:
            batch_filename = os.path.join(dossier, f'preprocessed_batch_{i // batch_size + 1}.npy')
            np.save(batch_filename, preprocess_images)
            preprocess_images = []
    
    return preprocess_images

def charger_images_par_morceaux(dossier):
    fichiers = [f for f in os.listdir(dossier) if f.startswith('preprocessed_batch_') and f.endswith('.npy')]
    toutes_les_images_traite = None

    for fichier in fichiers:
        chemin_fichier = os.path.join(dossier, fichier)
        images_batch = np.load(chemin_fichier, allow_pickle=True)
        print(f"Dimensions des images dans {chemin_fichier}: {images_batch.shape}")
        
        if toutes_les_images_traite is None:
            toutes_les_images_traite = images_batch
        else:
            toutes_les_images_traite = np.concatenate((toutes_les_images_traite, images_batch), axis=0)

    print(f"Dimensions finales de toutes les images chargées : {toutes_les_images_traite.shape}")
    return toutes_les_images_traite

def JeSetCreate2(images, annotations, label_columns):
    
    # Diviser les indices pour les jeux d'entraînement, de validation et de test
    train_indices, temp_indices = train_test_split(range(len(images)), test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Créer les ensembles d'images
    X_train = images[train_indices]
    X_val = images[val_indices]
    X_test = images[test_indices]
    
    # Créer les ensembles d'étiquettes
    y_train = annotations.iloc[train_indices][label_columns].values
    y_val = annotations.iloc[val_indices][label_columns].values
    y_test = annotations.iloc[test_indices][label_columns].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# X_train : Données d'entraînement
# y_train : Étiquettes d'entraînement
# X_val : Données de validation
# y_val : Étiquettes de validation
# X_test : Données de test
# y_test : Étiquettes de test

def create_vgg19(num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Dossier où sont stockées les images
dossier = input("Entrer le nom du fichier contenant vos images:\n") 

print("Voulez-vous prétraiter vos images ou charger les images prétraitées ?")
reponse = input("Prétraiter ou Charger\n:").lower()
if reponse == "pretraiter" or reponse == "prétraiter":
    print("Prétraitement ")
    preprocess_and_save_images_MiniLotCompresse(dossier, batch_size=10)
    images = charger_images_par_morceaux(dossier)
    print(f"Dimensions finales des images prétraitées et chargées : {images.shape}")
else:
    print("Chargement")
    images = charger_images_par_morceaux(dossier)
    print(f"Dimensions finales des images chargées : {images.shape}")

# Chemin vers le fichier d'annotations
annotation_file = input("Entrez le chemin d'accès vers le fichier contenant les étiquettes (list_attr_celeba.csv):\n") 
annotations = pd.read_csv(annotation_file)
label_columns = annotations.columns[1:]  # La première colonne est 'image_id'

# Appeler la fonction avec les annotations et les colonnes d'étiquettes
X_train, y_train, X_val, y_val, X_test, y_test = JeSetCreate2(images, annotations, label_columns)

# Vérifier les types de colonnes après la conversion
print("Dimension apres le découpage des données")
print("Train:\n", X_train.shape)
print("Val:\n", X_val.shape)
print("Test:\n", X_test.shape)

# Créer un OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Ajuster et transformer les étiquettes d'entraînement
y_train_encoded = encoder.fit_transform(y_train)

# Transformer les étiquettes de validation et de test
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)

# Vérifier les formes des étiquettes encodées
print(f"Forme des étiquettes d'entraînement encodées : {y_train_encoded.shape}")
print(f"Forme des étiquettes de validation encodées : {y_val_encoded.shape}")
print(f"Forme des étiquettes de test encodées : {y_test_encoded.shape}")


num_classes = 40  # Modifier en fonction de votre dataset

# Créer le modèle VGG19
model = create_vgg19(num_classes)
model.summary()

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(f"Modèle compilé : optimizer='adam'  loss='categorical_crossentropy'  metrics=['accuracy']")

# Définir le callback de checkpoint
checkpoint = ModelCheckpoint('mon_modele_vgg19_checkpoint.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

try:
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=60, 
        callbacks=[checkpoint, early_stopping],
        #workers=5,  # Nombre de threads pour le chargement des données
        #use_multiprocessing=True,
        batch_size=32)
except ValueError as e:
    print(e)

model.save('mon_modele_vgg19.keras')
print("Modèle sauvegardé sous le nom 'mon_modele_vgg19.keras'")

# Charger le modèle sauvegardé
modele_charge = load_model('mon_modele_vgg19.keras')
print("Modèle chargé depuis 'mon_modele_vgg19.keras'")


evaluation = modele_charge.evaluate(X_test, y_test)
print(f"Évaluation sur le jeu de test : Loss = {evaluation[0]}, Accuracy = {evaluation[1]}")

y_pred = model.predict(X_test)
results = X_test[['image_id']].copy()
results['predictions'] = y_pred.argmax(axis=1)
print(results)