from flask import Flask, render_template, request, redirect, url_for, send_file
from zipfile import ZipFile
import tempfile
import cv2
import numpy as np
import os
import shutil
import re
from google.cloud import vision
from google.cloud.vision_v1 import types


def crop(directory):

    # Copie
    image_directory = directory
    copy_image_directory = "Images"
    shutil.copytree(image_directory, copy_image_directory)
    shutil.rmtree("Images/__MACOSX")
    new_directory = os.listdir("Images")[0]
    os.makedirs("Images/Autres")
    os.makedirs("Images/Succes")
    os.makedirs("Images/Echec")

    # Charger le modèle YOLO pré-entraîné
    net = cv2.dnn.readNet("static/Model/plaque_yolov4_29-07-23.weights", "static/Model/plaque_yolov4.cfg")
    classes = ["plaque de cadre"]

    # Récupérer les noms des couches de sortie du réseau
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    # Convertir les indices en noms de couches
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

    # Charger l'image
    directory_path = "Images/" + new_directory
    directory_list = os.listdir(directory_path)
    for f in directory_list:
        if f == ".DS_Store":
            directory_list.remove(".DS_Store")

    for image in directory_list:
        img = cv2.imread(directory_path + "/" + image)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Redimensionner l'image pour accélérer le traitement
        height, width, channels = img.shape

    # Prétraitement de l'image pour la détection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

    # Initialisation des variables pour le meilleur objet
        best_confidence = 0
        best_box = None

    # Parcourir les détections d'objets
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > best_confidence:
                    best_confidence = confidence

                # Objet détecté
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                # Coordonnées du rectangle
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    best_box = (x, y, w, h)

    # Enregistrer l'image finale dans un dossier
        if best_box is not None:
            x, y, w, h = best_box
            cropped_object = img[y:y + h, x:x + w]

        # Chemin complet de l'image de sortie
            os.makedirs("Traitement", exist_ok=True)
            os.makedirs("Traitement/Echec", exist_ok=True)
            chemin_image_sortie = "Traitement/" + image.rstrip(".jpg") + "_cropped.jpg"
        # Enregistrement de l'image de sortie
            cv2.imwrite(chemin_image_sortie, cropped_object)

        else:
            chemin_image_sortie = "Traitement/Echec/" + image
            cv2.imwrite(chemin_image_sortie, img)
    return copy_image_directory


def zip_file(src, dst):

    # Chemin du répertoire que vous souhaitez compresser
    source_directory = src
    src_files = os.listdir(src)
    src_files.remove("Autres")
    src_files.remove("Echec")
    src_files.remove("Succes")
    shutil.rmtree(src + "/" + src_files[0])

    # Chemin de l'archive ZIP que vous souhaitez créer
    zip_file_path = dst

    # Créer une archive ZIP du répertoire source
    shutil.make_archive(zip_file_path, 'zip', source_directory)
    print("zipé")


def ocr(img, key_path='static/Model/linen-epigram-232417-b3663218b851.json'):

    # Initialisation du client
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)

    # Chemin vers l'image à analyser
    image_path = img

    # Charger l'image dans un objet de type "Image"
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    # Appeler l'API pour détecter les étiquettes
    response = client.text_detection(image=image)
    texts = response.text_annotations
    image_path = image_path.rstrip(".jpg")
    json_path = image_path + ".json"
    with open(json_path, "w") as f:
        f.write(str(texts))
    return json_path


def choice(json_path):
    with open(json_path, 'r') as file:
        data_text = file.read()

    description_matches = re.findall(r'description: "(.*?)"', data_text)
    description_list = description_matches[1:]  # Excluding the first description

    filtered_description_list = [desc for desc in description_list if any(char.isdigit() for char in desc)]

    if len(filtered_description_list) > 1:
        result = "_" + str(filtered_description_list[0])
    elif len(filtered_description_list) == 1:
        result = "_" + str(filtered_description_list[0])
    else:
        result = "_erreur"
    return result


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process_folder(folder_path):
    original_directory = folder_path
    directory = crop(original_directory)
    directory_path = os.listdir(directory)
    directory_path.remove("Autres")
    directory_path.remove("Echec")
    directory_path.remove("Succes")
    directory_path = directory_path[0]

    traitement_directory = [f for f in os.listdir("Traitement") if f not in (".DS_Store", "Echec", "result.txt")]

    for image in traitement_directory:
        if image.endswith(".jpg"):
            source_path = os.path.join(directory, directory_path, f"{image.rstrip('_cropped.jpg')}.jpg")
            json_path = ocr(os.path.join("Traitement", image))
            result = choice(json_path)
            result_folder = "Echec" if result == "_erreur" else "Succes"
            result_path = os.path.join(directory, result_folder, f"{image.rstrip('_cropped.jpg')}{result}.jpg")
            os.rename(source_path, result_path)
            os.remove(os.path.join("Traitement", image))

    echec_directory = [f for f in os.listdir(os.path.join("Traitement", "Echec")) if f != ".DS_Store"]

    for image in echec_directory:
        if image.endswith(".jpg"):
            source_path = os.path.join(directory, directory_path, image)
            json_path = ocr(os.path.join("Traitement", "Echec", image))
            result = choice(json_path)
            result_folder = "Echec" if result == "_erreur" else "Autres"
            result_path = os.path.join(directory, result_folder, f"{image.rstrip('.jpg')}{result}.jpg")
            os.rename(source_path, result_path)
            os.remove(os.path.join("Traitement", "Echec", image))

    original_dirname = os.path.basename(directory_path)
    return original_dirname


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if os.path.exists("Images.zip"):
        os.remove("Images.zip")

    os.makedirs("uploads")
    if request.method == 'POST':
        uploaded_zip = request.files['zip_file']
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, uploaded_zip.filename)
                uploaded_zip.save(zip_path)
                with ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(app.config['UPLOAD_FOLDER'])
                process_folder(app.config['UPLOAD_FOLDER'])
                zip_file('Images', 'Images')
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
                return redirect(url_for('download_zip'))


@app.route('/download_zip')
def download_zip():
    shutil.rmtree("Images")
    shutil.rmtree("Traitement")
    return send_file('Images.zip', as_attachment=True)
