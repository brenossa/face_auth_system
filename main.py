import cv2
import numpy as np
import face_recognition
import os
import pickle
import time


class FaceAuthorizationSystem:
    """
    Sistema de autorização por reconhecimento facial
    """

    def __init__(
        self,
        images_folder="authorized_faces",
        access_logs_folder="access_logs",
        encodings_file="authorized_faces.pkl",
        resize_factor=0.25,
        tolerance=0.6,
    ):
        """
        Inicializa o sistema de autorização facial

        Args:
            images_folder (str): Pasta com imagens de faces autorizadas
            access_logs_folder (str): Pasta para salvar logs de acesso
            encodings_file (str): Arquivo para armazenar/carregar encodings
            resize_factor (float): Fator de redimensionamento para processamento
            tolerance (float): Tolerância para comparação de faces (0.6 é o padrão)
        """
        # Configurações
        self.images_folder = images_folder
        self.access_logs_folder = access_logs_folder
        self.encodings_file = encodings_file
        self.resize_factor = resize_factor
        self.tolerance = tolerance
        self.authorized_encodings = []
        self.names = []

    def load_images(self):
        """Carrega as imagens da pasta e extrai os nomes das pessoas autorizadas."""
        images = []
        names = []

        try:
            # Cria a pasta se não existir
            os.makedirs(self.images_folder, exist_ok=True)

            files = os.listdir(self.images_folder)

            for file in files:
                # Verifica se é uma imagem
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(self.images_folder, file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        # Já redimensiona a imagem para salvar memória
                        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                        images.append(img)
                        # Extrai o nome sem a extensão
                        names.append(os.path.splitext(file)[0])

            print(f"Carregadas {len(images)} imagens de pessoas autorizadas.")
            return images, names

        except Exception as e:
            print(f"Erro ao carregar imagens: {e}")
            return [], []

    def generate_encodings(self, images, names):
        """Gera as codificações faciais para todas as imagens carregadas."""
        encodings = []
        valid_names = []

        print("Gerando encodings faciais...")
        start_time = time.time()

        for img, name in zip(images, names):
            try:
                # Converte para RGB (face_recognition usa RGB)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Encontra localizações de rostos
                face_locations = face_recognition.face_locations(rgb_img, model="hog")

                if face_locations:
                    # Se encontrou rosto, gera o encoding
                    face_encoding = face_recognition.face_encodings(
                        rgb_img, face_locations
                    )[0]
                    encodings.append(face_encoding)
                    valid_names.append(name)
                else:
                    print(f"Aviso: Nenhum rosto encontrado na imagem de '{name}'")

            except Exception as e:
                print(f"Erro ao processar a imagem '{name}': {e}")

        elapsed = time.time() - start_time
        print(f"Encodings gerados em {elapsed:.2f} segundos.")

        return valid_names, encodings

    def load_or_generate_encodings(self):
        """Carrega encodings de um arquivo ou gera novos se o arquivo não existir."""
        if os.path.exists(self.encodings_file):
            print(f"Carregando encodings do arquivo {self.encodings_file}")
            try:
                with open(self.encodings_file, "rb") as f:
                    data = pickle.load(f)
                    print(
                        f"Carregados {len(data['encodings'])} encodings de pessoas autorizadas."
                    )
                    return data["names"], data["encodings"]
            except Exception as e:
                print(f"Erro ao carregar encodings: {e}")

        # Se não foi possível carregar, gera novos
        images, names = self.load_images()
        valid_names, encodings = self.generate_encodings(images, names)

        # Salva para uso futuro
        try:
            with open(self.encodings_file, "wb") as f:
                pickle.dump({"names": valid_names, "encodings": encodings}, f)
            print(f"Encodings salvos em {self.encodings_file}")
        except Exception as e:
            print(f"Erro ao salvar encodings: {e}")

        return valid_names, encodings

    def capture_face(self, camera_id=0, timeout=10, attempts=3):
        """
        Captura uma imagem da câmera e tenta encontrar um rosto

        Args:
            camera_id: ID da câmera a ser usada (0 é a padrão)
            timeout: Tempo máximo para captura (em segundos)
            attempts: Número máximo de tentativas

        Returns:
            tuple: (sucesso, frame_com_rosto, posição_do_rosto) ou (False, None, None) se falhar
        """
        print("Iniciando captura de imagem...")

        # Abre a câmera
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Erro: Não foi possível acessar a câmera")
            return False, None, None

        # Ajusta propriedades da câmera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            for attempt in range(attempts):
                print(f"Tentativa {attempt+1}/{attempts}. Olhe para a câmera...")

                start_time = time.time()
                face_found = False
                face_frame = None
                face_location = None

                # Loop para capturar e verificar
                while time.time() - start_time < timeout / attempts:
                    ret, frame = cap.read()
                    if not ret:
                        print("Erro ao capturar imagem da câmera")
                        break

                    # Mostra o frame para o usuário
                    cv2.imshow("Captura de Face - Olhe para a câmera", frame)

                    # Redimensiona para processamento mais rápido
                    small_frame = cv2.resize(
                        frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor
                    )
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Verifica se há um rosto
                    face_locations = face_recognition.face_locations(
                        rgb_small_frame, model="hog"
                    )

                    if face_locations:
                        # Encontrou um rosto! Salva o frame
                        face_found = True
                        face_frame = frame.copy()

                        # Converte a localização para o tamanho original
                        y1, x2, y2, x1 = face_locations[0]
                        scale = 1.0 / self.resize_factor
                        y1 = int(y1 * scale)
                        x1 = int(x1 * scale)
                        y2 = int(y2 * scale)
                        x2 = int(x2 * scale)

                        face_location = (x1, y1, x2, y2)

                        # Desenha retângulo no rosto encontrado
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            "Rosto Detectado",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        cv2.imshow("Captura de Face - Olhe para a câmera", frame)
                        cv2.waitKey(
                            1000
                        )  # Pausa por 1 segundo para mostrar o rosto identificado
                        break

                    # Verifica se o usuário quer sair
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break

                if face_found:
                    break
                else:
                    print("Nenhum rosto detectado. Tente novamente.")

        finally:
            # Libera recursos
            cap.release()
            cv2.destroyAllWindows()

        if face_found:
            return True, face_frame, face_location
        else:
            print("Tempo esgotado. Não foi possível capturar um rosto.")
            return False, None, None

    def authorize_face(
        self, camera_id=0, save_unauthorized=True
    ):
        """
        Captura e verifica se uma face está autorizada

        Args:
            camera_id: ID da câmera a ser usada (0 é a padrão)
            save_unauthorized: Se True, salva imagem de tentativas não autorizadas

        Returns:
            tuple: (autorizado, nome_identificado) ou (False, None) se não autorizado
        """
        # Se não houver nenhuma face autorizada, não faz sentido continuar
        if not os.path.exists(self.encodings_file):
            print("Erro: Nenhuma face autorizada encontrada.")
            return False, None
        
        # Carrega os encodings autorizados
        self.names, self.authorized_encodings = self.load_or_generate_encodings()

        # Primeiro captura uma face
        success, frame, face_location = self.capture_face(camera_id=camera_id)

        if not success:
            return False, None

        # Extraindo a face
        x1, y1, x2, y2 = face_location
        face_image = frame[y1:y2, x1:x2]

        # Convertendo para RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Obtendo o encoding da face capturada
        face_encoding = face_recognition.face_encodings(rgb_face)

        if not face_encoding:
            print("Não foi possível extrair características do rosto capturado.")
            return False, None

        face_encoding = face_encoding[0]

        matches = face_recognition.compare_faces(
            self.authorized_encodings, face_encoding, tolerance=self.tolerance
        )
        face_distances = face_recognition.face_distance(
            self.authorized_encodings, face_encoding
        )

        # Verifica se há qualquer correspondência
        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = self.names[best_match_index]

            print(f"Acesso Autorizado: {name}")

            return True, name

        # Se chegou aqui, não está autorizado
        print("Acesso Negado: Face não autorizada")

        # Desenha no frame original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "DESCONHECIDO",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Acesso Negado", frame)
        cv2.waitKey(2000)  # Mostra por 2 segundos
        cv2.destroyAllWindows()

        # Salva a imagem da tentativa não autorizada
        if save_unauthorized:
            try:
                # Cria a pasta se não existir
                os.makedirs(self.access_logs_folder, exist_ok=True)
                # Adiciona timestamp para evitar sobrescrever arquivos
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = os.path.join(self.access_logs_folder, f"unauthorized_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Imagem da tentativa não autorizada salva como {image_path}")
            except Exception as e:
                print(f"Erro ao salvar imagem não autorizada: {e}")

        return False, None

    def add_authorized_person(self, name, camera_id=0):
        """
        Adiciona uma nova pessoa autorizada capturando sua face

        Args:
            name: Nome da pessoa a ser autorizada
            camera_id: ID da câmera a ser usada

        Returns:
            bool: True se adicionada com sucesso, False caso contrário
        """
        # Captura a face
        success, frame, face_location = self.capture_face(camera_id)

        if not success:
            print("Não foi possível capturar uma face para autorização")
            return False

        try:
            # Primeiro extraímos a parte da face para confirmar que há um rosto válido
            x1, y1, x2, y2 = face_location
            face_image = frame[y1:y2, x1:x2]
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Verificamos se conseguimos gerar um encoding válido
            face_encoding = face_recognition.face_encodings(rgb_face)
            if not face_encoding:
                print(
                    "Erro: Não foi possível extrair características faciais da imagem capturada"
                )
                return False

            # Salva a imagem na pasta de autorizados
            os.makedirs(self.images_folder, exist_ok=True)
            image_path = os.path.join(self.images_folder, f"{name}.jpg")

            # Salva o frame inteiro com o rosto detectado
            cv2.imwrite(image_path, frame)
            print(f"Imagem salva como {image_path}")

            # Força regeneração dos encodings (ignorando o cache)
            images, names = self.load_images()
            valid_names, encodings = self.generate_encodings(images, names)

            # Atualiza as listas em memória
            self.names = valid_names
            self.authorized_encodings = encodings

            # Salva os encodings atualizados
            with open(self.encodings_file, "wb") as f:
                pickle.dump({"names": valid_names, "encodings": encodings}, f)

            print(f"Pessoa {name} adicionada com sucesso à lista de autorizados")
            print(f"Total de pessoas autorizadas: {len(self.names)}")
            return True

        except Exception as e:
            print(f"Erro ao adicionar pessoa autorizada: {e}")
            return False


# Exemplo de uso
if __name__ == "__main__":
    auth_system = FaceAuthorizationSystem()

    # Menu simples
    while True:
        print("\n===== Sistema de Autorização Facial =====")
        print("1. Adicionar pessoa autorizada")
        print("2. Verificar autorização")
        print("3. Sair")

        choice = input("Escolha uma opção: ")

        if choice == "1":
            name = input("Nome da pessoa a autorizar: ")
            auth_system.add_authorized_person(name)

        elif choice == "2":
            auth_system.authorize_face()

        elif choice == "3":
            print("Saindo...")
            break

        else:
            print("Opção inválida!")
