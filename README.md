# FaceAuthorizationSystem

Este é um sistema simples de autorização baseado em reconhecimento facial. Ele utiliza a biblioteca OpenCV para captura de imagens e face_recognition para comparar rostos com uma base de dados de imagens autorizadas.

## Requisitos
Antes de rodar o sistema, é importante que se tenha o conda instalado para carregar as dependências:
```bash
conda config --set channel_priority flexible
conda env create -f environment.yml
conda activate face_recognition
```

## Como Executar
1. Execute o script principal:
```bash
python main.py
```
2. Escolha a opção de adicionar um rosto
3. Após adicionar já pode testar a autorização

## Estrutura do Projeto
```
face_auth_system/
│── authorized_faces/        # Imagens de pessoas autorizadas
│── access_logs/             # Logs de acesso
│── authorized_faces.pkl     # Encodings faciais salvos
│── main.py                  # Arquivo principal
```

## Observações
- O sistema pode registrar imagens de tentativas não autorizadas.

---
**Autor:** Breno Dantas
**Versão:** 1.0
**Licença:** MIT
